"""08-data-flywheel：数据飞轮——eval → 案例回流 → RAG 增强。

场景（中性化）：产品帮助中心的文档问答。本讲把第 6 讲（混合检索 RAG）和第 7 讲（评测）
接成一个闭环，是前两讲的「合体」：

  ① 评测（第 7 讲）：跑测试集，用 LLM-judge 找出答不对的问题（失败案例）
  ② 案例回流：把失败案例（问题 + 标准答案）写成新 FAQ 块，写回检索库（第 6 讲的 DOCS）
  ③ 再评测：同一套测试集再跑一遍，看 accuracy 怎么变

飞轮用 LangGraph 建模成「meta-loop」：评测 →（有失败）→ 回流 → 再评测 →（无失败）→ 结束。
这是和第 1~7 讲里「agent 工具循环」不同的另一种循环——循环体是「评测 + 改进」整个闭环，
而不是单个 agent 的「调工具」。

运行（需要 embedding 服务 + LLM）：
  pip install -r requirements.txt
  export EMBED_BASE_URL=http://127.0.0.1:8011/v1   # 33-server 的 BGE-M3 vLLM
  export EMBED_API_KEY=<共享key>
  export DEEPSEEK_API_KEY=sk-xxx                    # 生成回答 / 打分的 LLM
  python data_flywheel.py
"""

import math
import os
from typing import Literal, TypedDict

import jieba
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi


# ── 检索库（初始 8 条，复用第 6 讲语料）──────────────────────────
# 只有「退款 / 密码 / 发票」三个主题是测试集会问到的；「修改配送地址 / 价格保护」
# 两个主题刻意不在库里，用来制造第 1 轮的失败案例（知识缺口）。

INITIAL_DOCS = [
    "退款政策：如果您对购买的商品或服务不满意，可以在订单页面发起退款申请。客服审核通过后，款项会在三个工作日内原路退回您的支付账户。",
    "会员到账说明：购买会员后，权益通常会在五分钟内自动到账。如果超过半小时仍未到账，可能是支付延迟，请联系在线客服协助核查。",
    "账号密码重置：如果忘记登录密码，可以在登录页面点击忘记密码，通过注册时绑定的邮箱或手机号接收验证码来重置密码。",
    "电子发票开具：订单支付完成后，您可以在订单详情页面申请开具电子发票，发票会发送到您填写的邮箱。",
    "账号注销流程：如需注销账号，请在设置页面提交注销申请，客服会通过电话与您确认身份后再执行注销。",
    "优惠券使用规则：下单结算时，您可以在结算页面选择一张可用的优惠券来抵扣订单金额，部分优惠券有最低消费限制。",
    "物流信息查询：商品发货后，您可以在订单页面查看物流单号，并通过该单号在快递官网查询实时运输轨迹。",
    "客服联系方式：在线客服的工作时间为每天早九点到晚九点，您也可以拨打客服热线 400-123-4567 进行咨询。",
]


# ── 测试集 + ground-truth（标准答案）───────────────────────────
# 前 3 条：检索库能回答（第 1 轮应判对）。
# 后 2 条：检索库没有相关文档（知识缺口），第 1 轮应判错，回流后第 2 轮判对。

TEST_SET = [
    ("怎么退款", "在订单页面发起退款申请，客服审核通过后款项三个工作日内原路退回"),
    ("密码忘了怎么找回", "登录页点击忘记密码，通过绑定邮箱或手机号接收验证码来重置密码"),
    ("如何申请电子发票", "订单支付完成后在订单详情页申请开具，发票会发送到填写的邮箱"),
    ("怎么修改配送地址", "下单后若商品未发货，可在订单页面修改配送地址；已发货需联系客服协助处理"),
    ("如何申请价格保护", "商品购买后 7 天内若价格下调，可申请价格保护，差价退回原支付账户"),
]

MAX_ROUNDS = 3  # 防死循环：最多回流 3 轮（正常情况下 1 轮就能清零失败）


# ── LLM + embedding（复用第 6 / 7 讲的连接方式）────────────────

def build_model() -> ChatOpenAI:
    """DeepSeek 走 OpenAI 兼容接口。temperature=0：回答/打分都要确定性。"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )


def build_embeddings() -> OpenAIEmbeddings:
    """OpenAI 兼容的 embedding 客户端，指向 BGE-M3 的 vLLM 服务。"""
    api_key = os.environ.get("EMBED_API_KEY")
    if not api_key:
        raise ValueError("请先设置 EMBED_API_KEY 环境变量（33-server BGE-M3 vLLM 的共享 key）")
    return OpenAIEmbeddings(
        model=os.environ.get("EMBED_MODEL", "BAAI/bge-m3"),
        base_url=os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8011/v1"),
        api_key=api_key,
    )


# ── 检索三件套（第 6 讲算法，改成参数化：docs 作为入参，因为回流会改库）──

STOPWORDS = {
    "的", "了", "在", "是", "我", "你", "您", "他", "她", "它", "这", "那",
    "和", "与", "或", "对", "把", "被", "如果", "可以", "会", "吗", "呢",
    "如何", "怎么", "什么", "请问",
}

# BGE-M3 官方不加前缀，但短查询实测加前缀更稳，故 opt-in（同第 6 讲）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def jieba_tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip() and w not in STOPWORDS]


def bm25_search(query: str, docs: list[str], k: int = 3) -> list[int]:
    """BM25 关键词检索，返回 top-k 文档下标（入参 docs 支持回流后增长的库）。"""
    bm25 = BM25Okapi([jieba_tokenize(d) for d in docs])
    scores = bm25.get_scores(jieba_tokenize(query))
    if max(scores) <= 0:  # 没有任何词命中 → 诚实返回空
        return []
    return sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def dense_search(query: str, docs: list[str], embeddings: OpenAIEmbeddings, k: int = 3) -> list[int]:
    qv = embeddings.embed_query(BGE_QUERY_INSTRUCTION + query)
    dvs = embeddings.embed_documents(docs)
    sims = [_cos(qv, dv) for dv in dvs]
    return sorted(range(len(docs)), key=lambda i: sims[i], reverse=True)[:k]


def rrf_fuse(*rank_lists: list[int], k: int = 60, top_k: int = 3) -> list[int]:
    """Reciprocal Rank Fusion：score(d) = Σ 1/(k + rank_i(d))。"""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]


_embeddings = build_embeddings()  # 建一次 embedding 客户端（demo 从简，生产要持久化）


def hybrid_retrieve(query: str, docs: list[str], k: int = 3) -> list[str]:
    """BM25 + dense → RRF 混合检索，返回 top-k 文档文本。"""
    b = bm25_search(query, docs, k)
    d = dense_search(query, docs, _embeddings, k)
    return [docs[i] for i in rrf_fuse(b, d, top_k=k)]


# ── ① 被评测对象：有据生成的 RAG 问答（只按检索到的文档回答）────

def rag_qa(query: str, docs: list[str]) -> str:
    """检索 → 有据生成。没有相关信息就明说，不编造（这是「答错」能稳定暴露的前提）。"""
    top = hybrid_retrieve(query, docs, k=3)
    context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(top))
    prompt = (
        "你是产品帮助中心的客服助手。只根据下面检索到的文档回答用户问题；"
        "如果这些文档里没有回答问题所需的信息，就回答「我没有查到相关信息」，不要编造。\n\n"
        f"文档：\n{context}\n\n用户问题：{query}"
    )
    return build_model().invoke(prompt).content


# ── ② 复检口径：LLM-judge 判对错（复用第 7 讲的 correctness）────

class Verdict(BaseModel):
    """候选答案与标准答案的一致性判断。"""
    correct: Literal["yes", "no"] = Field(description="候选答案是否正确：yes=正确 / no=错误")


def judge(query: str, ground_truth: str, candidate: str) -> bool:
    """LLM-judge：候选答案是否回答了问题、与标准答案语义一致。"""
    model = build_model().with_structured_output(Verdict, method="function_calling")
    prompt = (
        f"问题：{query}\n标准答案：{ground_truth}\n候选答案：{candidate}\n"
        "候选答案是否回答了问题、与标准答案语义一致？回答 yes 或 no。"
    )
    return model.invoke(prompt).correct == "yes"


def run_eval(docs: list[str]) -> tuple[float, list[tuple[str, str]]]:
    """跑一遍测试集，返回 (accuracy, 失败案例列表)。失败案例 = (问题, 标准答案)。"""
    correct = 0
    failures: list[tuple[str, str]] = []
    for q, gt in TEST_SET:
        ans = rag_qa(q, docs)
        ok = judge(q, gt, ans)
        correct += ok
        print(f"  [{'✓' if ok else '✗'}] {q}")
        if not ok:
            failures.append((q, gt))
            print(f"       候选答案：{ans[:40]}…")
    return correct / len(TEST_SET), failures


# ── ③ 案例回流：把失败案例写成 FAQ 块写回检索库 ────────────────

def faq_chunk(question: str, answer: str) -> str:
    """失败案例 → 检索库新块。问句本身成了可被 BM25/dense 命中的关键词。"""
    return f"常见问题：{question}\n答：{answer}"


# ── 飞轮：LangGraph 建模成 meta-loop ──────────────────────────

class FlywheelState(TypedDict):
    docs: list[str]                        # 检索库（回流会往里加块）
    round: int                             # 已回流次数
    failures: list[tuple[str, str]]        # 当前轮失败案例
    accuracy: float                        # 当前轮 accuracy


def evaluate_node(state: FlywheelState) -> dict:
    rnd = state["round"] + 1
    print(f"\n== 第 {rnd} 轮评测（检索库 {len(state['docs'])} 条）==")
    accuracy, failures = run_eval(state["docs"])
    print(f"accuracy = {accuracy:.2f}")
    return {"accuracy": accuracy, "failures": failures}


def feedback_node(state: FlywheelState) -> dict:
    n = len(state["failures"])
    print(f"\n== 案例回流：把 {n} 条失败案例写成 FAQ 写回检索库 ==")
    for q, a in state["failures"]:
        print(f"  + 问：{q}  →  答：{a[:24]}…")
    new_docs = state["docs"] + [faq_chunk(q, a) for q, a in state["failures"]]
    return {"docs": new_docs, "round": state["round"] + 1}


def should_continue(state: FlywheelState) -> str:
    """有失败案例且没超上限 → 回流再评；否则结束。"""
    if state["failures"] and state["round"] < MAX_ROUNDS:
        return "feedback"
    return END


def build_graph():
    g = StateGraph(FlywheelState)
    g.add_node("evaluate", evaluate_node)
    g.add_node("feedback", feedback_node)
    g.add_edge(START, "evaluate")
    g.add_conditional_edges("evaluate", should_continue, {"feedback": "feedback", END: END})
    g.add_edge("feedback", "evaluate")   # 回流完回到评测 → 形成飞轮的环
    return g.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke({
        "docs": list(INITIAL_DOCS),
        "round": 0,
        "failures": [],
        "accuracy": 0.0,
    })
    print(f"\n== 飞轮结束：检索库 {len(result['docs'])} 条，最终 accuracy = {result['accuracy']:.2f} ==")


# ── 实际运行输出（33-server：BGE-M3 via vLLM 8011 + DeepSeek deepseek-chat）────
# 注：首次运行 jieba 会打印 "Building prefix dict..." 等词典加载信息，属正常噪声，此处略去。
#
# == 第 1 轮评测（检索库 8 条）==
#   [✓] 怎么退款
#   [✓] 密码忘了怎么找回
#   [✓] 如何申请电子发票
#   [✗] 怎么修改配送地址
#        候选答案：我没有查到相关信息。…
#   [✗] 如何申请价格保护
#        候选答案：我没有查到相关信息。…
# accuracy = 0.60
#
# == 案例回流：把 2 条失败案例写成 FAQ 写回检索库 ==
#   + 问：怎么修改配送地址  →  答：下单后若商品未发货，可在订单页面修改配送地址；已…
#   + 问：如何申请价格保护  →  答：商品购买后 7 天内若价格下调，可申请价格保护，…
#
# == 第 2 轮评测（检索库 10 条）==
#   [✓] 怎么退款
#   [✓] 密码忘了怎么找回
#   [✓] 如何申请电子发票
#   [✓] 怎么修改配送地址
#   [✓] 如何申请价格保护
# accuracy = 1.00
#
# == 飞轮结束：检索库 10 条，最终 accuracy = 1.00 ==
#
# 关键点：两条知识缺口（修改配送地址 / 价格保护）第 1 轮稳定判错——不是因为模型「答错」，
# 而是有据生成让它在「库里没相关内容」时老实说「我没有查到相关信息」。回流把失败案例
# 写成 FAQ 块后，问句本身成了可被 BM25/dense 命中的关键词，第 2 轮全部命中。


if __name__ == "__main__":
    main()
