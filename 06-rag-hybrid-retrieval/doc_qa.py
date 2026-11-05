"""06-rag-hybrid-retrieval：RAG 混合检索——dense + BM25 → RRF（优化进阶版）。

场景（中性化）：产品帮助中心的文档问答检索——给定用户问题，从文档库召回最相关文档。

优化进阶四步（每步只改一个组件，看 Recall@3 怎么变）：
  0. baseline：BM25 + 按空格切分（中文没空格 → 整句一个 token）→ 召回差
  1. 优化①：BM25 + jieba 中文分词 → 修掉「中文分词坑」（Recall 0.00 → 0.80）
  2. 优化②：加 dense（BGE-M3 embedding）→ 语义召回（关键词漏掉的同义词能抓到）
  3. 优化③：RRF 融合 dense + BM25 → 混合检索（两种信号互补）

运行（dense 部分需要 embedding 服务）：
  pip install -r requirements.txt
  export EMBED_BASE_URL=http://127.0.0.1:8011/v1   # 33-server 的 BGE-M3 vLLM
  export EMBED_API_KEY=<共享key>
  python doc_qa.py
"""

import math
import os
from typing import Callable

import jieba
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi


# ── 语料 + 标注（每个查询标注了它真正相关的文档下标）────────────────

DOCS = [
    "退款政策：如果您对购买的商品或服务不满意，可以在订单页面发起退款申请。客服审核通过后，款项会在三个工作日内原路退回您的支付账户。",
    "会员到账说明：购买会员后，权益通常会在五分钟内自动到账。如果超过半小时仍未到账，可能是支付延迟，请联系在线客服协助核查。",
    "账号密码重置：如果忘记登录密码，可以在登录页面点击忘记密码，通过注册时绑定的邮箱或手机号接收验证码来重置密码。",
    "电子发票开具：订单支付完成后，您可以在订单详情页面申请开具电子发票，发票会发送到您填写的邮箱。",
    "账号注销流程：如需注销账号，请在设置页面提交注销申请，客服会通过电话与您确认身份后再执行注销。",
    "优惠券使用规则：下单结算时，您可以在结算页面选择一张可用的优惠券来抵扣订单金额，部分优惠券有最低消费限制。",
    "物流信息查询：商品发货后，您可以在订单页面查看物流单号，并通过该单号在快递官网查询实时运输轨迹。",
    "客服联系方式：在线客服的工作时间为每天早九点到晚九点，您也可以拨打客服热线 400-123-4567 进行咨询。",
]

QUERIES = [
    ("怎么退款", [0]),
    ("会员没到账怎么办", [1]),
    ("密码忘了怎么找回", [2]),
    ("如何拿回我的钱", [0]),   # 「拿回钱」vs「退款」：无共享字/词 → BM25 漏、dense 语义抓到
    ("优惠券能用吗", [5]),
]


# ── 分词器（优化① 改的就是这个组件）───────────────────────────────

# 常见停用词（噪声词）：不过滤的话，"的/我/如何"这类词会让 BM25 假命中
STOPWORDS = {
    "的", "了", "在", "是", "我", "你", "您", "他", "她", "它", "这", "那",
    "和", "与", "或", "对", "把", "被", "如果", "可以", "会", "吗", "呢",
    "如何", "怎么", "什么", "请问",
}


def space_tokenize(text: str) -> list[str]:
    """baseline 的错误分词：按空格切。中文没空格 → 整句一个 token。"""
    return text.split()


def jieba_tokenize(text: str) -> list[str]:
    """优化①：jieba 中文分词 + 过滤停用词。"""
    return [w for w in jieba.cut(text) if w.strip() and w not in STOPWORDS]


# ── BM25 关键词检索 ─────────────────────────────────────────────

def bm25_search(query: str, tokenize: Callable, k: int = 3) -> list[int]:
    """返回与 query 最相关的 top-k 个文档下标。"""
    tokenized_docs = [tokenize(d) for d in DOCS]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenize(query))
    if max(scores) <= 0:  # 没有任何词命中 → 诚实返回空，而不是靠稳定排序硬凑 top-k
        return []
    ranked = sorted(range(len(DOCS)), key=lambda i: scores[i], reverse=True)
    return ranked[:k]


# ── dense 语义检索（优化② 加的组件，BGE-M3 embedding）─────────────

# BGE-M3 官方不加前缀（"trained without query prompt"），但短查询实测加前缀更稳，故 opt-in
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def build_embeddings() -> OpenAIEmbeddings:
    """OpenAI 兼容的 embedding 客户端，指向 BGE-M3 的 vLLM 服务。"""
    api_key = os.environ.get("EMBED_API_KEY")
    if not api_key:  # 新版 openai SDK 不接受空 api_key，缺 key 时给出清晰报错而非构造时抛异常
        raise ValueError("请先设置 EMBED_API_KEY 环境变量（33-server BGE-M3 vLLM 的共享 key）")
    return OpenAIEmbeddings(
        model=os.environ.get("EMBED_MODEL", "BAAI/bge-m3"),
        base_url=os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8011/v1"),
        api_key=api_key,
    )


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def dense_search(query: str, embeddings: OpenAIEmbeddings, k: int = 3) -> list[int]:
    """向量余弦相似度检索 top-k。

    注意：BGE-M3 官方不强制 query 指令前缀（"trained without query prompt"），
    但本 demo 短查询实测加前缀才排对（"拿回钱"余弦：退款 0.32→0.68、物流 0.42→0.36），
    故 opt-in 加上（文档侧不加）。
    """
    qv = embeddings.embed_query(BGE_QUERY_INSTRUCTION + query)
    dvs = embeddings.embed_documents(DOCS)
    sims = [_cos(qv, dv) for dv in dvs]
    ranked = sorted(range(len(DOCS)), key=lambda i: sims[i], reverse=True)
    return ranked[:k]


# ── RRF 融合（优化③ 改的组件）────────────────────────────────────

def rrf_fuse(*rank_lists: list[int], k: int = 60, top_k: int = 3) -> list[int]:
    """Reciprocal Rank Fusion：score(d) = Σ 1/(k + rank_i(d))。"""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]


# ── 评测：Recall@k ──────────────────────────────────────────────

def recall_at_k(retriever: Callable, queries: list, k: int = 3) -> float:
    hits = 0
    for q, rel in queries:
        retrieved = retriever(q, k=k)
        if any(r in rel for r in retrieved):
            hits += 1
    return hits / len(queries)


def main() -> None:
    print(f"语料文档数：{len(DOCS)} ｜ 标注查询数：{len(QUERIES)}\n")

    base = recall_at_k(lambda q, k: bm25_search(q, space_tokenize, k), QUERIES)
    print(f"baseline（BM25 + 按空格切）    Recall@3 = {base:.2f}")

    seg = recall_at_k(lambda q, k: bm25_search(q, jieba_tokenize, k), QUERIES)
    print(f"优化①（BM25 + jieba 分词）    Recall@3 = {seg:.2f}")

    embeddings = build_embeddings()
    dense = recall_at_k(lambda q, k: dense_search(q, embeddings, k), QUERIES)
    print(f"优化②（dense / BGE-M3）        Recall@3 = {dense:.2f}")

    def hybrid(q: str, k: int = 3) -> list[int]:
        b = bm25_search(q, jieba_tokenize, k)
        d = dense_search(q, embeddings, k)
        return rrf_fuse(b, d, top_k=k)

    hyb = recall_at_k(hybrid, QUERIES)
    print(f"优化③（RRF 融合 dense+BM25）  Recall@3 = {hyb:.2f}")


# ── 实际运行输出（BGE-M3 via 33-server vLLM，供博客引用）────────────────
#
# baseline（BM25 + 按空格切）    Recall@3 = 0.00   ← 没分词：整句一个 token，词无法匹配
# 优化①（BM25 + jieba 分词）    Recall@3 = 0.80   ← 漏「如何拿回我的钱」→ 退款（无共享词）
# 优化②（dense / BGE-M3）        Recall@3 = 0.60   ← 漏「密码忘了」「优惠券」（相似短文档上塌缩）
# 优化③（RRF 融合 dense+BM25）  Recall@3 = 1.00   ← 互补：BM25 补关键词、dense 补同义词
#
# 逐条 top-3（相关文档下标在结果里即命中）：
#   「怎么退款」        bm25=[0,1,2]  dense=[1,0,3]   相关=0  → 都命中
#   「会员没到账怎么办」 bm25=[1,0,2]  dense=[1,0,3]   相关=1  → 都命中
#   「密码忘了怎么找回」 bm25=[2,0,1]  dense=[1,0,3]   相关=2  → bm25 命中、dense 漏
#   「如何拿回我的钱」  bm25=[]       dense=[1,0,3]   相关=0  → dense 命中、bm25 漏
#   「优惠券能用吗」    bm25=[5,0,1]  dense=[1,0,3]   相关=5  → bm25 命中、dense 漏
#
# 关键点：dense 在 8 条语义相近的短文档上「塌缩」到 [1,0,3]（会员/退款/发票），
# 漏掉「密码忘了」「优惠券」；bm25 漏「拿回钱」（无共享词）。两者漏的不是同一条，
# 所以 RRF 融合后 5/5 全中。


if __name__ == "__main__":
    main()
