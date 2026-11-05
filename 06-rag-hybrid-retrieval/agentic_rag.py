"""06-rag-hybrid-retrieval / agentic_rag.py：把 §5 的混合检索器组装进 LangGraph——agentic RAG 最小实现。

场景（中性化）：产品帮助中心问答。用户问一个问题，agent 自己决定「要不要查文档、查什么、
查几次」——检索不再是 §1–§6 里那条固定的「查询 → 检索 → 生成」直线，而是 agent 循环里的一个工具。

三个部件（对应博客 §7 的 ①②③）：
  ① retrieve：把 BM25 + dense → RRF 的混合检索包成 @tool（检索从「固定步骤」变成「可调用工具」）。
  ② generate_query_or_respond：路由节点，模型用 tool calling 决定「检索」还是「直接回答」，形成检索的环。
  ③ grade_documents：给检索结果逐条打 relevant/not relevant，补「召回对不对」（precision）。

运行（需要 embedding 服务 + LLM）：
  pip install -r requirements.txt
  export EMBED_BASE_URL=http://127.0.0.1:8011/v1   # 33-server 的 BGE-M3 vLLM
  export EMBED_API_KEY=<共享key>
  export DEEPSEEK_API_KEY=sk-xxx                    # 生成/评分的 LLM
  python agentic_rag.py
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# 复用 doc_qa 的检索三件套 + 语料（BM25 / dense / RRF / DOCS）
from doc_qa import DOCS, bm25_search, build_embeddings, dense_search, jieba_tokenize, rrf_fuse


# ── LLM（生成回答 / 评分共用）───────────────────────────────────

def build_model() -> ChatOpenAI:
    """DeepSeek 走 OpenAI 兼容接口。temperature=0：路由/评分要确定性。"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )


# ── ① 检索变工具：混合检索包成 @tool ─────────────────────────────

_embeddings = build_embeddings()  # 建一次 embedding 客户端（§6 提过生产要持久化，demo 从简）


def hybrid_search(query: str, k: int = 3) -> str:
    """BM25 + dense → RRF 混合检索，返回 top-k 文档文本（复用 doc_qa）。"""
    b = bm25_search(query, jieba_tokenize, k)
    d = dense_search(query, _embeddings, k)
    fused = rrf_fuse(b, d, top_k=k)
    return "\n\n".join(f"[{i}] {DOCS[i]}" for i in fused)


@tool
def retrieve(query: str) -> str:
    """检索产品帮助中心文档，返回与 query 最相关的文档片段。"""
    return hybrid_search(query)


# ── ② 路由节点：模型决定「检索」还是「直接回答」────────────────────

SYSTEM_PROMPT = (
    "你是产品帮助中心的客服助手。回答用户问题时："
    "如果仅凭常识就能回答，直接回答；"
    "如果需要查具体政策/流程（退款、会员到账、发票、优惠券等），"
    "调用 retrieve 工具检索文档后再回答。"
)


def generate_query_or_respond(state: MessagesState) -> dict:
    """模型用 tool calling 决定：调用 retrieve 检索，还是直接回答。"""
    model = build_model().bind_tools([retrieve])
    response = model.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def route(state: MessagesState) -> str:
    """有 tool_calls → 去检索；没有 → 结束（模型已直接给出回答）。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "retrieve"
    return END


def build_graph():
    g = StateGraph(MessagesState)
    g.add_node("generate_query_or_respond", generate_query_or_respond)
    g.add_node("retrieve", ToolNode([retrieve]))          # 执行 retrieve 工具，结果作为 ToolMessage
    g.add_conditional_edges(
        "generate_query_or_respond",
        route,
        {"retrieve": "retrieve", END: END},
    )
    g.add_edge("retrieve", "generate_query_or_respond")    # 检索完回到路由节点 → 形成环
    g.add_edge(START, "generate_query_or_respond")
    return g.compile()


# ── ③ 评分：给检索结果打 relevant/not relevant ───────────────────
# 注：最小实现只演示「打分」这一步，没接回环；完整 corrective RAG 会把
# 「全不相关 → 改写 query 再查」这条边补上。

class GradeDocuments(BaseModel):
    """单条文档与问题的相关性判断。"""
    binary_score: str = Field(description="文档是否与问题相关：'yes' 或 'no'")


def grade_documents(question: str, docs: str) -> list[str]:
    """对检索回来的每条文档打 relevant/not relevant（补「召回对不对」= precision）。"""
    model = build_model().with_structured_output(GradeDocuments, method="function_calling")
    scores = []
    for doc in docs.split("\n\n"):   # hybrid_search 返回的文档之间用空行分隔
        r = model.invoke(f"问题：{question}\n文档：{doc}\n这条文档是否与问题相关？")
        scores.append(r.binary_score)
    return scores


# ── main ────────────────────────────────────────────────────────

def main() -> None:
    question = "会员没到账怎么办"

    print("== agentic RAG：agent 自己决定要不要检索、查什么 ==")
    graph = build_graph()
    out = graph.invoke({"messages": [HumanMessage(question)]})
    retrieved = ""
    for m in out["messages"]:
        if m.type == "ai" and getattr(m, "tool_calls", None):
            print(f"[agent 决定检索] query = {m.tool_calls[0]['args']['query']}")
        elif m.type == "tool":
            retrieved = m.content          # 记下这次实际检索回来的文档
            print(f"[检索结果]\n{m.content}\n")
        elif m.type == "ai":
            print(f"[最终回答]\n{m.content}\n")

    print("== ③ grade_documents：给上面的检索结果逐条打分 ==")
    for doc, score in zip(retrieved.split("\n\n"), grade_documents(question, retrieved)):
        print(f"[{score}] {doc[:24]}…")


# ── 实际运行输出（33-server：BGE-M3 via vLLM 8011 + DeepSeek deepseek-chat）────
# 注：首次运行 jieba 会打印 "Building prefix dict..." 等词典加载信息，属正常噪声，此处略去。
#
# == agentic RAG：agent 自己决定要不要检索、查什么 ==
# [agent 决定检索] query = 会员购买后未到账怎么办     ← 模型改写了 query（原问：会员没到账怎么办）
# [检索结果]
# [1] 会员到账说明：购买会员后，权益通常会在五分钟内自动到账。如果超过半小时仍未到账，可能是支付延迟，请联系在线客服协助核查。
#
# [0] 退款政策：如果您对购买的商品或服务不满意，可以在订单页面发起退款申请。客服审核通过后，款项会在三个工作日内原路退回您的支付账户。
#
# [7] 客服联系方式：在线客服的工作时间为每天早九点到晚九点，您也可以拨打客服热线 400-123-4567 进行咨询。
# [最终回答]
# 会员购买后一般会在 5 分钟内自动到账。如果超过半小时还没到账，可能是支付延迟，建议您：
# 1. 先确认支付是否成功（查看支付渠道的扣款记录）；
# 2. 若已扣款但权益未到账，请联系在线客服协助核查（在线客服工作时间：每天 9:00–21:00），
#    也可拨打客服热线 400-123-4567。
# 联系时建议提供订单号和支付凭证，方便客服快速定位处理。
#
# == ③ grade_documents：给上面的检索结果逐条打分 ==
# [yes] [1] 会员到账说明：购买会员后，权益通常会在五…
# [no]  [0] 退款政策：如果您对购买的商品或服务不满意…
# [yes] [7] 客服联系方式：在线客服的工作时间为每天早…


if __name__ == "__main__":
    main()
