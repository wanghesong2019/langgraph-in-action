"""01-state-graph：LangGraph 入门 —— 把 Agent 建模成有状态图。

这一篇只讲 LangGraph 最核心的四件套，不引入 LLM / 工具 / HITL：
  State(有状态) → Node(节点函数) → Edge(固定/条件路由) → Checkpointer(持久化)

场景（中性化，从真实项目踩坑提炼）：
  文档处理流水线：analyze 分析 →（长文本才走）summarize 摘要 → report 汇总

运行：
  pip install -r requirements.txt
  python text_summary.py
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver  # 1.x 也可用 InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    """节点间流动的共享状态。节点读它、返回部分字段更新它，框架负责合并。"""

    text: str
    word_count: int
    needs_summary: bool
    summary: str
    report: str


def analyze(state: State) -> dict:
    """节点 1：分析输入。返回 state 的「部分更新」，框架自动合并进原 state。"""
    text = state["text"]
    word_count = len(text)
    # 规则判断（真实项目换成 LLM）：>50 字视为长文本，才需要摘要
    return {"word_count": word_count, "needs_summary": word_count > 50}


def route(state: State) -> str:
    """条件边：读 state 决定拐向哪个节点。返回「下一个节点名」。"""
    return "summarize" if state["needs_summary"] else "report"


def summarize(state: State) -> dict:
    """节点 2：只被长文本路由进来。demo 用规则模拟摘要，真实项目换成 LLM。"""
    text = state["text"]
    return {"summary": text[:50] + ("…" if len(text) > 50 else "")}


def report(state: State) -> dict:
    """节点 3：收尾，汇总输出。"""
    summary = state.get("summary", "(无需摘要)")
    return {"report": f"字数 {state['word_count']}｜摘要：{summary}"}


def build_graph():
    g = StateGraph(State)
    g.add_node("analyze", analyze)
    g.add_node("summarize", summarize)
    g.add_node("report", report)

    g.add_edge(START, "analyze")
    # 条件路由：analyze 之后由 route() 决定去 summarize 还是直接 report
    g.add_conditional_edges("analyze", route, {"summarize": "summarize", "report": "report"})
    g.add_edge("summarize", "report")
    g.add_edge("report", END)

    # checkpointer：每步 state 落盘，配 thread_id 才能按会话存取/回溯
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    graph = build_graph()

    # ① 长文本（>50 字）→ 走 summarize
    long_text = "这是一篇很长的文档，" * 10
    r1 = graph.invoke({"text": long_text}, {"configurable": {"thread_id": "doc-long"}})
    print("① 长文本 →", r1["report"])

    # ② 短文本（不同 thread）→ 跳过 summarize，直接 report
    r2 = graph.invoke({"text": "短文本"}, {"configurable": {"thread_id": "doc-short"}})
    print("② 短文本 →", r2["report"])

    # ③ checkpointer 把 state 落盘，随时可取回（=「有状态」的直观体现）
    snap = graph.get_state({"configurable": {"thread_id": "doc-long"}})
    print("③ 取回 doc-long 的最后 state：", snap.values)


if __name__ == "__main__":
    main()
