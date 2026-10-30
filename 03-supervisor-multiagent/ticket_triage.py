"""03-supervisor-multiagent：多 Agent —— Supervisor 分诊 + 专职 Worker（带 loop-back）。

场景（中性化）：工单客服。supervisor 分诊 → 逐轮派给需要的专职 worker → 全部处理完聚合。

核心演示点：
  - supervisor 维护「待办队列」pending，每个专职处理完出队，控制流回到 supervisor 继续调度下一轮（loop-back）；
  - 每个专职 worker 是「单独 compile 的子图」，有自己的内部步骤。

运行：
  pip install -r requirements.txt
  python ticket_triage.py
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    """顶层共享 state：supervisor 与所有 worker 共用。"""

    ticket: str        # 工单原文
    pending: list[str] # 待处理的专职类别队列（supervisor 据此逐轮调度）
    handled: list[str] # 已处理结果（按顺序累积）
    report: str        # 最终聚合报告


def classify(state: State) -> dict:
    """Supervisor 分诊（只在第一轮执行）：判断工单需要哪些专职（可能多个）。"""
    if state.get("pending") is not None:
        return {}  # 后续轮次：已分诊过，不再改
    ticket = state["ticket"]
    pending = []
    if "退款" in ticket:
        pending.append("refund")
    if "报错" in ticket or "登录" in ticket:
        pending.append("tech")
    if "续费" in ticket or "会员" in ticket:
        pending.append("billing")
    if not pending:
        pending = ["billing"]  # 兜底
    return {"pending": pending, "handled": []}


def route(state: State) -> str:
    """条件路由：待办非空 → 派给队首专职；空了 → 去聚合。"""
    return state["pending"][0] if state["pending"] else "FINISH"


def _make_worker(name: str, action: str):
    """构造一个两步骤的专职 worker 子图。真实项目里这里是该专职自己的 prompt + 工具 + state。"""
    def analyze(state: State) -> dict:
        # 专职第一步：处理（demo 用规则模拟）
        return {"handled": state.get("handled", []) + [f"{name}：{action}"]}

    def done(state: State) -> dict:
        # 专职第二步：出队——把自己从待办队列移除，触发 supervisor 调度下一轮
        return {"pending": state["pending"][1:]}

    g = StateGraph(State)
    g.add_node("analyze", analyze)
    g.add_node("done", done)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "done")
    g.add_edge("done", END)
    return g.compile()


refund_worker = _make_worker("退款专员", "走退款审批")
tech_worker = _make_worker("技术支持", "排查登录报错")
billing_worker = _make_worker("续费顾问", "解答续费规则")


def aggregate(state: State) -> dict:
    """聚合：把各专职的处理结果汇总成最终报告。"""
    return {"report": " ｜ ".join(state["handled"])}


def build_graph():
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("refund_worker", refund_worker)
    g.add_node("tech_worker", tech_worker)
    g.add_node("billing_worker", billing_worker)
    g.add_node("aggregate", aggregate)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        route,
        {
            "refund": "refund_worker",
            "tech": "tech_worker",
            "billing": "billing_worker",
            "FINISH": "aggregate",
        },
    )
    # loop-back：每个专职处理完，回到 supervisor（classify）继续调度下一轮
    g.add_edge("refund_worker", "classify")
    g.add_edge("tech_worker", "classify")
    g.add_edge("billing_worker", "classify")
    g.add_edge("aggregate", END)
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    graph = build_graph()

    tickets = [
        "我要退款，订单号 12345",                # 只需退款
        "登录一直报错，而且我也要退款",            # 退款 + 技术（loop-back 两轮）
        "问一下会员怎么续费",                    # 只需续费
    ]
    for i, ticket in enumerate(tickets):
        config = {"configurable": {"thread_id": f"ticket-{i}"}}
        result = graph.invoke({"ticket": ticket}, config)
        print(f"工单：{ticket}\n  → {result['report']}\n")


if __name__ == "__main__":
    main()
