"""02-hitl：HITL 最小可运行示例 —— 高危操作需人工审批。

机制（从真实生产项目提炼，已脱敏）：
  risk_level → requires_approval(派生属性) → 条件路由 → interrupt 暂停 → Command(resume) 恢复

运行：
  pip install -r requirements.txt
  python refund_approval.py

预期输出：
  ⏸  暂停，待审批： [{'name': '给用户退款 ¥199', 'risk_level': 'high'}]
  审批结果： [{'name': '给用户退款 ¥199', 'decision': 'approved'}]
"""

from dataclasses import dataclass
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver  # LangGraph 1.x 也可用 InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


@dataclass
class Action:
    """一个待执行动作 + 风险等级。"""

    name: str
    risk_level: str = "low"  # low / high

    @property
    def requires_approval(self) -> bool:
        """派生属性 = 单一真相源：risk_level 是唯一事实，由它算出，避免两字段打架。"""
        return self.risk_level == "high"


class State(TypedDict):
    actions: list[dict]      # 存 dict（可被 checkpointer 干净序列化；自定义对象进 state 会有 msgpack 警告）
    approvals: list[dict]    # 人工审批结果


def propose(state: State) -> dict:
    """产出带风险分级的动作清单。demo 硬编码；真实场景这里换成 LLM 生成 + 按 prompt 规则填 risk_level。"""
    actions = [
        Action(name="记一条内部日志", risk_level="low"),
        Action(name="给用户退款 ¥199", risk_level="high"),  # 不可逆（涉及钱）→ 需审批
        Action(name="更新工单备注", risk_level="low"),
    ]
    # 只把字段进 state（requires_approval 是派生属性，不落盘）
    return {"actions": [{"name": a.name, "risk_level": a.risk_level} for a in actions]}


def route(state: State) -> str:
    """条件路由：只要有一条高危，就拐进 approve；否则直接收尾（低危不打断，避免审批疲劳）。"""
    return "approve" if any(a["risk_level"] == "high" for a in state["actions"]) else "report"


def approve(state: State) -> dict:
    """审批节点：把高危动作吐给外部，interrupt 暂停等人。"""
    pending = [a for a in state["actions"] if a["risk_level"] == "high"]
    # interrupt 的「两个方向」：参数 = 节点→调用者（吐待审批清单）；返回值 = 调用者→节点（= Command(resume=X) 的 X）
    decision = interrupt({"pending": pending})
    approvals = []
    for i, a in enumerate(pending):
        d = decision.get(str(i)) if isinstance(decision, dict) else None  # 防御：非 dict 降级 pending
        approvals.append({"name": a["name"], "decision": d or "pending"})
    return {"approvals": approvals}


def report(state: State) -> dict:
    """收尾：打印最终审批结果。"""
    print("审批结果：", state.get("approvals"))
    return {}


def build_graph():
    g = StateGraph(State)
    g.add_node("propose", propose)
    g.add_node("approve", approve)
    g.add_node("report", report)
    g.add_edge(START, "propose")
    g.add_conditional_edges("propose", route, {"approve": "approve", "report": "report"})
    g.add_edge("approve", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=MemorySaver())  # checkpointer 必须，否则 resume 不了


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-1"}}  # 两次 invoke 用同一个 thread_id

    # 第一次 invoke：跑到 interrupt 暂停
    result = graph.invoke({"actions": [], "approvals": []}, config)
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        pending = interrupts[0].value["pending"]  # Interrupt 对象的 .value = interrupt() 的载荷
        print("⏸  暂停，待审批：", pending)
    else:
        print("⚠ 未触发审批（本 demo 必有高危动作，不应走到这里）")

    # 第二次 invoke：模拟人工 approve 第 0 条（即「给用户退款 ¥199」）；报告在 report 节点打印
    graph.invoke(Command(resume={"0": "approved"}), config)


if __name__ == "__main__":
    main()
