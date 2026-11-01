"""04-react-vs-executor-critic：两种循环 —— ReAct 工具循环 vs Executor+Critic。

场景（中性化）：一个研究助理，用两种不同的循环干活：
  1. ReAct 循环（查资料）：「思考 → 搜索 → 观察」反复，信息够了再回答。
  2. Executor+Critic 循环（写方案）：「生成 → 评审 → 改写」反复，评审通过才定稿。

（demo 用规则模拟「思考 / 评审」的判断；真实项目里这些是 LLM 调用。）

运行：
  pip install -r requirements.txt
  python research_assistant.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# ── 循环一：ReAct 工具循环（查资料）──────────────────────────────

class ResearchState(TypedDict):
    query: str
    searches: list[str]   # 已做的搜索
    answer: str


def search(state: ResearchState) -> dict:
    """「工具调用」：模拟一次搜索，追加一条结果。"""
    searches = state.get("searches", [])
    n = len(searches) + 1
    return {"searches": searches + [f"搜索{n}：关于「{state['query']}」的信息片段{n}"]}


def should_keep_searching(state: ResearchState) -> str:
    """「思考」：信息够了吗？demo 规则 = 搜满 3 次就够（真实项目是 LLM 判断）。"""
    return "search" if len(state.get("searches", [])) < 3 else "answer"


def answer(state: ResearchState) -> dict:
    """「观察综合」：把搜到的信息汇总成答案。"""
    return {"answer": "综合 " + "；".join(state.get("searches", []))}


def build_research_graph():
    g = StateGraph(ResearchState)
    g.add_node("search", search)
    g.add_node("answer", answer)
    g.add_edge(START, "search")
    # ReAct 的「循环」：search 之后由 should_keep_searching 决定「继续搜」还是「回答」
    g.add_conditional_edges("search", should_keep_searching, {"search": "search", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()


# ── 循环二：Executor + Critic 循环（写方案）─────────────────────

class WritingState(TypedDict):
    draft: str
    reviews: list[str]   # 评审记录
    final: str


def generate(state: WritingState) -> dict:
    """Executor：生成 / 改写方案。第 1 次给初稿，之后按评审意见改一版。"""
    n = len(state.get("reviews", [])) + 1
    return {"draft": "方案初稿" if n == 1 else f"方案修订版 v{n}"}


def critic(state: WritingState) -> dict:
    """Critic：评审。demo 规则 = 改到 v3（评审两轮）才算过。"""
    reviews = state.get("reviews", [])
    n = len(reviews) + 1
    if n >= 3:
        return {"reviews": reviews + ["通过"]}
    return {"reviews": reviews + [f"评审{n}：还不完善，继续改"]}


def should_continue(state: WritingState) -> str:
    """评审通过 → 定稿；否则 → 回去改。"""
    return "finish" if state["reviews"][-1] == "通过" else "generate"


def finish(state: WritingState) -> dict:
    return {"final": f"{state['draft']}（已通过评审）"}


def build_writing_graph():
    g = StateGraph(WritingState)
    g.add_node("generate", generate)
    g.add_node("critic", critic)
    g.add_node("finish", finish)
    g.add_edge(START, "generate")
    g.add_edge("generate", "critic")
    # Executor+Critic 的「循环」：critic 之后由 should_continue 决定「回去改」还是「定稿」
    g.add_conditional_edges("critic", should_continue, {"generate": "generate", "finish": "finish"})
    g.add_edge("finish", END)
    return g.compile()


def main() -> None:
    print("== 循环一：ReAct 工具循环（查资料）==")
    research = build_research_graph()
    r = research.invoke({"query": "LangGraph 是什么"})
    for s in r["searches"]:
        print("  ", s)
    print("  →", r["answer"], "\n")

    print("== 循环二：Executor + Critic 循环（写方案）==")
    writing = build_writing_graph()
    w = writing.invoke({})
    for review in w["reviews"]:
        print("  ", review)
    print("  →", w["final"])


if __name__ == "__main__":
    main()
