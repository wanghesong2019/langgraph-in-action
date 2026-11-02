"""05-function-calling-schema：把「交卷」伪装成 tool，用 schema 逼合法 JSON。

场景（中性化）：客服工单信息抽取——从一段自由文本里抽出结构化字段。

对比两种做法：
  1. 朴素：直接让模型「输出 JSON」→ 拿到的是自由文本字符串，要自己 json.loads，易碎。
  2. 结构化：把「交卷」伪装成一次 tool call，用 Pydantic schema 逼模型返回类型正确、字段齐全的对象。

运行：
  pip install -r requirements.txt
  export DEEPSEEK_API_KEY=sk-xxx
  python ticket_extract.py
"""

import json
import os
from typing import Literal, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


# ── 「交卷 tool」的参数 schema（合同）──────────────────────────────

class TicketExtraction(BaseModel):
    """从客服工单文本中抽取的结构化字段。这个类就是「交卷 tool」的参数 schema。"""
    ticket_type: Literal["refund", "tech", "billing"] = Field(description="工单类型：退款/技术/账单")
    priority: Literal["high", "medium", "low"] = Field(description="优先级：高/中/低")
    summary: str = Field(description="一句话摘要，不超过 30 字")
    account: Optional[str] = Field(description="涉及的账号，未提及则为 None")
    refund_needed: bool = Field(description="是否需要退款")


TICKET = (
    "客服你好，我昨天在你们平台买的会员到现在还没到账，账号是 wanghs2019，"
    "钱已经扣了，请尽快帮我看看，不行就给我退款。"
)


def build_model() -> ChatOpenAI:
    """DeepSeek 走 OpenAI 兼容接口。temperature=0：结构化输出要确定性。"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )


def naive_json(model: ChatOpenAI) -> None:
    """朴素做法：直接让模型「输出 JSON」，然后自己解析。"""
    prompt = (
        "把下面工单抽取成 JSON，包含 ticket_type/priority/summary/account/refund_needed：\n"
        + TICKET
    )
    raw = model.invoke(prompt)
    print("== 朴素做法：模型原始输出 ==")
    print(repr(raw.content), "\n")
    try:
        parsed = json.loads(raw.content)
        print("json.loads 成功：", parsed, "\n")
    except json.JSONDecodeError as e:
        print(f"json.loads 失败（这就是坑）：{e}\n")


def structured(model: ChatOpenAI) -> None:
    """结构化输出：把「交卷」伪装成 tool call，schema 逼出类型化对象。"""
    # method="function_calling"：DeepSeek 不支持 OpenAI 的 json_schema 结构化输出
    # （response_format），必须走 tool calling（把 schema 当成「交卷 tool」的参数）。
    structured_model = model.with_structured_output(TicketExtraction, method="function_calling")
    result = structured_model.invoke(TICKET)
    print("== 结构化输出 ==")
    print(f"返回类型：{type(result).__name__}")
    print("内容：", result, "\n")


# ── 组装进 LangGraph：结构化结果直接进 state ──────────────────────

class ExtractState(TypedDict):
    ticket: str
    extraction: TicketExtraction  # 类型化的抽取结果，直接作为 state 字段
    action: str


def extract_node(state: ExtractState) -> dict:
    model = build_model().with_structured_output(TicketExtraction, method="function_calling")
    return {"extraction": model.invoke(state["ticket"])}


def route_node(state: ExtractState) -> dict:
    """按结构化字段决定下一步（demo：refund_needed → 走退款）。"""
    e = state["extraction"]
    return {"action": "退款流程" if e.refund_needed else "普通处理"}


def build_graph():
    g = StateGraph(ExtractState)
    g.add_node("extract", extract_node)
    g.add_node("route", route_node)
    g.add_edge(START, "extract")
    g.add_edge("extract", "route")
    g.add_edge("route", END)
    return g.compile()


def main() -> None:
    model = build_model()
    naive_json(model)
    structured(model)

    print("== 组装进 LangGraph ==")
    graph = build_graph()
    out = graph.invoke({"ticket": TICKET})
    print("抽取结果：", out["extraction"])
    print("路由决策：", out["action"])


# ── 实际运行输出（DeepSeek deepseek-chat，供博客引用）────────────────
#
# == 朴素做法：模型原始输出 ==
# '```json\n{\n  "ticket_type": "未到账",\n  "priority": "高",\n  "summary": "用户昨天购买的会员至今未到账，款项已扣除，要求尽快核查，若无法解决则申请退款。",\n  "account": "wanghs2019",\n  "refund_needed": true\n}\n```'
#   → json.loads 失败：Expecting value: line 1 column 1 (char 0)
#   → 两个坑：① 包了 ```json 围栏；② ticket_type 抽成「未到账」（枚举外，应为 refund）
#
# == 结构化输出 ==
# 返回类型：TicketExtraction
# 内容：ticket_type='refund' priority='high'
#       summary='购买会员未到账，已扣款，要求尽快处理或退款'
#       account='wanghs2019' refund_needed=True
#   → 枚举被校正为 'refund'，bool/str/Literal 类型被保证
#
# == 组装进 LangGraph ==
# 抽取结果：ticket_type='refund' priority='high' summary='会员购买后未到账，已扣款，要求尽快处理或退款' account='wanghs2019' refund_needed=True
# 路由决策：退款流程


if __name__ == "__main__":
    main()
