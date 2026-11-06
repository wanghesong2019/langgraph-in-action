"""07-agent-eval：Agent 评测——ground-truth 注入 + 复检口径。

场景（中性化）：客服工单分类 Agent，把工单分到 refund / tech / billing / other 四类。
本 demo 的焦点不是"再写一个 Agent"，而是"怎么评测它好不好"：

  1. ground-truth 注入：构造带真值标注的测试集（正常样本 + 异常样本，故意做成类别不平衡）。
  2. 跑 Agent：对每个样本调用分类 Agent，拿到预测标签。
  3. 复检口径：按写死的规则判定对错（标签精确匹配 + 异常样本规则）。
  4. 算 F1：precision / recall / F1 + 混淆矩阵（不是只看 accuracy）。

运行：
  pip install -r requirements.txt
  export DEEPSEEK_API_KEY=sk-xxx
  python eval_agent.py
"""

import os
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ── 被评测对象：单 LLM 工单分类器 ──────────────────────────────

class TicketClass(BaseModel):
    """工单分类结果。other 兜"异常/无关"输入（空文本、乱码、无关话题）。"""
    ticket_type: Literal["refund", "tech", "billing", "other"] = Field(
        description="工单类型：refund=退款 / tech=技术 / billing=账单 / other=其他或异常"
    )


def build_model() -> ChatOpenAI:
    """DeepSeek 走 OpenAI 兼容接口。temperature=0：评测要确定性。"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )


def classify(model: ChatOpenAI, text: str) -> str:
    """分类 Agent：输入工单文本，输出预测标签。"""
    structured = model.with_structured_output(TicketClass, method="function_calling")
    return structured.invoke(text).ticket_type


# ── 测试集 + ground-truth（注入异常样本 + 故意不平衡）──────────

# 每条：(工单文本, 真值标签)。8 条 billing（多数类）+ 少数类各 1 条 + 2 条异常。
# 故意不平衡是为了演示：accuracy 会被多数类"刷高"，只有 F1 能暴露少数类的漏报。
TEST_SET = [
    # —— 多数类 billing（8 条）——
    ("会员扣费了但没到账", "billing"),
    ("优惠券结算时用不了", "billing"),
    ("如何开具电子发票", "billing"),
    ("会员自动续费怎么取消", "billing"),
    ("账单金额和实际不符", "billing"),
    ("发票抬头开错了能重开吗", "billing"),
    ("余额充值一直没到账", "billing"),
    ("会员价格怎么算", "billing"),
    # —— 少数类（各 1 条）——
    ("我要退款，商品有质量问题", "refund"),
    ("密码忘了怎么找回", "tech"),
    # —— 异常样本（2 条）——
    ("", "other"),                        # 空文本
    ("今天天气不错，适合出去玩", "other"),  # 无关话题
]

LABELS = ["refund", "tech", "billing", "other"]


# ── 复检口径：判定对错的标准（写死，别含糊）──────────────────

def judge(pred: str, label: str) -> bool:
    """复检口径：预测 == 真值 算命中。

    异常样本的真值就是 other，所以"异常样本规则"天然落在 pred == label 里：
      - 正常样本被分到别的类 → 漏报（FN）
      - 异常样本被硬分到某个业务类 → 误报（FP，Agent 没识别出"这不是客服工单"）
    """
    return pred == label


# ── 算 F1：precision / recall / F1 + 混淆矩阵 ────────────────

def evaluate(predictions: list[str], labels: list[str]) -> dict:
    """按复检口径逐类统计 TP/FP/FN，算 P/R/F1 + macro F1。"""
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in LABELS}
    for pred, label in zip(predictions, labels):
        if pred == label:
            per_class[pred]["tp"] += 1
        else:
            per_class[pred]["fp"] += 1   # 预测成 pred、真值却是别的 → pred 类的误报
            per_class[label]["fn"] += 1  # 真值是 label、却被预测成别的 → label 类的漏报

    for c in LABELS:
        tp = per_class[c]["tp"]; fp = per_class[c]["fp"]; fn = per_class[c]["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_class[c]["precision"] = p
        per_class[c]["recall"] = r
        per_class[c]["f1"] = 2 * p * r / (p + r) if p + r else 0.0

    macro_f1 = sum(per_class[c]["f1"] for c in LABELS) / len(LABELS)
    return per_class, macro_f1


# ── 退化基线：只会猜多数类，用来暴露 accuracy 的欺骗性 ────────

def majority_baseline(text: str) -> str:
    """退化基线：不管输入是什么，都预测多数类 billing。"""
    return "billing"


def main() -> None:
    model = build_model()
    labels = [label for _, label in TEST_SET]

    # ① 退化基线：一律预测 billing —— accuracy 虚高、macro F1 露馅
    base_preds = [majority_baseline(text) for text, _ in TEST_SET]
    _, base_f1 = evaluate(base_preds, labels)
    base_acc = sum(judge(p, l) for p, l in zip(base_preds, labels)) / len(labels)
    print("== ① 退化基线：不管输入都预测多数类 billing ==")
    print(f"accuracy = {base_acc:.2f} ｜ macro F1 = {base_f1:.2f}   ← accuracy 虚高、F1 露馅\n")

    # ② 真实 LLM 分类器
    predictions = [classify(model, text) for text, _ in TEST_SET]
    per_class, macro_f1 = evaluate(predictions, labels)

    print("== ② 真实 LLM 分类器（ground-truth 注入 + 复检口径）==\n")
    print(f"{'类别':<8}{'精确率':>8}{'召回率':>8}{'F1':>8}")
    for c in LABELS:
        print(f"{c:<8}{per_class[c]['precision']:>8.2f}{per_class[c]['recall']:>8.2f}{per_class[c]['f1']:>8.2f}")
    print(f"\nmacro F1 = {macro_f1:.2f}")

    print("\n== 逐条复检明细 ==")
    for (text, label), pred in zip(TEST_SET, predictions):
        shown = text if text else "(空文本)"
        print(f"[{'✓' if judge(pred, label) else '✗'}] 真值={label:<7} 预测={pred:<7}  {shown}")

    correct = sum(judge(p, l) for p, l in zip(predictions, labels))
    acc = correct / len(labels)
    print(f"\naccuracy = {acc:.2f} ｜ macro F1 = {macro_f1:.2f}")
    print("（对比 ①：accuracy 会被多数类刷高，macro F1 才暴露「只会猜多数类」的退化分类器）")


# ── 实际运行输出（DeepSeek deepseek-chat）────────────────────────────
#
# == ① 退化基线：不管输入都预测多数类 billing ==
# accuracy = 0.67 ｜ macro F1 = 0.20   ← accuracy 虚高、F1 露馅
#
# == ② 真实 LLM 分类器（ground-truth 注入 + 复检口径）==
# 类别      精确率   召回率      F1
# refund      1.00     1.00     1.00
# tech        0.50     1.00     0.67
# billing     1.00     0.88     0.93
# other       1.00     1.00     1.00
# macro F1 = 0.90
#
# == 逐条复检明细 ==
# [✓] 真值=billing 预测=billing  会员扣费了但没到账
# [✗] 真值=billing 预测=tech     优惠券结算时用不了   ← 唯一误判：优惠券被分到 tech
# [✓] 真值=billing 预测=billing  如何开具电子发票
# [✓] 真值=billing 预测=billing  会员自动续费怎么取消
# [✓] 真值=billing 预测=billing  账单金额和实际不符
# [✓] 真值=billing 预测=billing  发票抬头开错了能重开吗
# [✓] 真值=billing 预测=billing  余额充值一直没到账
# [✓] 真值=billing 预测=billing  会员价格怎么算
# [✓] 真值=refund  预测=refund   我要退款，商品有质量问题
# [✓] 真值=tech    预测=tech     密码忘了怎么找回
# [✓] 真值=other   预测=other    (空文本)
# [✓] 真值=other   预测=other    今天天气不错，适合出去玩
#
# accuracy = 0.92 ｜ macro F1 = 0.90
# （对比 ①：accuracy 会被多数类刷高，macro F1 才暴露「只会猜多数类」的退化分类器）


if __name__ == "__main__":
    main()
