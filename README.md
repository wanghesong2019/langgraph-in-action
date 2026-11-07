# langgraph-in-action

> LangGraph 实战入门系列：**每个知识点 = 一个可运行 demo + 一篇技术博客 + 一份标注到章节的参考文献**。
> 所有 demo 提炼自真实生产项目踩过的坑（已脱敏），场景中性化，不涉及任何具体公司或内部项目。

## 一句话定位

面向 Agent / LLM 应用工程师的 LangGraph 实战系列——不讲"API 是什么"的抄文档教程，而是**从真实项目里踩过的坑出发**，每个坑配一个能独立跑起来的最小 demo，读者 clone 下来就能复现。

## 差异化（为什么看这个而不是官方文档）

| 官方文档 | 本系列 |
|---|---|
| 讲"API 是什么" | 讲"真实项目里为什么这么用、踩了什么坑、怎么防" |
| 单点知识 | 每个点 = demo + 博客 + 参考文献（**具体到章节**） |
| 不关心生产 | 每个 demo 都带"诚实边界 + 生产化延伸" |

## 系列规划（8 篇，按入门 → 进阶）

| # | 知识点 | demo 场景（中性化） | 状态 |
|---|---|---|---|
| 1 | LangGraph 入门：把 Agent 建模成有状态图 | 文档处理流水线 + 条件路由 + checkpointer | ✅ 完成（demo + 博客 + 8 篇参考文献） |
| 2 | HITL：高危操作人工审批 | 客服工单 Agent：退款/删除需 approve | ✅ 完成（demo + 博客 + 7 篇参考文献） |
| 3 | 多 Agent：Supervisor 分诊 + 专职 Worker | 工单分类 → 三个专职处理 → 聚合 | ✅ 完成（demo + 博客 + 6 篇参考文献） |
| 4 | 两种循环：ReAct 工具循环 vs Executor+Critic | 查资料(工具循环) vs 写方案(生成-评审-重写) | ✅ 完成（demo + 博客 + 7 篇参考文献） |
| 5 | Function Calling 做强制结构化输出 | 把"交卷"伪装成 tool，用 schema 逼合法 JSON | ✅ 完成（demo + 博客 + 5 篇参考文献） |
| 6 | RAG 检索优化：分词坑 → 混合检索 → 生产化落地 | 文档问答检索（Recall 0.00→1.00 的分词坑） | ✅ 完成（demo + 博客 + 7 篇参考文献） |
| 7 | Agent 评测：ground-truth 注入 + 复检口径 | 注入异常 → 跑 Agent → 复检 → 算 F1 | ✅ 完成（demo + 博客 + 3 篇参考文献） |
| 8 | 数据飞轮：eval → 案例回流 → RAG 增强 | 跑完评测把失败案例回流进检索库 | ✅ 完成（demo + 博客 + 5 篇参考文献） |

> 候选扩展（先不做，看前 4 篇反馈）：工具安全（SafeSQL 只读护栏）、可观测（Span 树 + token 归因）。

## 你可以学到什么

- **有状态 Agent 建模**：把 LLM 应用建模成 LangGraph 图——节点 / 边 / 条件路由 / checkpointer（第 1 讲）。
- **HITL 人工审批**：高危操作在流程里挂一个人工 approve 节点，模型只提方案、人来拍板（第 2 讲）。
- **多 Agent 编排**：Supervisor 分诊 + 专职 Worker 各司其职，分诊-处理-聚合（第 3 讲）。
- **两种循环范式**：ReAct 工具循环 vs Executor+Critic 生成-评审-重写（第 4 讲）。
- **强制结构化输出**：用 Function Calling + Pydantic schema 逼 LLM 交合法 JSON（第 5 讲）。
- **RAG 检索优化**：中文分词坑 → BM25+dense 混合检索 → RRF 融合，召回 0.00→1.00（第 6 讲）。
- **Agent 评测**：ground-truth 注入 + 复检口径 + macro F1 + LLM-judge（第 7 讲）。
- **数据飞轮**：把评测失败案例回流进检索库，让系统越用越准（第 8 讲）。

## 项目结构

```
langgraph-in-action/
├── README.md              # 本文件：系列导航 + 项目结构
├── 01-state-graph/        # 第 1 讲
│   ├── text_summary.py    # 可运行 demo（按业务主题命名，而非 demo.py）
│   ├── requirements.txt   # 钉死 langgraph 版本
│   └── README.md          # 这篇的博客全文（md）
├── 02-hitl/               # 第 2 讲（demo: refund_approval.py）
├── 03-supervisor-multiagent/
├── 04-react-vs-executor-critic/
├── 05-function-calling-schema/
├── 06-rag-hybrid-retrieval/
├── 07-agent-eval/
└── 08-data-flywheel/
```

## 快速开始

每篇目录独立可运行。以 02-hitl 为例：

```bash
cd 02-hitl
pip install -r requirements.txt   # 钉死 langgraph 版本，保证可复现
python refund_approval.py
```

