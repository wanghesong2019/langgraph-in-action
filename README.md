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
| 1 | LangGraph 入门：把 Agent 建模成有状态图 | 3 节点流水线 + 条件路由 + checkpointer | 待写 |
| 2 | HITL：高危操作人工审批 | 客服工单 Agent：退款/删除需 approve | ✅ 完成（demo + 博客 + 7 篇参考文献） |
| 3 | 多 Agent：Supervisor 分诊 + 专职 Worker | 工单分类 → 三个专职处理 → 聚合 | 待写 |
| 4 | 两种循环：ReAct 工具循环 vs Executor+Critic | 查资料(工具循环) vs 写方案(生成-评审-重写) | 待写 |
| 5 | Function Calling 做强制结构化输出 | 把"交卷"伪装成 tool，用 schema 逼合法 JSON | 待写 |
| 6 | RAG 混合检索：dense+BM25→RRF + 中文分词坑 | 文档问答检索（Recall 0.3→1.0 的分词坑） | 待写 |
| 7 | Agent 评测：ground-truth 注入 + 复检口径 | 注入异常 → 跑 Agent → 复检 → 算 F1 | 待写 |
| 8 | 数据飞轮：eval → 案例回流 → RAG 增强 | 跑完评测把失败案例回流进检索库 | 待写 |

> 候选扩展（先不做，看前 4 篇反馈）：工具安全（SafeSQL 只读护栏）、可观测（Span 树 + token 归因）。

## 平台选择（每篇博客最多发 2 处）

- **主阵地：自有博客**（GitHub Pages + VitePress，后续配域名）—— SEO 主阵地、长期技术资产、个人品牌沉淀。
- **分发：掘金**（juejin.cn）—— 中文 AI 社区流量 + 百度/谷歌可收录 + 导流回 GitHub。
- **GitHub 仓库不占"发布地"额度**——它是代码 + md 的家，两个发布平台都从它引流。
- 公众号**暂不做**：封闭生态，外部搜索引擎搜不到，与"搜索引擎可搜"的目标互斥。

## 项目结构

```
langgraph-in-action/
├── README.md              # 本文件：系列导航 + 平台 + 结构
├── 01-state-graph/        # 每篇一个目录
│   ├── demo.py            # 可运行 demo
│   ├── requirements.txt   # 钉死 langgraph 版本
│   └── README.md          # 这篇的博客全文（md）
├── 02-hitl/
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
python demo.py
```

## 维护原则

1. **钉死版本**：每个 demo 的 `requirements.txt` 锁 `langgraph==x.y.z`，README 写清基于哪个版本，避免 API 迭代后跑不起来。
2. **先跑通 1-2 篇，再铺开**：不一次承诺全部 8 篇；先把第 1、2 篇的"demo + 博客 + 双平台发布"完整流水线跑通。
3. **诚实边界**：每个 demo 都标注"生产该怎么做"，不吹满。

## 当前进度

- [x] 系列规划 + 平台选择 + 项目结构（本 README）
- [x] 02-hitl：完整完成（demo 可运行 + 博客 §1-§8 + 参考文献 7 篇）
- [ ] 01-state-graph：demo 代码 + 参考文献 + 大纲
