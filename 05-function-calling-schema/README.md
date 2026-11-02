# 第 5 讲：Function Calling 做强制结构化输出——把「交卷」伪装成 tool

> 本系列用可运行的最小 demo 讲 LangGraph 实战。这是第 5 讲：让 LLM 稳定输出 JSON 的可靠做法——不是让它"输出 JSON"，而是把"交卷"伪装成一次 tool call，用 Pydantic schema 当合同逼它返回合法、类型正确的对象。配套 demo 见同目录 `ticket_extract.py`。

**TL;DR**：直接让 LLM"输出 JSON"不可靠（markdown 代码围栏、废话、漏字段、类型漂移）。可靠做法是把"交卷"伪装成 tool call——`model.with_structured_output(Schema)` 一行，Pydantic schema 就是合同，返回类型化对象，零手工解析。

---

## 正文

### §1 为什么「输出 JSON」不可靠

朴素"输出 JSON"不可靠——拿到的是**字符串**，`json.loads` 只保证"语法合法"、不保证"语义正确"，字段、类型、枚举全无合同约束。

**① 拿到的是字符串，解析一步就脆。** 模型输出是自由文本，要 `json.loads` 才能变 `dict`。这一步常常直接崩：模型爱给 JSON 套一层 markdown 代码围栏（三反引号 `json` 那种）或加几句废话，导致字符串首字符不是 `{` 或 `[`，`json.loads` 立刻抛 `Expecting value: line 1 column 1`——demo 里就是这么挂的（首字符是围栏的反引号）。模型的原样输出长这样：

````text
```json
{
  "ticket_type": "未到账",
  "priority": "高",
  "summary": "用户昨天购买的会员至今未到账，款项已扣除，要求尽快核查，若无法解决则申请退款。",
  "account": "wanghs2019",
  "refund_needed": true
}
```
````

想抢救，就得自己 strip 围栏、猜边界，全是手工活。

**② `json.loads` 过了，数据也未必能用。** 它只校验 JSON 的**语法**，不校验**语义**——字段名、类型、枚举一概不管。demo 里 `ticket_type` 被抽成「未到账」（枚举外）、`priority` 被抽成「高」（不是 high/medium/low），这些值 `json.loads` 毫无意见，却是下游没法用的脏数据。

一句话：朴素拿到的 `dict` 是"信不过"的——它缺一道 **schema 校验**的闸门，这正是下一节要引入的"合同"。

### §2 schema 就是合同：用 Pydantic 定义输出形状

"合同"长这样——一个 `TicketExtraction` 类：

```python
class TicketExtraction(BaseModel):
    """从客服工单文本中抽取的结构化字段。这个类就是「交卷 tool」的参数 schema。"""
    ticket_type: Literal["refund", "tech", "billing"] = Field(description="工单类型：退款/技术/账单")
    priority: Literal["high", "medium", "low"] = Field(description="优先级：高/中/低")
    summary: str = Field(description="一句话摘要，不超过 30 字")
    account: Optional[str] = Field(description="涉及的账号，未提及则为 None")
    refund_needed: bool = Field(description="是否需要退款")
```

类型注解 + `Field(description)` 共同定义输出的形状与语义，这个类就是"交卷 tool"的参数 schema。

**① 类型注解锁死字段。** `ticket_type: Literal["refund", "tech", "billing"]` 是**枚举约束**——只能从这三个值里选；`refund_needed: bool` 是**类型约束**——必须是布尔；`account: Optional[str]` 是**可空约束**——要么 `None`、要么 `str`。这些"合同条款"是双保险：**吐之前**模型看得到（schema 里写明了允许值，等于把模型往正确方向引导）；**吐之后** Pydantic 还会校验——就算模型还是吐了「未到账」，校验会直接拒绝，而不是让脏数据溜进下游。

**② `Field(description)` 是给模型的"指令"。** 它进 JSON Schema，出现在每个字段的 `description` 里，模型生成参数时就是靠这段话理解"这个字段该填什么"。所以 description 写得敷衍，模型就会乱填或误解字段含义，起反作用。要澄清一句：`Field(description=...)` **不会出现在 `ValidationError` 里**——它只进 JSON Schema（给模型看、给文档看）。比如给 `ticket_type`（description 明明写着"工单类型：退款/技术/账单"）塞一个非法值 `"未到账"`，报错长这样：

```text
1 validation error for TicketExtraction
ticket_type
  Input should be 'refund', 'tech' or 'billing' [type=literal_error, input_value='未到账', input_type=str]
```

错误里只有**字段名、期望枚举、实际值**，一个字都没提 description。想让描述进错误信息，得另写 `error_messages` 或自定义校验器。

**③ 合同 vs 无约束 dict 的本质差别。** 朴素 JSON 拿到的是 `dict`，正确性靠人肉猜（拿到后自己 `if` 检查字段在不在、类型对不对）；Pydantic 类把这份"该长什么样"的预期**写进了 schema**，校验自动发生在"模型吐之后"（`with_structured_output` 内部调 `model_validate`）。一句话：把"靠人肉猜"变成"机器强制执行"。

### §3 把「交卷」伪装成 tool call：with_structured_output

把"输出"伪装成"调用一个交卷工具"——这是 LangChain 的 `with_structured_output` 干的事，demo 里 `structured()` 两行就是核心：

```python
structured_model = model.with_structured_output(TicketExtraction, method="function_calling")
result = structured_model.invoke(TICKET)
```

模型不再"输出文本"，而是"填一张表"：它以为自己要调用一个 `TicketExtraction` 工具，于是把字段值填进工具参数，参数就是结构化输出。填完即交卷，工具**不会被执行**；`json.loads` 解析 + Pydantic 校验这两步由 LangChain 和 Pydantic 自动代劳。

```mermaid
graph LR
    A[定义 Pydantic 类] -->|with_structured_output| B[转成 tool 签名]
    B --> C[模型填参数 = 交卷]
    C --> D[arguments JSON 字符串]
    D -->|json.loads + model_validate| E[TicketExtraction 实例]
```

**① 伪装：Pydantic 类 → 交卷工具签名。** `with_structured_output` 把 `TicketExtraction` 转成一个 tool 的 JSON Schema（函数签名），`method="function_calling"` 告诉它走 `tools` 通道。模型收到这个"工具"后，能"调用"它的唯一方式就是填参数——而这些参数正是我们要的字段。

**② 交卷：结构化数据落在 `arguments`。** 模型返回 `function_call`，字段值塞在 `arguments`（注意是 JSON **字符串**，不是对象）。这两步由 LangChain 和 Pydantic 代劳：`json.loads` 把 `arguments` 解析成 dict → `TicketExtraction.model_validate(dict)` 校验成实例。呼应 §1：你手写会踩的 `json.loads`，现在它们替你干了。demo 里 `result` 直接就是 `TicketExtraction` 实例（`ticket_type='refund'`、`refund_needed=True`）。

**③ method 三种：走哪条底层通道。** `with_structured_output` 只是"门面"，它底层有三种实现，差别在**模型被约束到什么程度**：

- **`function_calling`（最通用）**：把 schema 伪装成一个"交卷工具"，走 `tools` 参数、`tool_choice` 强制模型必须调用它。模型被迫"填表"，参数又被 Pydantic 校验——所有支持 function calling 的模型都吃这套。
- **`json_schema`（OpenAI 专有）**：走 `response_format={"json_schema":...}`，是 OpenAI 的"原生结构化输出"，模型直接被约束到 schema。最省事，但只有 OpenAI（及少数兼容厂商）支持。
- **`json_mode`（约束最弱）**：走 `response_format={"json_object"}`，只告诉模型"输出合法 JSON"，**不告诉它长什么样**。模型可能吐合法 JSON，但字段、类型、枚举全没保证（§6 会细讲这个坑）。

所以 DeepSeek 上只能选 `function_calling`——它不支持 `json_schema`，`json_mode` 又约束不住形状，上次那个 400 就是默认走了 `json_schema` 导致的（细节留给 §6/§7 展开）。


### §4 对照：naive vs structured

同一个抽取任务，naive vs structured 两种做法——差距在代码量、可靠性、可维护性三个维度。

**① 代码量：手动三步 vs 两行。** naive 从"字符串"到"能用的数据"至少要三步：`strip` 掉 markdown 围栏 → `json.loads` 解析 → 逐字段 `if/else` 校验类型和枚举。这三步里最后一步最重——每个字段都要写判断（`ticket_type` 在不在枚举里、`refund_needed` 是不是真 bool、`account` 缺没缺），少写一个就埋一颗雷。structured 用 `with_structured_output` 把三步全折叠成两行：

```python
structured_model = model.with_structured_output(TicketExtraction, method="function_calling")
result = structured_model.invoke(TICKET)
```

**② 可靠性：dict 提心吊胆 vs 实例放心用。** naive 拿到 `dict` 后，下游写 `d["refund_needed"]` 时仍提心吊胆——它可能是 `"true"` 字符串（`if` 判断会误判为真）、可能是 `None`、甚至直接缺字段抛 `KeyError`。structured 拿到 `TicketExtraction` 实例后，`.refund_needed` 是真正 `bool`、`.ticket_type` 是 `Literal` 枚举，都由 Pydantic 校验保证——demo 里 `未到账` 被校正成 `refund` 就是证据。

**③ 可维护性：单一事实源。** naive 的事实源分散在 prompt、校验代码、下游判断**三处**，加一个 `"urgent"` 枚举值得三处同步、漏改即脏数据；structured 的事实源只有 Pydantic 类**一处**，改一处自动传导到 schema、校验、类型。一句话：把"三处同步"变成"一处定义"。

### §5 组装进 LangGraph：结构化结果直接进 state

结构化输出和 LangGraph 是天然搭档——类型化字段直接进 state，下游放心用，条件路由"简洁又精确"。

**① 类型化字段直接进 state。** `ExtractState` 把 `extraction` 声明成 `TicketExtraction` 类型，结构化结果就以**实例**的形式在节点之间流动：

```python
class ExtractState(TypedDict):
    ticket: str
    extraction: TicketExtraction  # 类型化的抽取结果，直接作为 state 字段
    action: str


def extract_node(state: ExtractState) -> dict:
    model = build_model().with_structured_output(TicketExtraction, method="function_calling")
    return {"extraction": model.invoke(state["ticket"])}
```

`extract_node` 返回的 `extraction` 是一个 `TicketExtraction` 实例（不是字符串、不是 dict），下游节点拿到就能直接用 `.refund_needed`、`.ticket_type`，无需再 `json.loads`、无需再校验。诚实边界一句：demo 没加 checkpointer，state 在内存里节点间直传；**加了 checkpointer 时 state 仍会被序列化**，Pydantic 模型要可序列化（引 §7）。

**② 结构化字段 = 控制流信号。** 下游 `route_node` 直接拿 `bool` 字段做路由：

```python
def route_node(state: ExtractState) -> dict:
    """按结构化字段决定下一步（demo：refund_needed → 走退款）。"""
    e = state["extraction"]
    return {"action": "退款流程" if e.refund_needed else "普通处理"}
```

`if e.refund_needed` 是拿一个真正的 `bool` 当路由条件——"简洁又精确"。对比 naive 做法：模型输出一段文本，节点里写 `if "退款" in text`，靠猜关键词，同义词（"退钱"、"refund"）漏判、无关词误判，全是坑。正因为 §4② 里类型被 Pydantic 保证成了真 `bool`，这里才敢放心地拿它做控制流。

**③ 生产化的一个细节（点到为止）。** demo 里 `extract_node` 每次执行都 `build_model()` 现建模型，浪费连接、重复实例化；生产应"建一次、注入复用"（引 §7 展开）。


### §6 几个坑

结构化输出不是万能药，它的坑大多来自"模型仍会犯错"和"不同 method 语义差别"。

**① JSON mode ≠ structured output。** `json_object`（JSON mode）只保证"合法 JSON"，不保证字段、类型、枚举——模型完全可以吐 `{"refund_needed": "maybe"}` 这种语法合法但语义错误的脏数据。OpenAI 文档还提了一个坑：`json_object` 模式下不写指令，模型可能一直吐空白直到撞 token 上限。所以 demo 走 `function_calling`，不用 `json_mode`。

**② 让模型少犯错的常用措施。** 一是 schema 宁简勿繁——字段越多、嵌套越深，模型"填表"难度越大、越容易填错，字段够用即可；二是 `Literal` 枚举要覆盖全——漏了边界值，模型遇到只能瞎猜；三是 `description` 写清楚——写敷衍了，模型会误解字段含义、填错。这三样都是在"给模型完整上下文"，少给犯错机会。

**③ 结构化输出要 `temperature=0`。** 高 `temperature` 会让枚举/类型输出不稳定，同一个输入每次吐的不一样；确定性任务用 `temperature=0`（demo 就这么设的）。

**④ schema 本身占输入 token。** tool 定义（JSON Schema + description）每一轮都进输入，字段越多越费 token。OpenAI best practices 明确建议：tool 定义要精简——10 个 tool + 冗长 description，单请求可多出 1–3K token。

### §7 生产化延伸

demo 直接 `invoke` 一次就完，是为了讲清机制；生产里 function calling 要进 agent 循环、接真实工具、补失败处理。

**① DeepSeek 兼容性（呼应 §3③ 的 400 坑）。** DeepSeek 支持标准 `tools`/function calling，但不支持 OpenAI 的 `json_schema`（`response_format`）。所以 `method` 得显式写 `function_calling`——demo 里那行注释就是这么来的，它适用范围最广、OpenAI 兼容模型基本都支持。这是"多 provider 部署"绕不开的一个真实约束：不是所有兼容接口都实现了 OpenAI 的全部能力。

**② 生产形态：`create_agent` + `@tool`。** 生产里 `@tool` 是**真工具**——会被 agent 调用、真正执行、影响外部环境（查库、发请求）；demo 的"交卷工具"是**假工具**——只接结构化输出，不执行、无副作用。本质差别就一句：工具是否真的执行并影响外部环境。二者各司其职：结构化输出仍靠 `with_structured_output`（或"交卷 tool"），`@tool` 是给 agent 干活的真工具。

**③ 诚实边界（demo 没做、生产要补的）。** demo 直接 `invoke` 一次就完，零重试、零错误处理。生产要补四样：① 进 agent 循环（多轮、有状态）；② 失败重试（校验失败时把错误反馈给模型重试，LangChain 有智能重试机制）；③ 错误处理兜底（模型仍可能吐脏值）；④ token 成本放大（§6④ 的 schema 占 token，在循环里逐轮累积）。

### §8 总结 + 面试题

一句话收尾：让 LLM 稳定输出 JSON 的可靠做法，不是让它"输出 JSON"，而是把"交卷"伪装成一次 tool call——`model.with_structured_output(Schema)` 一行，Pydantic schema 当合同，返回类型化对象，零手工解析、类型有保证。

两道高频面试题：

**「怎么让 LLM 稳定输出 JSON？」** 朴素做法不可靠——模型会包一层 markdown 代码围栏、吐废话、漏字段、类型漂移，`json.loads` 只保证"合法"不保证"正确"（§1）。可靠做法是用 `with_structured_output` 把输出伪装成 tool call，用 Pydantic schema 当合同：字段、类型、枚举都在 schema 里锁死，模型吐之前被引导、吐之后被校验（§2/§3）。

**「`function_calling` 和 `json_mode` 有什么区别？」** 两者都是 `with_structured_output` 的 `method`，底层通道不同：`function_calling` 走 `tools` 参数（把 schema 伪装成交卷工具、`tool_choice` 强制调用），保证**符合 schema**；`json_mode` 走 `response_format={"json_object"}`，只保证**是合法 JSON**，不保证字段/类型/枚举。所以结构化输出选 `function_calling`——也更通用，OpenAI 兼容模型基本都支持（DeepSeek 不支持 json_schema，就选它）。

---

## 参考文献

1. **LangChain 官方文档《Structured output》** — <https://docs.langchain.com/oss/python/langchain/structured-output>
2. **OpenAI 官方文档《Function calling》** — <https://platform.openai.com/docs/guides/function-calling>
3. **OpenAI 官方文档《Structured Outputs》** — <https://platform.openai.com/docs/guides/structured-outputs>
4. **DeepSeek 官方文档《Function Calling》** — <https://api-docs.deepseek.com/guides/function_calling>
5. **Pydantic 官方文档《Models》** — <https://docs.pydantic.dev/latest/concepts/models/>

---

完整代码：[`ticket_extract.py`](https://github.com/wanghesong2019/langgraph-in-action/blob/main/05-function-calling-schema/ticket_extract.py)
