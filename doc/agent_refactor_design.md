# Agent 重构设计文档（API 草案）

## 1. 目标

本次重构采用分层但尽量收敛的设计：

1. `Agent` 是唯一核心业务类，表示一个完整算法流程。
2. `phase` 和 `session` 不再单独设计成类，而是写成 `Agent` 的方法。
3. `turn` 不单独抽象成模块，直接并入模型适配层。
4. 多轮交互循环只保留一套，统一由 `BaseAgent._run_session(...)` 实现。
5. 不再以全局 `AgentState` 作为主输入，session 主要采用 `SessionContext -> SessionResult` 的传递方式。
6. `benchmark` 和 `satnet_agent` 共享同一套主流程，任务差异主要通过 benchmark/task 类型、prompt 和可用 tools 体现。

---

## 2. 推荐目录结构

```text
src/
  agent/
    base_agent.py
    types.py
    message_manager.py
    errors.py

  llm/
    base.py
    qwen.py
    ollama.py
    types.py

  tools/
    base.py
    types.py
    benchmark_tools.py
    satnet_tools.py

  loggingx/
    base_logger.py
    logger.py
    txt_logger.py
    json_logger.py
    types.py

  benchmark/
    prompts.py
    tools.py
    agent.py
    run_benchmark.py

  satnet_agent/
    prompts.py
    tools.py
    agent.py
    run_benchmark.py
```

---

## 3. 设计原则

### 3.1 Agent 是主类

`Agent` 负责：

- 定义整个算法流程
- 组织 phase/session 顺序
- 根据任务类型选择可用 tools
- 传递 session 输入和承接 session 输出
- 决定何时结束

一个 agent 的 `run()` 应该能直接读成算法骨架：

```python
def run(self):
    self.bootstrap_phase_session()
    self.plan_phase_session()
    self.solve_phase_session()
    self.verify_phase_session()
```


并且 `run()` 入口应当直接接收任务类型，例如 `benchmark_type` / `task_type`，这样同一个 agent 主流程就能统一处理 benchmark 和 satnet，只把差异留在 prompt、tool list 和少量业务逻辑里。

---

### 3.2 Phase 是业务实现上的函数

phase 不作为架构层抽象，也不单独设计成类。

在当前设计里，phase 现在更适合看成 `Agent` 里的大 session，例如：

- `bootstrap_phase_session`
- `plan_phase_session`
- `solve_phase_session`
- `verify_phase_session`

它的本质是：

- 组织若干个 session
- 汇总这些 session 的结果
- 决定下一步调用哪个 session 或 phase

也就是说，phase 更像“一个比较大的业务步骤”，不是单独一层框架接口。

从实现角度看：

- phase 可以看成“包含多个 session 的函数”
- 如果需要，它本质上也可以被理解成一个更大的 session
- 因此 phase / session 更接近业务上的层次嵌套，而不是两套完全不同的架构对象

---

### 3.3 Session 也是 Agent 的方法

session 不作为类实现，而是 `Agent` 的普通方法，例如：

- `plan_session`
- `execute_plan_session`
- `verify_session`

这里的 session 不应该设计得太“框架化”。

更贴近实际的理解是：

- session 接收一段 prompt/context
- 这个 context 可以包含前面 session 的结论、工具调用结果、历史消息
- 模型按当前 session 的流程完成一次局部求解
- session 返回本次局部求解的文本结论，以及原始 response/message
- `MessageManager` 再统一记录和管理这些历史输出
- 如果有工具调用，session 仍由 agent 自己执行具体工具

一句话：

- session 是一个“围绕局部目标的求解过程”
- 它的核心产物是“文本结论 + 原始消息记录”

---

### 3.4 通用多轮循环只保留一套

所有 session 的通用执行流程统一放在：

```python
BaseAgent._run_session(...)
```

业务 agent 不重复写多轮循环，只提供 session 的差异化回调。

---

### 3.5 Turn 不单独设计

`turn` 不单独建模块，不设计 `Turn` 类。

一次 request / response 直接由：

```python
LLMClient.generate(request) -> ModelResponse
```

负责。

---

### 3.6 MessageManager 统一管理 message 来源与回溯

`MessageManager` 的核心目的不是单纯“存消息”，而是管理“本次发给模型的 message 是由哪些来源组成的”，并保证这些来源在日志里可追溯。

典型场景：

- 当前要发给模型一个新的 request
- 这个 request 的 message 可能来自：
  - user 初始输入
  - 某个 session 的结论
  - 某个工具调用结果
  - 某个 phase 的阶段输出
- 希望最后在 log 里能明确看到：这一轮 message 是由哪些来源组合出来的

因此 `MessageManager` 统一负责：

- 存储不同来源的 message/output
- 记录每条 message 的来源信息
- 根据 phase / session / user / tool 等来源组装本轮 request messages
- 让 logger 能追溯“本轮 request 用到了哪些历史来源”

一句话：

- `MessageManager` 解决的不是单纯查 message
- 而是“构造 request 时，message 来源必须清楚且可追溯”

---

### 3.7 Logger 采用分发设计

日志系统只保留两类最终输出：

- 一个给人看的 `txt` 全流程日志
- 一个包含完整 request / response / messages 的 `json` 大日志

结构上采用：

- `BaseLogger`
- `Logger`
- `TxtLogger`
- `JsonLogger`

其中：

- `Logger` 是统一入口，只负责分发日志事件
- `TxtLogger` 负责写人类可读的全过程日志
- `JsonLogger` 负责写完整结构化日志

实现上可以直接采用经典的 `handler` 模式：

- `Logger` 持有多个 handler
- 每个 handler 决定如何把同一个事件写到自己的目标格式

---

## 4. 数据结构放置原则

数据结构不单独追求“定义得很全”，而是遵循最简原则：

1. 只有跨模块共享、必须统一的结构，才放到公共位置。
2. 某个 provider、某个 manager、某个 logger 自己内部使用的数据结构，放回对应模块。
3. `Message` 尽量贴近 Qwen 的消息格式，避免再做一层过重抽象。

---

### 4.1 建议保留在 `agent/types.py` 的公共结构

当前版本不再以 `AgentState` 作为主输入对象，而是采用更直观的 `SessionContext -> SessionResult` 传递方式。

建议保留在 `agent/types.py` 的核心公共结构是：

```python
@dataclass
class SessionContext:
    phase_name: str
    session_name: str
    goal: str
    history: list[Message]
    tools: list[BaseTool]
    max_turns: int = 8
    metadata: dict[str, Any] = field(default_factory=dict)
```

```python
@dataclass
class SessionResult:
    session_name: str
    success: bool
    summary: str = ""
    response: ModelResponse | None = None
    messages: list[Message] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
```

其中：

- `SessionContext` 是一次 session 的明确输入
- `SessionResult` 是一次 session 的明确输出

这两个对象分别承担：

- session 需要什么才能运行
- session 运行后产出什么

这样比传一个全局 `state` 更直观。

---

### 4.2 Message 放到 `llm/types.py`

`Message` 尽量和 Qwen 使用的消息格式保持一致，因此不建议把它设计成过重的通用对象。

建议放到 `llm/types.py`，并保持接近 provider message 结构：

```python
@dataclass
class Message:
    role: str
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
```

设计原则：

- `role` 直接兼容 provider 侧的 `system/user/assistant/tool`
- `content` 不强行只限制成 `str`，允许后面兼容多模态或分段 content
- `tool_calls` 保持 provider 风格，不额外包一层通用类
- 不在 `Message` 上堆太多 phase/session/turn 元信息；这些信息更适合放日志和 request context

一句话：`Message` 更像 provider 协议对象，不像全局业务对象。

---

### 4.3 放到 `llm/types.py` 的结构

这些结构主要服务于模型适配层，建议放到 `llm/types.py`：

```python
@dataclass
class ModelRequest:
    agent_name: str
    phase_name: str
    session_name: str
    turn_index: int
    messages: list[Message]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
```

```python
@dataclass
class ModelResponse:
    message: Message | None
    raw_response: Any
    stop_reason: str | None = None
```

说明：

- `tools` 直接传 provider 兼容 schema 即可
- `ModelResponse` 只保留最核心的返回
- 如果某个 provider 需要解析 `tool_calls`，可直接从 `message.tool_calls` 读取

这样可以减少很多重复包装。

---

### 4.4 放到 `tools/types.py` 的结构

工具模块自己的数据结构放回 `tools/types.py`：

```python
@dataclass
class ToolResult:
    success: bool
    output: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

建议优先直接使用标准 schema dict：

```python
tool_schema: dict[str, Any]
```

而不是先定义很多 dataclass 再来回转换。

---

### 4.5 不建议单独抽出来的结构

以下这些结构，当前版本不建议作为公共 dataclass 强行抽象：

- `ToolCall`

原因：

- 它们更偏 provider/工具协议层，不是全局业务层概念
- 抽出来后往往还是要在模型适配层和业务工具层之间来回转换
- 第一版先直接使用 dict 更简单

也就是：

- provider 返回的 tool call，先在 `QwenClient` 内部按 dict 处理
- 工具执行直接按 `tool_name + arguments` 调业务侧实现即可
- 真到后面多个 provider/协议都稳定复用时，再决定要不要上升为公共 dataclass

---

### 4.6 当前推荐的最小公共数据结构集合

当前版本建议只稳定保留这几类：

- `SessionContext` in `agent/types.py`
- `SessionResult` in `agent/types.py`
- `Message` in `llm/types.py`
- `ModelRequest` in `llm/types.py`
- `ModelResponse` in `llm/types.py`
- `ToolResult` in `tools/types.py`

其余结构优先作为对应模块内部细节处理。

## 5. Agent 设计

## 5.1 BaseAgent

`BaseAgent` 是所有业务 agent 的通用基类。

职责：

- 提供总体 `run()` 结构支撑
- 提供通用 session loop
- 负责 phase/session 之间的结果衔接
- 调用 `MessageManager`
- 调用 `QwenAdapter` / model adapter
- 调用 `Logger`

---

### 5.2 BaseAgent 最小公开 API

```python
class BaseAgent:
    def __init__(
        self,
        name: str,
        model_adapter: QwenAdapter,
        message_manager: MessageManager,
        logger: Logger,
        max_turns: int = 8,
    ) -> None: ...

    def run(self, task_type: str, input_data: dict[str, Any]) -> SessionResult | None: ...
    def should_stop(self) -> bool: ...
    def resolve_available_tools(self, task_type: str) -> list[BaseTool]: ...
    def on_agent_start(self) -> None: ...
    def on_agent_end(self) -> None: ...
```

---

### 5.3 BaseAgent 内部通用 API

这些 API 主要供子类 agent 复用。

```python
class BaseAgent:
    def _run_session(
        self,
        context: SessionContext,
        prompt: str,
        source_trace: list[dict[str, Any]] | None = None,
    ) -> SessionResult: ...

    def _call_model(
        self,
        context: SessionContext,
        prompt: str,
    ) -> ModelResponse: ...
```

---

### 5.4 BaseAgent 的实现边界

`BaseAgent` 只提供 session 这一层真正需要复用的通用能力。

phase 不需要单独提供通用接口，因为 phase 本来就是业务上的大 session：

- 业务 agent 直接写 `plan_phase_session()` / `solve_phase_session()` / `verify_phase_session()`
- phase_session 内部自己决定要调几个普通 session
- phase_session 内部自己决定怎么组织嵌套

也就是说：

- 通用能力收敛在 `_run_session(...)`
- phase_session 直接保留为业务实现

## 6. 业务 Agent API

业务层保留为一个主类。

---

### 6.1 BenchmarkAgent

```python
class BenchmarkAgent(BaseAgent):
    def run(self, task_type: str, input_data: dict[str, Any]) -> SessionResult | None: ...

    def bootstrap_phase_session(self) -> SessionResult: ...
    def plan_phase_session(self) -> SessionResult: ...
    def solve_phase_session(self) -> SessionResult: ...
    def verify_phase_session(self) -> SessionResult: ...

    def plan_session(self, context: SessionContext) -> SessionResult: ...
    def execute_plan_session(self, context: SessionContext, plan: dict[str, Any]) -> SessionResult: ...
    def verify_session(self, context: SessionContext, target: dict[str, Any]) -> SessionResult: ...
```

---

### 6.2 SatNetAgent

```python
class SatNetAgent(BaseAgent):
    def run(self, task_type: str, input_data: dict[str, Any]) -> SessionResult | None: ...

    def bootstrap_phase_session(self) -> SessionResult: ...
    def plan_phase_session(self) -> SessionResult: ...
    def solve_phase_session(self) -> SessionResult: ...
    def verify_phase_session(self) -> SessionResult: ...

    def plan_session(self, context: SessionContext) -> SessionResult: ...
    def execute_plan_session(self, context: SessionContext, plan: dict[str, Any]) -> SessionResult: ...
    def verify_session(self, context: SessionContext, target: dict[str, Any]) -> SessionResult: ...
```

## 7. 业务 Agent 内部 helper 设计

这里也建议收敛，不要为了 session 强行拆太多 callback。

每个 session 最多只需要少量 helper：

### 7.1 推荐的最小 helper

```python
def _build_plan_prompt(self, goal: str, history: list[Message]) -> str: ...
def _build_execute_prompt(self, goal: str, plan: dict[str, Any], history: list[Message]) -> str: ...
def _build_verify_prompt(self, goal: str, target: dict[str, Any], history: list[Message]) -> str: ...
```

```python
def _make_context(
    self,
    phase_name: str,
    session_name: str,
    goal: str,
    history: list[Message],
    tools: list[BaseTool],
    max_turns: int = 8,
    **metadata: Any,
) -> SessionContext: ...

```python
def resolve_available_tools(self, task_type: str) -> list[BaseTool]: ...
def get_session_tools(self, task_type: str, session_name: str) -> list[BaseTool]: ...
```
```

```python
def _extract_session_summary(self, response: ModelResponse) -> str: ...
def _extract_tool_results(self, response: ModelResponse) -> list[dict[str, Any]]: ...
```

如果没有必要，不要继续往下拆更多 hook。

## 8. phase 建议写法

phase 现在直接看成更大的 session，因此建议统一返回 `SessionResult`。

---

### 8.1 示例：plan_phase_session

```python
def plan_phase_session(self) -> SessionResult:
    bundle = self.message_manager.collect(
        sources=[
            ("system", "default_system"),
            ("user", "initial_input"),
        ]
    )
    context = self._make_context(
        phase_name="plan",
        session_name="plan_session",
        goal="Generate candidate plans for this task",
        history=bundle.messages,
        tools=self.get_session_tools(self.task_type, "plan_session"),
        source_trace=bundle.source_trace,
    )
    return self.plan_session(context)
```

---

### 8.2 示例：solve_phase_session

```python
def solve_phase_session(self) -> SessionResult:
    results = []

    for plan in self.candidate_plans:
        bundle = self.message_manager.collect(
            sources=[
                ("system", "default_system"),
                ("user", "initial_input"),
                ("session", "plan_session"),
            ]
        )
        context = self._make_context(
            phase_name="solve",
            session_name="execute_plan_session",
            goal="Execute one candidate plan and return the result",
            history=bundle.messages,
            tools=self.get_session_tools(self.task_type, "execute_plan_session"),
            source_trace=bundle.source_trace,
        )
        result = self.execute_plan_session(context, plan)
        if result.success:
            results.append(result.data)

    if not results:
        return SessionResult(
            session_name="solve_phase_session",
            success=False,
            error="no_plan_executed_successfully",
        )

    self.best_result = self.select_best_result(results)
    return SessionResult(
        session_name="solve_phase_session",
        success=True,
        summary="selected best execution result",
        data={"results": results, "best_result": self.best_result},
    )
```

---

### 8.3 示例：verify_phase_session

```python
def verify_phase_session(self) -> SessionResult:
    bundle = self.message_manager.collect(
        sources=[
            ("system", "default_system"),
            ("user", "initial_input"),
            ("session", "execute_plan_session"),
        ]
    )
    context = self._make_context(
        phase_name="verify",
        session_name="verify_session",
        goal="Verify the best execution result",
        history=bundle.messages,
        tools=self.get_session_tools(self.task_type, "verify_session"),
        source_trace=bundle.source_trace,
    )
    result = self.verify_session(context, self.best_result)
    if result.success:
        self.finished = True
        self.success = True
    return result
```

---

### 8.4 phase 和 session 的关系

当前推荐理解为：

- session 是局部求解单元
- phase 是更大的嵌套 session
- phase 内部可以调用多个 session
- 如果需要，phase 也可以继续嵌套 phase/session

所以 phase 不是独立架构层，而是业务上人为划分出的较大 session。

## 9. session 建议写法

每个 session 方法应该尽量像“一个明确输入、明确输出”的局部流程。

推荐写法是：

- phase_session 先通过 `MessageManager` 收集历史消息
- phase_session 组装出 `SessionContext`
- session 接收 `SessionContext`
- session 根据 `context.goal / context.history / context.tools` 构造 prompt
- 最后调用统一的 `_run_session(...)`

---

### 9.1 示例：plan_session

```python
def plan_session(self, context: SessionContext) -> SessionResult:
    prompt = self._build_plan_prompt(context.goal, context.history)
    return self._run_session(context, prompt)
```

---

### 9.2 示例：execute_plan_session

```python
def execute_plan_session(self, context: SessionContext, plan: dict[str, Any]) -> SessionResult:
    prompt = self._build_execute_prompt(context.goal, plan, context.history)
    return self._run_session(context, prompt)
```

---

### 9.3 示例：verify_session

```python
def verify_session(self, context: SessionContext, target: dict[str, Any]) -> SessionResult:
    prompt = self._build_verify_prompt(context.goal, target, context.history)
    return self._run_session(context, prompt)
```

## 10. MessageManager 设计

`MessageManager` 统一管理历史输出，并负责给 phase/session 提供可追溯的历史消息。

它的核心目标是：

- 当前要发给模型的每条 message，都知道它来自哪里
- request 里的 message 组合关系可追溯
- logger 能记录“这次 request 用了哪些来源”
- agent 不需要手动维护复杂的 message 拼接逻辑

也就是说，`MessageManager` 的重点不是提供很多零碎查询函数，而是提供两类核心能力：

1. 记录一条输出属于哪个来源
2. 按来源一次性产出本轮 request bundle

---

### 10.1 推荐的来源类型

建议统一用一个很简单的来源概念：

- `user`
- `phase`
- `session`
- `tool`
- `system`

每条被记录的 message/output，都带一个来源标识。

例如：

- `user:initial_input`
- `session:plan_session`
- `session:verify_session`
- `phase:plan_phase`
- `tool:run_simulation`

这样 logger 在落 request 的时候，就能直接把来源一起写进去。

---

### 10.2 RequestBundle

推荐让 `MessageManager` 直接产出一个 request bundle，而不是拆成多个零散函数。

```python
@dataclass
class RequestBundle:
    messages: list[Message]
    source_trace: list[dict[str, Any]]
```

其中：

- `messages` 是本轮真正发给模型的消息列表
- `source_trace` 是这些消息的来源说明，直接给 logger 使用

---

### 10.3 MessageManager 最小 API

```python
class MessageManager:
    def __init__(self) -> None: ...

    def record(
        self,
        source_type: str,
        source_name: str,
        message: Message,
    ) -> None: ...

    def record_many(
        self,
        source_type: str,
        source_name: str,
        messages: list[Message],
    ) -> None: ...

    def get_by_source(
        self,
        source_type: str,
        source_name: str,
    ) -> list[Message]: ...

    def collect(
        self,
        *,
        sources: list[tuple[str, str]],
    ) -> RequestBundle: ...

    def clear_source(
        self,
        source_type: str,
        source_name: str,
    ) -> None: ...
```

---

### 10.4 API 含义

#### `record(...)`
记录一条 message 属于哪个来源。

例如：

- 把 user 输入记为 `("user", "initial_input")`
- 把 `plan_session` 的输出记为 `("session", "plan_session")`
- 把 `plan_phase` 的阶段结论记为 `("phase", "plan_phase")`

#### `get_by_source(...)`
按来源取 message。

这满足“我想直接找到某个 phase/session 的输出”。

#### `collect(...)`
按来源一次性产出本轮 request 所需内容。

例如：

```python
bundle = message_manager.collect(
    sources=[
        ("user", "initial_input"),
        ("session", "plan_session"),
        ("phase", "plan_phase"),
    ]
)
```

得到：

- `bundle.messages`
- `bundle.source_trace`

这样既能发给模型，也能直接记录来源。

---

### 10.5 MessageManager 职责边界

负责：

- 记录 message/output 属于哪个来源
- 按来源取历史输出
- 按来源组合下一轮 request bundle
- 为 logger 提供 request 来源说明

不负责：

- prompt 文案生成
- tool 调用
- 模型请求发送
- 日志格式本身

一句话：

- `MessageManager` 负责“这轮 request 的 message 从哪里来”
- `Logger` 负责“把这些来源记下来”
- `Agent` / session 负责“决定本轮到底要用哪些来源”

## 11. LLM 设计

这一层建议继续收敛，不再把 LLM 拆成很多子模块。

最终保留两层就够了：

- `QwenAdapter`：负责所有和模型协议相关的事情
- `OllamaChatClient`：负责和 Ollama 通信

也就是说：

- message 怎么组织
- tools schema 怎么暴露
- tool calls 怎么解析
- tool result message 怎么回填

这些都放进 `QwenAdapter`。

而 `OllamaChatClient` 只负责把请求发出去，把原始 response 收回来。

---

### 11.1 OllamaChatClient

建议直接参考当前仓库已有 API。

```python
class OllamaChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 300,
        temperature: float = 0.0,
    ) -> None: ...

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]: ...
```

职责：

- 组装 Ollama 请求 payload
- 调用 `/chat/completions`
- 返回原始 json response

不负责：

- session 流程
- message 来源管理
- 工具执行
- 日志分发

---

### 11.2 QwenAdapter

`QwenAdapter` 负责所有和 Qwen/Ollama 协议相关的事情。

```python
class QwenAdapter:
    def __init__(self, ollama_client: OllamaChatClient) -> None: ...

    def generate(
        self,
        messages: list[Message],
        tools: list[BaseTool] | None = None,
    ) -> ModelResponse: ...

    def build_tool_schemas(
        self,
        tools: list[BaseTool],
    ) -> list[dict[str, Any]]: ...

    def parse_tool_calls(
        self,
        response: dict[str, Any],
    ) -> list[dict[str, Any]]: ...

    def build_tool_result_message(
        self,
        tool_name: str,
        result: ToolResult,
    ) -> Message: ...
```

职责：

- 组织 provider 兼容的 messages
- 把 `BaseTool` 转成 Qwen/Ollama 兼容 schema
- 调用 `OllamaChatClient`
- 解析 response
- 解析 tool calls
- 构造 tool result message

也就是说，模型特化的 message/tool 逻辑统一收进 `QwenAdapter`，不要再拆出去。

---

### 11.3 和 Agent 的职责边界

当前推荐分工是：

- `Agent` 决定当前 session 暴露哪些 tools
- `Agent` 负责执行工具本体
- `QwenAdapter` 负责把这些 tools 暴露给模型，并解析模型返回的 tool calls
- `OllamaChatClient` 负责通信

这样职责最清楚：

- 工具属于 agent 的业务能力
- 工具协议属于 Qwen 的模型适配能力

## 12. Tool 设计

当前版本希望统一 benchmark 和 satnet 两类任务，因此 tool 设计也应统一到同一个入口上。

推荐做法是：

- `run(...)` 直接接收 `task_type` / `benchmark_type`
- `Agent` 内部通过一个必须实现的函数，解析该任务可用的工具列表
- 后续每个 session 再从这批可用工具里选择子集

也就是说：

- task 级：当前任务总共有哪批 tools
- session 级：当前 session 暴露哪一部分 tools

这两个决策都放在 `Agent` 里，不再单独抽一个工具管理器。

---

### 12.1 BaseTool

```python
class BaseTool(ABC):
    name: str
    description: str
    parameters_schema: dict[str, Any]

    @abstractmethod
    def invoke(self, arguments: dict[str, Any]) -> ToolResult: ...
```

---

### 12.2 Agent 必须实现的 tool 函数

```python
def resolve_available_tools(self, task_type: str) -> list[BaseTool]: ...
def get_session_tools(self, task_type: str, session_name: str) -> list[BaseTool]: ...
def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult: ...
```

职责：

- `resolve_available_tools(...)`
  根据当前 benchmark/task 类型，返回这类任务总共可用的 tools

- `get_session_tools(...)`
  从当前任务可用 tools 中，选出某个 session 要暴露的子集

- `invoke_tool(...)`
  按名字执行工具

---

### 12.3 推荐写法

例如：

```python
def resolve_available_tools(self, task_type: str) -> list[BaseTool]:
    if task_type == "benchmark":
        return [
            self.read_case_tool,
            self.query_constraint_tool,
            self.run_simulation_tool,
            self.commit_plan_tool,
        ]
    if task_type == "satnet":
        return [
            self.read_week_tool,
            self.query_visibility_tool,
            self.build_schedule_tool,
            self.verify_schedule_tool,
        ]
    return []
```

```python
def get_session_tools(self, task_type: str, session_name: str) -> list[BaseTool]:
    available = {tool.name: tool for tool in self.resolve_available_tools(task_type)}

    if session_name == "plan_session":
        names = ["read_case", "query_constraint"] if task_type == "benchmark" else ["read_week", "query_visibility"]
    elif session_name == "execute_plan_session":
        names = ["read_case", "run_simulation", "commit_plan"] if task_type == "benchmark" else ["build_schedule_tool"]
    elif session_name == "verify_session":
        names = ["commit_plan"] if task_type == "benchmark" else ["verify_schedule_tool"]
    else:
        names = []

    return [available[name] for name in names if name in available]
```

```python
def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
    for tool in self.current_available_tools:
        if tool.name == tool_name:
            return tool.invoke(arguments)
    raise ValueError(f"Unknown tool: {tool_name}")
```

这个设计的关键点是：

- benchmark 和 satnet 共用同一个 `run(...)`
- 任务差异通过 `task_type` 解析可用工具集合
- session 只负责从可用集合里选子集

## 13. Logger 设计

logger 部分只保留两类最终输出：

- `txt`：给人看的全流程日志
- `json`：包含完整 request / response / messages 的大日志文件

这里不追求拆很多细粒度 log 函数，而是追求：

- 对上层调用足够简单
- 对下层输出足够完整
- `Logger` 只做事件分发
- 具体格式由 `TxtLogger` / `JsonLogger` 各自实现

实现上推荐直接使用经典的 `handler` 模式。

---

### 13.1 BaseLogger

`BaseLogger` 代表一个日志 handler 的最小接口。

```python
class BaseLogger(ABC):
    def log_event(self, event_type: str, payload: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

说明：

- `event_type` 表示事件类型，例如 `session_start` / `model_request` / `tool_result`
- `payload` 是该事件的统一数据
- `BaseLogger` 不规定具体输出格式，只规定 handler 的最小行为

---

### 13.2 Logger

`Logger` 是统一入口，负责把日志事件分发给多个 handler。

```python
class Logger:
    def __init__(self, handlers: list[BaseLogger]) -> None: ...

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

推荐内部实现方式：

- `Logger` 持有多个 handler
- 调用 `log_event(...)` 时，遍历分发给每个 handler
- `flush()` / `close()` 也统一广播

---

### 13.3 TxtLogger

```python
class TxtLogger(BaseLogger):
    def __init__(self, log_path: str | Path) -> None: ...
```

职责：

- 输出给人看的全过程日志
- 重点保留对话内容、thinking、tool 调用、tool 返回、阶段切换
- 让人能够顺着文本直接看完整运行过程

建议内容：

- phase / session 起止
- 每轮 prompt / assistant 输出
- thinking 内容
- tool 调用参数
- tool 返回结果
- 最终结果摘要

---

### 13.4 JsonLogger

```python
class JsonLogger(BaseLogger):
    def __init__(self, log_path: str | Path) -> None: ...
```

职责：

- 输出一个完整的大 json 日志文件
- 保存所有 request / response / message / tool result
- 主要服务于调试、复盘、离线分析

建议内容：

- 所有 model request
- 所有 model response
- 所有 messages
- 所有 tool 调用和结果
- 最终 summary

---

### 13.5 推荐事件类型

虽然不需要拆很多单独的 log 函数，但建议统一约定少量事件类型，例如：

- `agent_start`
- `agent_end`
- `phase_start`
- `phase_end`
- `session_start`
- `session_end`
- `model_request`
- `model_response`
- `tool_call`
- `tool_result`
- `summary`

上层统一调用：

```python
logger.log_event(event_type, payload)
```

具体是写成 txt 还是 json，由对应 logger handler 决定。

---

## 14. prompts 设计

`prompts.py` 只保留 prompt 构造函数，不做复杂类设计。

例如：

```python
def build_plan_prompt(goal: str, history: list[Message]) -> str: ...
def build_execute_prompt(goal: str, plan: dict[str, Any], history: list[Message]) -> str: ...
def build_verify_prompt(goal: str, target: dict[str, Any], history: list[Message]) -> str: ...
```

---

## 15. 一条完整执行链路

以 `plan -> solve -> verify` 为例。

---

### 15.1 Agent 层

```python
def run(self):
    self.on_agent_start()

    bootstrap_result = self.bootstrap_phase_session()
    if not bootstrap_result.success or self.should_stop():
        return bootstrap_result

    plan_result = self.plan_phase_session()
    if not plan_result.success or self.should_stop():
        return plan_result

    solve_result = self.solve_phase_session()
    if not solve_result.success or self.should_stop():
        return solve_result

    verify_result = self.verify_phase_session()

    self.on_agent_end()
    return verify_result
```

---

### 15.2 plan_phase_session

`plan_phase_session()` 启动一个 `plan_session()`：

- 生成候选 plans
- 写入 `state.shared["candidate_plans"]`

---

### 15.3 solve_phase_session

`solve_phase_session()`：

- 遍历 `candidate_plans`
- 对每个 plan 调一次 `execute_plan_session()`
- 收集结果
- 选择最优结果
- 写入 `state.shared["best_plan_result"]`

---

### 15.4 verify_phase_session

`verify_phase_session()`：

- 启动一个 `verify_session()`
- 对最优结果做验证
- 成功则标记：

```python
state.finished = True
state.success = True
```

---

### 15.5 单个 Session 内部流程

统一由 `_run_session(...)` 实现：

1. 决定本轮 request 需要引用哪些来源
2. 让 `MessageManager.collect(...)` 一次产出 `messages + source_trace`
3. 把 `source_trace` 交给 logger 记录来源
4. 调用 `QwenAdapter.generate(...)`
5. 如果模型发起 tool 调用，就由 agent 执行并记录结果
6. 把本轮 response/message 记回对应的 `session` 来源
7. 输出 `SessionResult`

---

## 16. 最小接口清单

下面这部分可作为最终 API 总表。

---

### 16.1 Agent

```python
run(task_type, input_data) -> SessionResult | None
should_stop() -> bool
on_agent_start() -> None
on_agent_end() -> None

_run_session(context, prompt, source_trace=None) -> SessionResult
_call_model(context, prompt) -> ModelResponse
```

### 16.2 业务 Agent 的 phase_session 方法

```python
bootstrap_phase_session() -> SessionResult
plan_phase_session() -> SessionResult
solve_phase_session() -> SessionResult
verify_phase_session() -> SessionResult
```

---

### 16.3 业务 Agent 的 session + tool 方法

```python
plan_session(context) -> SessionResult
execute_plan_session(context, plan) -> SessionResult
verify_session(context, target) -> SessionResult

resolve_available_tools(task_type) -> list[BaseTool]
get_session_tools(task_type, session_name) -> list[BaseTool]
invoke_tool(tool_name, arguments) -> ToolResult
```

---

### 16.4 MessageManager

```python
record(source_type, source_name, message) -> None
record_many(source_type, source_name, messages) -> None
get_by_source(source_type, source_name) -> list[Message]
collect(sources) -> RequestBundle
clear_source(source_type, source_name) -> None
```

### 16.5 LLM / Qwen / Ollama

```python
class OllamaChatClient:
    __init__(base_url, model, timeout=300, temperature=0.0) -> None
    create_chat_completion(messages, tools) -> dict[str, Any]
```

```python
class QwenAdapter:
    __init__(ollama_client) -> None
    generate(messages, tools=None) -> ModelResponse
    build_tool_schemas(tools) -> list[dict[str, Any]]
    parse_tool_calls(response) -> list[dict[str, Any]]
    build_tool_result_message(tool_name, result) -> Message
```

---

### 16.6 SessionContext / SessionResult

```python
class SessionContext:
    phase_name: str
    session_name: str
    goal: str
    history: list[Message]
    tools: list[BaseTool]
    max_turns: int
    metadata: dict[str, Any]
```

```python
class SessionResult:
    session_name: str
    success: bool
    summary: str
    response: ModelResponse | None
    messages: list[Message]
    tool_results: list[dict[str, Any]]
    data: dict[str, Any]
    error: str | None
```

---

### 16.7 BaseTool

```python
name: str
description: str
parameters_schema: dict[str, Any]
invoke(arguments) -> ToolResult
```

---

### 16.8 BaseLogger

```python
log_event(event_type, payload) -> None
flush() -> None
close() -> None
```

---

### 16.9 Logger

```python
log_event(event_type, payload) -> None
flush() -> None
close() -> None
```

## 17. 当前版本的收敛结论

当前版本最终收敛为：

1. `Agent` 是业务主类。
2. `phase` / `session` 都是 `Agent` 的方法。
3. `phase` 更适合看成更大的嵌套 session，因此统一返回 `SessionResult`。
4. 不再以全局 `AgentState` 作为主输入，而是采用 `SessionContext -> SessionResult` 的传递方式。
5. `MessageManager` 负责管理历史消息来源，并给 phase/session 提供历史 messages。
6. `QwenAdapter` 统一负责 message + tools 的模型协议适配。
7. `OllamaChatClient` 只负责通信。
8. `run(...)` 直接接收 `task_type`，并由 `Agent.resolve_available_tools(...)` 统一解析当前任务可用工具。
9. tools 由 `Agent` 提供和执行。
10. `benchmark` / `satnet_agent` 只保留 prompt、tools、算法流程差异。
