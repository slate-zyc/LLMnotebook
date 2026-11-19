---
tags: [应用, Agent, 框架, LangGraph]
created: 2026-07-29
updated: 2026-07-29
related: [Agent与工具调用.md, Agentic RAG.md, RAG.md]
---

# LangGraph：LLM 应用的状态图编排框架

> 传统的 LLM 调用是"一问一答"的线性模式，但真正的 Agent 应用需要**循环、分支、状态管理、人机协作**。LangChain 的 Chain 抽象（SequentialChain、RouterChain 等）在处理复杂 Agent 逻辑时力不从心。LangGraph 将 Agent 建模为**有向图**，通过节点、边和状态实现了灵活的控制流，是目前构建复杂 LLM Agent 最主流的框架之一。
>
> LangGraph 的核心理念：**Agent = 图（Graph），工作流 = 状态机（State Machine）**。

---

## 一、为什么需要 LangGraph？

### 1.1 传统 Chain 的局限

```
LangChain Chain 模式：
  输入 → [步骤1] → [步骤2] → [步骤3] → 输出
  （线性、固定流程，难以处理分支和循环）

真实 Agent 需求：
  用户输入 → 思考 → 调用工具 → 检查结果
                  ↑                     ↓
                  └─── 不够好，再来一次 ←──┘
  （需要循环、条件判断、状态记忆）
```

### 1.2 LangGraph vs Chain vs 其他框架

| 特性 | LangChain Chain | AutoGPT 风格 | LangGraph |
|------|----------------|-------------|-----------|
| **控制流** | 预定义线性流程 | LLM 自主决定 | 显式图定义 |
| **可预测性** | 高（但僵硬） | 低（可能跑偏） | **高（可控+灵活）** |
| **循环支持** | ❌ | ✅ | ✅ |
| **条件分支** | 有限 | ✅ | ✅ 精细控制 |
| **状态管理** | 隐式 | 隐式 | **显式、可持久化** |
| **人机协作** | ❌ | ❌ | ✅ 内置支持 |
| **流式输出** | 部分支持 | 有限 | ✅ 原生流式 |
| **适用场景** | 简单管道 | 完全自主 | **复杂可控 Agent** |

### 1.3 LangGraph 的一句话定义

> LangGraph = **StateGraph（状态图）**：用图的节点（Node）表示计算步骤，边（Edge）表示流程控制，状态（State）在节点间流转和更新。

---

## 二、核心概念

### 2.1 四大核心组件

```
┌─────────────────────────────────────────────────────┐
│                    LangGraph 核心                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  State（状态）                                       │
│  ├── 图中流转的数据结构                                │
│  ├── 每个节点读写 State                               │
│  └── 可持久化（Checkpointing）                        │
│                                                     │
│  Node（节点）                                        │
│  ├── 计算单元（Python 函数 / Runnable）                │
│  ├── 接收 State，返回 State 更新                      │
│  └── 可以是 LLM 调用、工具执行、任意代码               │
│                                                     │
│  Edge（边）                                         │
│  ├── 普通边（Normal Edge）：直接连接两个节点            │
│  ├── 条件边（Conditional Edge）：根据结果路由          │
│  └── 入口/出口边：定义图的起止                         │
│                                                     │
│  Graph（图）                                        │
│  ├── 编译后成为可执行的 Runnable                       │
│  └── 支持 invoke / stream / astream                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 2.2 State（状态）详解

State 是 LangGraph 最核心的概念——它是**图中所有节点共享的数据结构**：

```python
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

# 方式一：TypedDict 定义
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]  # 消息自动追加
    context: str                              # 检索到的上下文
    next_step: str                            # 下一步动作

# 方式二：使用 MessageGraph（简化版，State 只含 messages）
from langgraph.graph import MessageGraph
graph = MessageGraph()
```

**State 更新的关键规则**：
- 普通字段：新值**覆盖**旧值
- `Annotated` 字段：通过指定的 reducer 函数合并（如 `add_messages` 用于追加消息）
- 每个节点返回的字典会**部分更新** State（不是完全替换）

### 2.3 节点（Node）详解

```python
# 节点可以是任何函数，接收 State，返回 State 的部分更新
def call_model(state: AgentState) -> dict:
    """调用 LLM 节点"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # 只返回要更新的字段

def search_tool(state: AgentState) -> dict:
    """搜索工具节点"""
    query = state["messages"][-1].content
    results = search(query)
    return {"context": results}

def should_continue(state: AgentState) -> str:
    """条件判断节点——返回下一个节点名称"""
    last_message = state["messages"][-1]
    if "FINAL ANSWER" in last_message.content:
        return "end"
    return "continue"
```

`★ Insight ─────────────────────────────────────`
节点的返回值只需要包含**你要更新的字段**，不必返回完整 State。LangGraph 内部会将返回值与当前 State 合并。这是 Redux reducer 模式的应用——每次只描述"改变了什么"，而不是"整个新的状态是什么"。
`─────────────────────────────────────────────────`

### 2.4 边（Edge）详解

```python
from langgraph.graph import StateGraph, END

# 普通边：A 完成后必定到 B
graph.add_edge("node_a", "node_b")

# 条件边：根据函数返回值选择下一个节点
graph.add_conditional_edges(
    "node_name",           # 源节点
    should_continue,       # 路由函数，返回字符串
    {
        "continue": "action_node",   # 返回 "continue" → 去 action_node
        "end": END                   # 返回 "end" → 结束
    }
)

# END 是特殊标记，表示图执行终止
```

---

## 三、核心工作流模式

### 3.1 模式 1：Agent 执行循环（最核心）

这是 LangGraph 最重要、最常用的模式——**带工具的 Agent 循环**：

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

# 1. 定义 State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. 定义工具
tools = [search_tool, calculator, weather_api]

# 3. 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", call_model)   # agent 节点（调用 LLM）
workflow.add_node("tools", ToolNode(tools))  # 工具执行节点（LangGraph 内置！）

# 设置入口
workflow.set_entry_point("agent")

# 添加条件边：agent 之后判断是否需要调用工具
def should_call_tools(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("agent", should_call_tools, {
    "tools": "tools",
    END: END
})

# 工具执行后回到 agent
workflow.add_edge("tools", "agent")

# 4. 编译
app = workflow.compile()
```

**执行流程图**：

```
    ┌─────────┐
    │  START  │
    └────┬────┘
         ↓
    ┌─────────┐     有工具调用     ┌─────────┐
    │  agent  │ ─────────────────→ │  tools  │
    └─────────┘                    └─────────┘
         │                              │
         │ 无工具调用                     │
         ↓                              ↓
    ┌─────────┐                   回到 agent
    │   END   │
    └─────────┘
```

### 3.2 模式 2：LangGraph 预构建 Agent（一行创建）

LangGraph 提供 `create_react_agent`，一行代码创建标准 Agent：

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=[search_tool, calculator],
    state_modifier="你是一个有帮助的助手，优先使用工具获取实时信息。"
)

# 运行
result = agent.invoke({
    "messages": [("user", "DeepSeek-V3 和 GPT-4 有什么区别？")]
})
```

**预构建 Agent 做了什么**：
它内部就是 3.1 中手动构建的图：`agent 节点 → 条件判断 → tools 节点 → 循环回到 agent`。

### 3.3 模式 3：含人机协作（Human-in-the-Loop）

```python
from langgraph.checkpoint import MemorySaver
from langgraph.types import interrupt

# 带中断的节点——敏感操作需要人工确认
def sensitive_action(state: AgentState) -> dict:
    """执行敏感操作前中断，等待人工审批"""
    approval = interrupt("是否确认执行此操作？")
    if approval == "yes":
        result = execute_action()
        return {"result": result}
    else:
        return {"result": "操作已取消"}

# 编译时启用 checkpointing
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 运行时使用 thread_id 追踪会话
config = {"configurable": {"thread_id": "user-123"}}

# 第一次运行——会触发中断
for event in app.stream(input_message, config):
    print(event)

# 人工审批后恢复执行
app.invoke(Command(resume="yes"), config)
```

### 3.4 模式 4：并行分支

```python
from langgraph.graph import StateGraph

# 并行：agent 分发任务给多个子 agent，汇总结果
workflow = StateGraph(State)

workflow.add_node("router", router_node)     # 分发
workflow.add_node("search_agent", search_agent)
workflow.add_node("code_agent", code_agent)
workflow.add_node("merge", merge_results)    # 汇总

workflow.set_entry_point("router")

# 从 router 并行发送到多个 agent
workflow.add_edge("router", "search_agent")
workflow.add_edge("router", "code_agent")

# 所有并行节点完成后，汇聚到 merge
workflow.add_edge("search_agent", "merge")
workflow.add_edge("code_agent", "merge")

workflow.add_edge("merge", END)
```

**并行执行图**：

```
         ┌────────┐
         │ router │
         └───┬────┘
        ┌────┴────┐
        ↓         ↓
  ┌──────────┐ ┌──────────┐
  │  search  │ │   code   │
  │  agent   │ │  agent   │
  └────┬─────┘ └────┬─────┘
       └──────┬─────┘
              ↓
         ┌────────┐
         │ merge  │
         └────────┘
```

### 3.5 模式 5：多步 RAG（Agentic RAG）

```python
class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    answer: str

workflow = StateGraph(RAGState)

def retrieve(state: RAGState) -> dict:
    """检索文档"""
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    return {"context": "\n".join([d.page_content for d in docs])}

def generate(state: RAGState) -> dict:
    """基于上下文生成"""
    prompt = f"基于以下文档回答问题：\n{state['context']}\n\n问题：{state['messages'][-1].content}"
    response = llm.invoke(prompt)
    return {"messages": [response], "answer": response.content}

def grade(state: RAGState) -> str:
    """评估答案质量，决定是否重新检索"""
    if "不相关" in state["context"] or len(state["context"]) < 100:
        return "rewrite"  # 检索结果不好，改写查询重试
    return "end"

def rewrite_query(state: RAGState) -> dict:
    """改写查询以增强检索"""
    better_query = llm.invoke(f"优化以下查询以改善搜索效果：{state['messages'][-1].content}")
    return {"messages": [HumanMessage(content=better_query.content)]}

# 构建图
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("rewrite", rewrite_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges("generate", grade, {
    "rewrite": "rewrite",
    "end": END
})
workflow.add_edge("rewrite", "retrieve")  # 回到检索，形成循环
```

### 3.6 模式 6：Supervisor 模式（Multi-Agent 调度）

Supervisor（主管）模式是 LangGraph 中最经典的**多 Agent 协作架构**。它模仿团队管理结构：一个 Supervisor Agent 作为"老板"接收任务、分发给专业子 Agent、根据结果决定下一步，最后汇总输出。

```
用户请求 → Supervisor 分析 → 路由到对应 Worker
                                  ↓
    ┌─────────────────────────────┴─────────────────────────┐
    ↓                    ↓                    ↓              ↓
┌─────────┐      ┌─────────────┐      ┌──────────┐    ┌──────────┐
│ 研究员  │      │  数学专家   │       │ 代码专家  │    │ 普通助手  │
│(搜索+阅读)│     │(计算+验证)   │       │(写代码+运行)│   │(闲聊回答) │
└─────────┘      └─────────────┘      └──────────┘    └──────────┘
    ↓                    ↓                    ↓              ↓
    └─────────────────────┴────────────────────┴──────────────┘
                                  ↓
                    Supervisor 收到结果 → 满意？→ 汇总输出
                                         → 不满意？→ 重新分配
```

**核心实现**：

```python
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Literal

# 定义 Supervisor 的 State
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str  # Supervisor 决定下一步去哪个 Worker

# 定义 Worker 节点（每个 Worker 是独立的 LLM Agent）
def research_agent(state: SupervisorState) -> dict:
    """研究员：负责搜索和阅读"""
    response = llm.invoke(
        [SystemMessage(content="你是研究员，擅长搜索和分析信息。"),
         state["messages"][-1]]
    )
    return {"messages": [response]}

def math_agent(state: SupervisorState) -> dict:
    """数学专家：负责计算和数学推理"""
    response = llm.invoke(
        [SystemMessage(content="你是数学专家，擅长计算和数学推理。"),
         state["messages"][-1]]
    )
    return {"messages": [response]}

def coder_agent(state: SupervisorState) -> dict:
    """代码专家：负责编写和执行代码"""
    response = llm.invoke(
        [SystemMessage(content="你是代码专家，擅长编写和执行代码。"),
         state["messages"][-1]]
    )
    return {"messages": [response]}

# 定义 Supervisor 节点——核心决策者
members = ["researcher", "math_expert", "coder", "finish"]
system_prompt = f"""你是 Supervisor，负责协调以下专家完成用户任务：
- researcher: 搜索和收集信息
- math_expert: 数学计算和推理
- coder: 编写和运行代码

请根据对话历史和用户需求，决定下一步应该由谁处理。
如果任务已完成，回复 "finish"。

可用选项：{members}
只回复选项名称，不要解释。"""

def supervisor(state: SupervisorState) -> dict:
    """Supervisor 决定下一步路由"""
    response = llm.invoke(
        [SystemMessage(content=system_prompt),
         HumanMessage(content=f"当前对话历史：{state['messages']}\n\n下一步应该由谁处理？")]
    )
    return {"next": response.content.strip()}

# 定义路由函数
def route_supervisor(state: SupervisorState) -> str:
    """根据 Supervisor 的决策路由到对应 Worker"""
    return state["next"]

# 构建图
workflow = StateGraph(SupervisorState)

# 添加节点
workflow.add_node("researcher", research_agent)
workflow.add_node("math_expert", math_agent)
workflow.add_node("coder", coder_agent)
workflow.add_node("supervisor", supervisor)

# 设置入口
workflow.set_entry_point("supervisor")

# Supervisor 用条件边路由
workflow.add_conditional_edges("supervisor", route_supervisor, {
    "researcher": "researcher",
    "math_expert": "math_expert",
    "coder": "coder",
    "finish": END,
})

# 所有 Worker 完成后回到 Supervisor（他判断下一步）
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("math_expert", "supervisor")
workflow.add_edge("coder", "supervisor")

app = workflow.compile()
```

**Supervisor 循环过程**：

```
用户: "帮我比较 GRPO 和 PPO 在数学推理中的性能差异"
    ↓
Supervisor: "需要搜索两种算法的信息" → 路由到 researcher
Worker(researcher): 搜索到 GRPO 和 PPO 的论文要点...
    ↓ 回到 Supervisor
Supervisor: "有数据了，需要计算对比" → 路由到 math_expert
Worker(math_expert): 计算可得 GRPO 在数学任务上比 PPO 提升约 15%...
    ↓ 回到 Supervisor
Supervisor: "信息足够，任务完成" → finish → 输出最终答案
```

**Supervisor 模式 vs 简单并行模式**：

| 维度 | 简单并行模式 | Supervisor 模式 |
|------|------------|-----------------|
| **决策方式** | 所有 Worker 同时执行 | Supervisor 按需调度 |
| **执行顺序** | 无顺序，并行 | 可按顺序迭代（A 结果指导 B 下一步） |
| **成本** | 高（每次全跑） | 低（按需调用） |
| **灵活性** | 低——Worker 间无依赖 | 高——可串行依赖、动态调整 |
| **适用场景** | 独立子任务 | 有依赖关系的多步任务 |

`★ Insight ─────────────────────────────────────`
Supervisor 模式的精妙之处在于**它是一个"LLM 驱动的路由器"**：Supervisor 本身也是 LLM，它通过阅读对话历史来动态决策"下一步让谁干活"。这让整个多 Agent 系统不是预先写死的 if-else，而是真正的智能调度——就像一个真正的团队 leader，会根据当前的进展灵活调配人手。这也意味着 Supervisor 的 system prompt 设计直接决定了整个系统的质量。
`─────────────────────────────────────────────────`

---

## 四、高级特性

### 4.1 Checkpointing（状态持久化）

LangGraph 最强大的特性之一：**自动保存每一步的 State**。

```python
from langgraph.checkpoint import MemorySaver, PostgresSaver

# 开发环境：内存存储
memory = MemorySaver()

# 生产环境：PostgreSQL 存储
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
checkpointer.setup()

# 编译时启用
app = workflow.compile(checkpointer=memory)

# 执行
config = {"configurable": {"thread_id": "conversation-1"}}

# 运行几次后，可以回溯到任意状态
state_snapshot = app.get_state(config)  # 当前状态
history = list(app.get_state_history(config))  # 所有历史状态

# 从某个历史状态重新执行（时间旅行！）
app.invoke(None, config)  # 从上次保存的状态继续
```

**Checkpointing 的应用场景**：
- 对话历史管理：自动记录每次交互
- 错误恢复：执行失败后从断点继续
- 时间旅行：回溯到任意历史状态重新执行
- A/B 测试：从同一中间状态分叉不同执行路径

`★ Insight ─────────────────────────────────────`
Checkpointing 让 LangGraph 的 Agent 有了"存档/读档"能力。这不是简单的日志记录——它保存的是完整的 State 快照，可以在任意时间点重新执行。这在调试复杂 Agent 时是无价的：你不需要从头重跑整个对话，直接从出错的那一步重试即可。
`─────────────────────────────────────────────────`

### 4.2 Streaming（流式输出）

LangGraph 原生支持多种流式模式：

```python
# 1. 流式输出 State 更新（每节点执行后）
for event in app.stream(input_message, config):
    print(event)  # {node_name: {state_update}}

# 2. 流式输出 LLM Token（逐 token 输出）
for event in app.astream_events(input_message, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")

# 3. 自定义流式模式
for mode, chunk in app.stream(input_message, config, stream_mode=["values"]):
    print(mode, chunk)
```

### 4.3 Memory（记忆管理）

```python
# LangGraph 的 Memory 机制
from langgraph.checkpoint import MemorySaver

# 短期记忆：通过 messages 列表（对话历史）
# 长期记忆：通过外部存储 + 自定义 retrieval 节点

workflow = StateGraph(State)

# 添加记忆检索节点
def retrieve_memory(state):
    """从长期记忆中检索相关信息"""
    relevant_memories = memory_store.search(state["messages"][-1].content)
    return {"context": relevant_memories}

workflow.add_node("retrieve_memory", retrieve_memory)
```

### 4.4 Subgraphs（子图复用）

```python
# 将复杂 Agent 拆分为可复用的子图
def create_retrieval_subgraph():
    subgraph = StateGraph(RetrievalState)
    subgraph.add_node("rewrite", rewrite_query)
    subgraph.add_node("search", search_docs)
    subgraph.add_node("rerank", rerank_results)
    subgraph.set_entry_point("rewrite")
    subgraph.add_edge("rewrite", "search")
    subgraph.add_edge("search", "rerank")
    subgraph.add_edge("rerank", END)
    return subgraph.compile()

# 在主图中作为节点使用
main_workflow = StateGraph(MainState)
retrieval_subgraph = create_retrieval_subgraph()
main_workflow.add_node("retrieval", retrieval_subgraph)  # 子图作为节点！
main_workflow.add_node("generate", generate_answer)
main_workflow.set_entry_point("retrieval")
main_workflow.add_edge("retrieval", "generate")
```

---

## 五、LangGraph vs 其他框架

### 5.1 框架对比

| 维度 | LangGraph | CrewAI | AutoGen | Dify |
|------|-----------|--------|---------|------|
| **定位** | 底层图编排框架 | 多 Agent 角色扮演 | 多 Agent 对话 | 可视化工作流 |
| **灵活性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **上手难度** | 中等（需写代码） | 简单 | 中等 | 简单（拖拽） |
| **底层控制** | 完全控制 | 黑盒较多 | 较好 | 受限于 UI |
| **流式支持** | 原生 | 有限 | 有限 | 基础 |
| **持久化** | 内置 Checkpointing | 有限 | 有限 | 内置 |
| **生产就绪** | ✅ | ✅ | ✅ | ✅ |
| **适用场景** | 需要精细控制的复杂 Agent | 简单多 Agent 协作 | Microsoft 生态 | 快速原型/非技术人员 |

### 5.2 什么时候选 LangGraph？

```
✅ 选 LangGraph：
  - 需要复杂的控制流（循环、条件、并行）
  - 需要人机协作（human-in-the-loop）
  - 需要状态持久化和断点恢复
  - 需要生产级的流式输出
  - 需要精细控制每个执行步骤
  - 与 LangChain 生态紧密集成

❌ 不选 LangGraph：
  - 简单的"检索→生成"管道 → 直接用 LangChain Chain
  - 快速原型验证 → 用 CrewAI 或 Dify
  - 完全自主的 Agent → AutoGPT 风格更合适
  - 不想写的代码 → Dify 拖拽式
```

---

## 六、最佳实践

### 6.1 图设计原则

| 原则 | 说明 |
|------|------|
| **单一职责** | 每个节点只做一件事 |
| **显式控制流** | 使用条件边而非在节点内做复杂判断 |
| **可测试性** | 每个节点可独立单元测试 |
| **状态最小化** | State 只包含必要字段，避免膨胀 |
| **错误处理** | 添加 fallback 节点处理异常 |
| **超时控制** | 设置递归限制（recursion_limit） |

### 6.2 常见陷阱

```python
# ❌ 陷阱 1：无限循环
# 解决：设置 recursion_limit
app = workflow.compile()
result = app.invoke(input, {"recursion_limit": 25})  # 最多 25 步

# ❌ 陷阱 2：State 字段覆盖而不是合并
# 解决：使用 Annotated 类型
class State(TypedDict):
    items: Annotated[list, add_items]  # 追加而非覆盖

# ❌ 陷阱 3：忘记添加 END 条件边
# 解决：确保每个条件边都有通向 END 的路径

# ❌ 陷阱 4：节点返回值不完整
# 好的做法：只返回需要更新的字段，不返回完整 State
def good_node(state):
    return {"messages": [new_message]}  # ✅ 只更新 messages
```

### 6.3 调试技巧

```python
# 1. 跟踪每个节点的输入输出
import logging
logging.basicConfig(level=logging.INFO)

# 2. 使用 get_state_history 查看历史
for snapshot in app.get_state_history(config):
    print(f"步骤 {snapshot.metadata['step']}: {snapshot.values}")

# 3. 检查编译后的图结构
print(app.get_graph().draw_ascii())
# 输出：
#       +-----------+
#       | __start__ |
#       +-----------+
#             *
#             *
#        +-------+
#        | agent |
#        +-------+
#        *       *
#       *         *
# +-------+   +---------+
# | tools |   | __end__ |
# +-------+   +---------+
```

---

## 七、面试高频问题

### Q1：什么是 LangGraph？它解决了什么问题？

> LangGraph 是 LangChain 生态中的状态图编排框架，它将 Agent 应用建模为有向图（StateGraph）。核心组件包括 State（共享状态）、Node（计算节点）、Edge（控制流）。它解决了传统 Chain 无法处理循环、条件分支、状态管理的问题，适合构建复杂 LLM Agent 应用。

### Q2：LangGraph 的核心组件有哪些？

> 1. **State**：图中所有节点共享的数据结构，支持 TypedDict 定义，通过 Annotated 控制合并规则
> 2. **Node**：计算单元，接收 State 返回部分更新，可以是 LLM 调用、工具执行或任意 Python 函数
> 3. **Edge**：控制流——普通边（直接连接）、条件边（根据函数返回值路由）、入口/出口边
> 4. **Graph**：编译后成为可执行 Runnable，支持 invoke/stream/astream

### Q3：LangGraph 中的 State 是如何工作的？

> State 是图中节点间共享的数据结构。每个节点接收完整 State，返回一个只包含需要更新字段的字典，LangGraph 内部将其与当前 State 合并。对于 `messages` 这类需要追加的字段，使用 `Annotated[list, add_messages]` 声明 reducer 函数来指定合并策略（追加而非覆盖）。

### Q4：LangGraph 和 LangChain 的 Chain 有什么区别？

> - **Chain**：预定义的线性流程，输入→步骤1→步骤2→...→输出。无法处理循环和复杂条件分支
> - **LangGraph**：基于有向图的控制流，支持循环、条件分支、并行、人机协作。Chain 适合简单管道，LangGraph 适合复杂 Agent 应用
> - 可以把 LangGraph 看作是"Chain 的超集"——任何 Chain 都可以用 LangGraph 图表示，反之不然

### Q5：什么是 Checkpointing？它有什么用？

> Checkpointing 是 LangGraph 的状态持久化机制——每次节点执行后自动保存 State 快照。用途：(1) 对话历史管理；(2) 错误恢复（断点续跑）；(3) 时间旅行（回溯到任意历史状态）；(4) 人机协作（中断后恢复执行）。支持多种后端：MemorySaver（开发）、PostgresSaver（生产）、SqliteSaver 等。

### Q6：LangGraph 如何实现 Human-in-the-Loop？

> 通过 `interrupt()` 函数在节点执行中暂停，等待人工输入后恢复。结合 Checkpointing，中断的 State 被持久化，人工审批后可通过 `Command(resume=...)` 恢复执行。典型场景：敏感操作需要人工确认、复杂决策需要专家审核。

### Q7：Agent 循环在 LangGraph 中如何实现？

> Agent 循环是最核心的模式：`agent 节点 → 条件边判断 → 如果有 tool_calls 则执行 tools 节点 → 回到 agent 节点 → 如果无 tool_calls 则结束`。这个循环让 LLM 可以反复调用工具直到得出最终答案。LangGraph 的预构建 `create_react_agent` 封装了此模式。

### Q8：何时选择 LangGraph 而非 CrewAI 或 Dify？

> - **选 LangGraph**：需要精细控制流、复杂循环/分支、生产级流式输出、人机协作、状态持久化
> - **选 CrewAI**：简单的多 Agent 角色扮演、快速原型
> - **选 Dify**：非技术人员使用、可视化工作流、无需写代码

---

## 八、总结

**LangGraph 的核心思想一句话**：

> 把 Agent 应用建模为有向图——节点做计算，边管流程，状态贯穿始终。

**核心模式速查**：

| 模式 | 图结构 | 适用场景 |
|------|--------|---------|
| **Agent 循环** | agent ↔ tools | 工具调用 Agent |
| **Supervisor 模式** | Supervisor → 按需调度 Worker → 汇总 | 多 Agent 协作、复杂任务分解 |
| **条件路由** | 条件边多路分发 | 根据内容选择处理路径 |
| **并行分支** | 单源→多目标→汇聚 | 独立子任务并行 |
| **Human-in-the-Loop** | 中断→等待→恢复 | 敏感操作审批 |
| **多步 RAG** | 检索→生成→评估→循环 | 复杂 QA 系统 |

**关键 API 速查**：

```python
# 构建图
from langgraph.graph import StateGraph, END
wf = StateGraph(StateType)
wf.add_node("name", node_func)
wf.set_entry_point("name")
wf.add_edge("a", "b")
wf.add_conditional_edges("a", router_func, {"route1": "b", "route2": END})

# 编译和运行
app = wf.compile(checkpointer=MemorySaver())
result = app.invoke(input, config)
for event in app.stream(input, config):
    ...

# 预构建 Agent
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model, tools)
```

**一句话记忆**：

> LangGraph = 用图（Graph）管流程，用状态（State）管数据，用条件边管决策，用 Checkpointing 存档/读档。如果 LangChain Chain 是"流水线"，LangGraph 就是"智能交通系统"——能循环、能分支、能并线、能暂停。Supervisor 模式则是"智能调度中心"——一个主管 Agent 按需调度多个专业 Worker，动态分配任务。

---

**相关笔记**：
- [[Agent与工具调用]] - Agent 架构基础
- [[Agentic RAG]] - Agent 增强的 RAG（LangGraph 核心应用场景）
- [[RAG]] - RAG 总览
- [[Prompt Engineering与CoT]] - 提示工程基础
