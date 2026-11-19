# Agentic RAG：智能体增强的检索增强生成

> 传统 RAG 是"检索→生成"的单次流程，遇到复杂问题时往往力不从心。Agentic RAG 将 AI Agent 的自主决策能力与 RAG 结合，让系统能够**多步推理、迭代检索、反思验证**，是当前 RAG 技术的前沿方向。

---

## 一、为什么需要 Agentic RAG？

### 1.1 传统 RAG 的局限

```
传统 RAG 流程：
  用户提问 → 检索 Top-K → 拼接 prompt → LLM 生成 → 返回答案

问题场景：
  1. 多跳问题："GRPO 和 PPO 的区别是什么？"
     → 需要先检索 GRPO，再检索 PPO，最后对比
     → 传统 RAG 只检索一次，可能漏掉相关信息

  2. 需要验证："这个公式推导对吗？"
     → 需要检索原始论文，逐步验证
     → 传统 RAG 无法自主验证

  3. 信息不完整："如何从零实现一个 RAG 系统？"
     → 需要多次检索不同组件的信息
     → 传统 RAG 只返回一次检索结果

  4. 需要推理："根据这些数据，预测明年的趋势"
     → 需要检索数据 + 调用计算工具 + 推理
     → 传统 RAG 无法调用外部工具
```

### 1.2 Agentic RAG 的核心优势

| 能力 | 传统 RAG | Agentic RAG |
|------|---------|-------------|
| **检索次数** | 单次 | 多次迭代 |
| **决策能力** | 无，固定流程 | 有，自主决策 |
| **工具使用** | 仅检索 | 多种工具 |
| **反思验证** | 无 | 有，自我纠错 |
| **多步推理** | 有限 | 支持多跳推理 |
| **适应性** | 固定策略 | 动态调整 |

---

## 二、Agentic RAG 核心架构

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────┐
│                    Agentic RAG 系统                      │
│                                                         │
│  用户提问                                                │
│      ↓                                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Agent 决策引擎                      │   │
│  │  - 分析查询意图                                   │   │
│  │  - 制定检索策略                                   │   │
│  │  - 决定是否需要检索                               │   │
│  │  - 选择使用哪些工具                               │   │
│  └─────────────────────────────────────────────────┘   │
│      ↓                                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │              工具集（Tools）                      │   │
│  │  - 向量检索                                      │   │
│  │  - 关键词检索                                     │   │
│  │  - SQL 查询                                      │   │
│  │  - API 调用                                      │   │
│  │  - 代码执行                                      │   │
│  │  - 计算器                                        │   │
│  └─────────────────────────────────────────────────┘   │
│      ↓                                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │              反思与验证                           │   │
│  │  - 检查检索结果是否完整                           │   │
│  │  - 验证推理过程是否正确                           │   │
│  │  - 决定是否需要重新检索                           │   │
│  └─────────────────────────────────────────────────┘   │
│      ↓                                                  │
│  生成最终答案                                            │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

```
1. Agent 控制器
   - 理解用户意图
   - 制定执行计划
   - 管理对话状态

2. 工具集
   - 检索工具：向量检索、关键词检索
   - 计算工具：计算器、代码执行
   - 外部 API：天气、股票、翻译等

3. 反思模块
   - 评估检索结果质量
   - 验证推理过程
   - 决定是否需要迭代

4. 记忆系统
   - 短期记忆：当前对话上下文
   - 长期记忆：历史交互记录
```

---

## 三、Agentic RAG 关键技术

### 3.1 多步推理（Multi-hop Reasoning）

```
用户问题："比较 GRPO 和 PPO 在 RLHF 中的应用差异"

Agent 推理过程：
  Step 1: 分析问题 → 需要同时了解 GRPO 和 PPO
  Step 2: 检索 GRPO → 找到相关文档
  Step 3: 检索 PPO → 找到相关文档
  Step 4: 对比分析 → 生成结构化对比
  Step 5: 验证结论 → 检查是否有遗漏
```

**实现方式**：

```python
class MultiHopAgent:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.memory = []
    
    def answer(self, question):
        # 1. 分解问题
        sub_questions = self.decompose(question)
        
        # 2. 逐步检索和推理
        context = []
        for sq in sub_questions:
            docs = self.retriever.retrieve(sq)
            context.extend(docs)
            reasoning = self.reason(sq, docs)
            self.memory.append(reasoning)
        
        # 3. 综合生成答案
        final_answer = self.synthesize(question, context, self.memory)
        return final_answer
    
    def decompose(self, question):
        """将复杂问题分解为子问题"""
        prompt = f"""请将以下问题分解为多个子问题：
        
问题：{question}

子问题（每行一个）："""
        return self.llm.generate(prompt).split('\n')
```

### 3.2 自适应检索（Adaptive Retrieval）

```
Agent 根据查询复杂度动态决定检索策略：

查询："1+1=?"
  → 判断：常识问题，无需检索
  → 直接回答：2

查询："RAG 是什么？"
  → 判断：简单定义问题
  → 检索：1次向量检索
  → 回答

查询："如何从零构建一个生产级 RAG 系统？"
  → 判断：复杂问题，需要多方面信息
  → 检索：多次检索不同组件
  → 回答
```

**实现方式**：

```python
class AdaptiveRAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def answer(self, question):
        # 1. 分析查询复杂度
        complexity = self.analyze_complexity(question)
        
        # 2. 根据复杂度选择策略
        if complexity == "simple":
            return self.llm.generate(question)
        elif complexity == "medium":
            docs = self.retriever.retrieve(question, top_k=3)
            return self.generate_with_docs(question, docs)
        elif complexity == "complex":
            return self.multi_hop_answer(question)
    
    def analyze_complexity(self, question):
        """分析查询复杂度"""
        prompt = f"""判断以下问题的复杂度（simple/medium/complex）：

问题：{question}

复杂度："""
        return self.llm.generate(prompt).strip()
```

### 3.3 反思与验证（Reflection & Verification）

```
Agent 对生成的答案进行自我验证：

原始答案："GRPO 比 PPO 更高效，因为..."

反思过程：
  1. 检查：是否准确描述了 GRPO 的原理？
  2. 检查：是否准确描述了 PPO 的原理？
  3. 检查：对比是否客观？
  4. 发现：缺少具体数据支撑
  5. 行动：重新检索相关论文中的实验数据
  6. 修正：补充数据后重新生成答案
```

**实现方式**：

```python
class ReflectiveAgent:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def answer_with_reflection(self, question, max_reflections=3):
        # 1. 生成初始答案
        answer = self.generate_initial_answer(question)
        
        # 2. 反思和验证
        for i in range(max_reflections):
            # 反思
            reflection = self.reflect(question, answer)
            
            if reflection["needs_improvement"]:
                # 根据反思结果改进
                if reflection["needs_more_docs"]:
                    # 重新检索
                    new_docs = self.retriever.retrieve(
                        reflection["improvement_query"]
                    )
                    answer = self.improve_with_docs(answer, new_docs)
                else:
                    # 直接改进
                    answer = self.improve_answer(answer, reflection)
            else:
                break
        
        return answer
    
    def reflect(self, question, answer):
        """反思答案质量"""
        prompt = f"""请评估以下答案的质量：

问题：{question}
答案：{answer}

评估：
1. 是否完整回答了问题？
2. 是否有事实错误？
3. 是否需要更多信息？
4. 改进建议："""

        reflection = self.llm.generate(prompt)
        return self.parse_reflection(reflection)
```

### 3.4 工具调用（Tool Use）

```
Agent 根据需要调用不同工具：

用户问题："北京今天天气怎么样？并计算 25℃ 转换为华氏度是多少？"

Agent 决策：
  1. 需要获取天气 → 调用天气 API
  2. 需要温度转换 → 调用计算器
  3. 综合生成答案

执行过程：
  Tool 1: weather_api("北京") → "晴，25℃"
  Tool 2: calculator("25 * 9/5 + 32") → "77℉"
  Answer: "北京今天晴，25℃（77℉）"
```

**实现方式**：

```python
class ToolUsingAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools  # {"weather": weather_api, "calc": calculator}
    
    def answer(self, question):
        # 1. 分析需要哪些工具
        tools_needed = self.analyze_tools_needed(question)
        
        # 2. 依次调用工具
        tool_results = {}
        for tool_name in tools_needed:
            tool = self.tools[tool_name]
            result = self.call_tool(tool, question)
            tool_results[tool_name] = result
        
        # 3. 综合生成答案
        return self.generate_with_tools(question, tool_results)
    
    def analyze_tools_needed(self, question):
        """分析需要哪些工具"""
        prompt = f"""分析以下问题需要使用哪些工具：

问题：{question}

可用工具：{list(self.tools.keys())}

需要的工具（逗号分隔）："""
        return self.llm.generate(prompt).split(',')
    
    def call_tool(self, tool, question):
        """调用工具"""
        # 根据工具类型构造参数
        if tool.name == "weather":
            return tool(city="北京")
        elif tool.name == "calc":
            return tool(expression=self.extract_expression(question))
```

---

## 四、Agentic RAG 实现模式

### 4.1 单 Agent 模式

```python
# 使用 LangGraph 实现
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 定义状态
class AgentState:
    messages: list
    context: str
    answer: str

# 定义节点
def retrieve(state):
    """检索节点"""
    question = state.messages[-1].content
    docs = retriever.retrieve(question)
    state.context = "\n".join([doc.content for doc in docs])
    return state

def reason(state):
    """推理节点"""
    prompt = f"""基于以下上下文回答问题：

上下文：{state.context}

问题：{state.messages[-1].content}

回答："""
    state.answer = llm.generate(prompt)
    return state

def reflect(state):
    """反思节点"""
    prompt = f"""评估以下答案是否完整：

问题：{state.messages[-1].content}
答案：{state.answer}

是否需要更多信息？（yes/no）："""
    needs_more = llm.generate(prompt).strip()
    return "retrieve" if needs_more == "yes" else "end"

# 构建图
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve)
graph.add_node("reason", reason)
graph.add_edge("retrieve", "reason")
graph.add_conditional_edges("reason", reflect, {"retrieve": "retrieve", "end": END})
```

### 4.2 Multi-Agent 模式

```python
# 多 Agent 协作
class MultiAgentRAG:
    def __init__(self):
        self.planner = PlannerAgent()      # 规划 Agent
        self.retriever = RetrieverAgent()  # 检索 Agent
        self.analyst = AnalystAgent()      # 分析 Agent
        self.verifier = VerifierAgent()    # 验证 Agent
    
    def answer(self, question):
        # 1. 规划 Agent 制定计划
        plan = self.planner.plan(question)
        
        # 2. 检索 Agent 执行检索
        docs = []
        for step in plan["retrieval_steps"]:
            retrieved = self.retriever.retrieve(step)
            docs.extend(retrieved)
        
        # 3. 分析 Agent 生成答案
        answer = self.analyst.analyze(question, docs)
        
        # 4. 验证 Agent 验证答案
        verification = self.verifier.verify(question, answer)
        
        if verification["needs_revision"]:
            # 重新执行
            return self.answer_with_revision(question, verification)
        
        return answer
```

### 4.3 ReAct 模式

```
ReAct（Reasoning + Acting）是 Agentic RAG 的经典模式：

循环执行：
  Thought: 分析当前状态，决定下一步行动
  Action: 调用工具（检索、计算等）
  Observation: 获取工具返回结果
  ... 重复直到得出最终答案

示例：
  Thought: 用户问 GRPO 和 PPO 的区别，我需要分别检索这两个概念
  Action: retrieve("GRPO 算法原理")
  Observation: 找到 3 篇关于 GRPO 的文档...

  Thought: 已经获取 GRPO 信息，现在检索 PPO
  Action: retrieve("PPO 算法原理")
  Observation: 找到 3 篇关于 PPO 的文档...

  Thought: 已经获取两个算法的信息，可以对比生成答案了
  Action: generate_answer("对比 GRPO 和 PPO 的区别")
  Observation: 生成了对比分析...
```

**实现方式**：

```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    def answer(self, question, max_steps=10):
        history = []
        
        for step in range(max_steps):
            # Thought
            thought = self.think(question, history)
            history.append({"thought": thought})
            
            # Action
            action = self.decide_action(thought)
            if action["type"] == "finish":
                return action["answer"]
            
            # Observation
            observation = self.execute_action(action)
            history.append({"action": action, "observation": observation})
        
        return "达到最大步数限制"
    
    def think(self, question, history):
        """思考下一步"""
        prompt = f"""基于当前信息，分析下一步应该做什么：

问题：{question}
历史记录：{history}

思考："""
        return self.llm.generate(prompt)
    
    def decide_action(self, thought):
        """决定行动"""
        prompt = f"""根据思考决定下一步行动：

思考：{thought}

可用工具：{list(self.tools.keys())}

行动（JSON 格式）："""
        return self.llm.generate_json(prompt)
```

---

## 五、Agentic RAG 框架

### 5.1 LangGraph

```python
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# 定义工具
tools = [
    Tool(name="vector_search", func=vector_retriever.search),
    Tool(name="web_search", func=web_search),
    Tool(name="calculator", func=calculator),
]

# 构建 Agent
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4"),
    tools=tools,
    state_modifier="你是一个智能助手，可以使用工具回答问题。"
)

# 运行
result = agent.invoke({"messages": [("user", "GRPO 和 PPO 的区别")]})
```

### 5.2 LlamaIndex

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# 创建工具
query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        name="vector_search",
        description="搜索本地知识库"
    ),
    QueryEngineTool.from_defaults(
        query_engine=web_query_engine,
        name="web_search",
        description="搜索互联网"
    ),
]

# 创建 Agent
agent = ReActAgent.from_tools(
    tools=query_engine_tools,
    llm=llm,
    verbose=True
)

# 运行
response = agent.chat("如何从零构建 RAG 系统？")
```

### 5.3 CrewAI

```python
from crewai import Agent, Task, Crew

# 定义 Agent
researcher = Agent(
    role="研究员",
    goal="检索和收集相关信息",
    backstory="你是一个专业的信息检索专家",
    tools=[search_tool]
)

analyst = Agent(
    role="分析师",
    goal="分析和综合信息",
    backstory="你是一个擅长分析和总结的专家",
    llm=llm
)

# 定义任务
research_task = Task(
    description="检索关于 Agentic RAG 的最新研究",
    agent=researcher
)

analysis_task = Task(
    description="分析检索结果并生成报告",
    agent=analyst
)

# 组建团队
crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    verbose=True
)

# 执行
result = crew.kickoff()
```

---

## 六、Agentic RAG 最佳实践

### 6.1 何时使用 Agentic RAG

```
适合使用：
  ✓ 多跳推理问题
  ✓ 需要验证的答案
  ✓ 复杂的分析任务
  ✓ 需要调用多种工具
  ✓ 信息不完整需要迭代检索

不适合使用：
  ✗ 简单的事实查询
  ✗ 对延迟要求极高
  ✗ 预算有限（Agent 调用成本高）
  ✗ 问题明确，传统 RAG 足够
```

### 6.2 性能优化

```python
# 1. 限制最大步数
max_steps = 10  # 防止无限循环

# 2. 缓存工具结果
@lru_cache(maxsize=100)
def cached_search(query):
    return search(query)

# 3. 并行工具调用
async def parallel_tools(queries):
    tasks = [search(q) for q in queries]
    return await asyncio.gather(*tasks)

# 4. 提前终止
if confidence > threshold:
    return answer  # 置信度足够高时直接返回
```

### 6.3 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **答案质量** | 回答的准确性和完整性 | 人工评估/GPT-4 评估 |
| **检索效率** | 检索到相关信息的效率 | 检索次数/答案质量 |
| **延迟** | 端到端响应时间 | 总耗时 |
| **成本** | Token 消耗和 API 调用费用 | Token 数 × 单价 |
| **成功率** | 成功完成任务的比例 | 成功任务数/总任务数 |

`★ Insight ─────────────────────────────────────`
**Agentic RAG 的本质是用"思考"换"效果"**：传统 RAG 是"拿到就用"，Agentic RAG 是"先想清楚再用"。Agent 会思考"这个信息够不够？需不需要再找？答案对不对？"，这种反思能力让它能处理复杂问题，但代价是更高的延迟和成本。
`─────────────────────────────────────────────────`

---

## 七、面试高频问题

### Q1：Agentic RAG 和传统 RAG 有什么区别？

> - **传统 RAG**：单次检索→生成，固定流程，无法自主决策
> - **Agentic RAG**：多步推理，迭代检索，反思验证，可以调用多种工具
> 
> 核心区别是 Agentic RAG 具备自主决策能力，能根据问题复杂度动态调整策略。

### Q2：Agentic RAG 的核心组件有哪些？

> 1. **Agent 控制器**：理解意图，制定计划
> 2. **工具集**：检索、计算、API 调用等
> 3. **反思模块**：验证答案质量，决定是否迭代
> 4. **记忆系统**：维护对话上下文和历史信息

### Q3：什么是 ReAct 模式？

> ReAct（Reasoning + Acting）是 Agentic RAG 的经典模式。循环执行：Thought（思考下一步）→ Action（调用工具）→ Observation（获取结果），直到得出最终答案。这种模式让 Agent 能够动态决策，而非固定流程。

### Q4：Agentic RAG 有哪些常见问题？

> 1. **无限循环**：Agent 反复调用工具但无法收敛 → 设置最大步数
> 2. **成本过高**：多次调用 LLM 和工具 → 优化策略，设置预算
> 3. **延迟过大**：多步推理耗时 → 并行工具调用，提前终止
> 4. **幻觉传播**：错误信息在迭代中放大 → 加强验证机制

---

## 八、总结

**Agentic RAG 的核心思想**：

> 让 RAG 系统具备 Agent 的自主决策能力，通过多步推理、迭代检索、反思验证来处理复杂问题。

**关键能力**：

| 能力 | 说明 | 价值 |
|------|------|------|
| 多步推理 | 将复杂问题分解为子问题 | 处理多跳问题 |
| 自适应检索 | 根据复杂度动态调整策略 | 提高效率 |
| 工具调用 | 调用多种工具完成任务 | 扩展能力边界 |
| 反思验证 | 自我检查和纠错 | 提高准确性 |

**适用场景**：

```
✓ 多跳推理问题（如对比分析）
✓ 需要验证的答案（如事实核查）
✓ 复杂的分析任务（如报告生成）
✓ 需要调用多种工具（如数据分析）
✗ 简单的事实查询
✗ 对延迟要求极高
```

**推荐框架**：

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| **LangGraph** | LangChain 生态，灵活 | 定制化需求 |
| **LlamaIndex** | 检索增强专精 | RAG 场景 |
| **CrewAI** | 多 Agent 协作 | 团队协作场景 |

**完整流程**：

```
用户提问 → Agent 分析意图 → 制定计划 → 迭代检索+推理 → 反思验证 → 生成答案
```

---

**相关笔记**：
- [[RAG]] - RAG 总览
- [[Agent与工具调用]] - Agent 架构基础
- [[LangGraph]] - 构建 Agentic RAG 的核心框架
- [[混合检索]] - 检索技术
- [[重排序Reranking]] - 检索优化
- [[查询改写QueryRewriting]] - 查询优化
