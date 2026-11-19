# Agent 与工具调用

> Prompt Engineering 和 CoT 让模型能更好地推理，但模型本身**无法获取实时信息、无法执行代码、无法访问外部系统**。Agent（智能体）和 Function Calling（函数调用）让大模型能够使用工具、与外部世界交互，是大模型应用的核心能力。如果你想快速构建 Agent 应用，可以参考 [[Coze平台使用教程]]。

---

## 一、什么是 Agent

### 1.1 Agent 的定义

**Agent（智能体）**：能够感知环境、做出决策、执行行动的自主系统。在大模型语境下，Agent = LLM + 工具 + 记忆 + 规划。

```
传统 LLM：
  用户输入 → LLM → 输出
  （只能生成文本，无法行动）

Agent：
  用户输入 → LLM → 决策 → 调用工具 → 获取结果 → 继续推理 → 输出
  （能思考、能行动、能观察结果）
```

### 1.2 Agent 的核心组件

| 组件 | 作用 | 示例 |
|------|------|------|
| **LLM（大脑）** | 理解任务、做出决策 | GPT-4、Claude |
| **工具（Tools）** | 执行具体操作 | 搜索引擎、计算器、代码执行器 |
| **记忆（Memory）** | 存储历史信息 | 对话历史、检索结果 |
| **规划（Planning）** | 分解任务、制定计划 | CoT、ReAct |

### 1.3 Agent 的工作流程

```
┌─────────────────────────────────────────────────────┐
│                    Agent 循环                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│   用户请求                                           │
│     ↓                                               │
│   ┌─────────────┐                                    │
│   │  思考（LLM） │ ← 记忆提供上下文                    │
│   └──────┬──────┘                                    │
│          ↓                                           │
│   ┌─────────────┐                                    │
│   │  决策：行动？ │ → 是 → 执行工具 → 观察结果 → 回到思考 │
│   └──────┬──────┘                                    │
│          ↓ 否                                        │
│   输出最终答案                                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 二、Function Calling（函数调用）

### 2.1 什么是 Function Calling

**Function Calling**：让大模型在回答问题时，可以调用外部函数/API 获取信息或执行操作。

```
传统方式：
  用户: "北京今天天气怎么样？"
  LLM: "我无法获取实时天气信息。"

Function Calling 方式：
  用户: "北京今天天气怎么样？"
  LLM: [调用 get_weather(city="北京")]
  系统: {temperature: 25, weather: "晴"}
  LLM: "北京今天 25°C，天气晴朗。"
```

### 2.2 Function Calling 的实现

#### OpenAI 的 Function Calling

```python
import openai

# 定义可用的函数
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索网页获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 调用模型
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto"  # 让模型决定是否调用函数
)

# 解析响应
if response.choices[0].message.tool_calls:
    # 模型决定调用函数
    function_call = response.choices[0].message.tool_calls[0]
    function_name = function_call.function.name
    arguments = json.loads(function_call.function.arguments)
    
    # 执行函数
    if function_name == "get_weather":
        result = get_weather(**arguments)
    
    # 将结果返回给模型
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": function_call.id,
        "content": json.dumps(result)
    })
    
    # 模型生成最终回答
    final_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        tools=tools
    )
```

#### Anthropic 的 Tool Use

```python
import anthropic

# 定义工具
tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    }
]

# 调用模型
response = anthropic.Client().messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "北京天气怎么样？"}]
)

# 检查是否需要调用工具
if response.stop_reason == "tool_use":
    tool_call = response.content[0]
    result = get_weather(**tool_call.input)
    
    # 继续对话
    response = anthropic.Client().messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "北京天气怎么样？"},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": f"工具返回结果：{result}"}
        ]
    )
```

### 2.3 Function Calling 的关键设计

| 设计点 | 说明 | 示例 |
|--------|------|------|
| **函数描述** | 清晰描述函数用途 | "获取天气信息" |
| **参数描述** | 每个参数的含义和类型 | "city: 城市名称" |
| **返回格式** | 统一的返回结构 | JSON 格式 |
| **错误处理** | 函数失败时的处理 | 返回错误信息 |
| **权限控制** | 限制可调用的函数 | 白名单机制 |

---

## 三、Agent 框架：ReAct

### 3.1 ReAct 的原理

**ReAct（Reasoning + Acting）**：结合推理（Reasoning）和行动（Acting）的框架。

```
ReAct 循环：

Thought: 我需要获取北京的天气信息
Action: get_weather(city="北京")
Observation: {temperature: 25, weather: "晴"}
Thought: 我已经获取到天气信息，可以回答了
Action: finish(answer="北京今天 25°C，天气晴朗。")
```

### 3.2 ReAct 的实现

```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []
    
    def run(self, query):
        # 初始提示
        prompt = f"""你是一个智能助手，可以使用以下工具：
{self.format_tools()}

请按照以下格式回答：
Thought: [你的思考]
Action: [工具名称](参数)
Observation: [工具返回结果]
... (可以重复 Thought/Action/Observation)
Thought: [最终思考]
Action: finish(answer=[最终答案])

问题：{query}
"""
        
        while True:
            # 调用 LLM
            response = self.llm.generate(prompt)
            
            # 解析响应
            thought, action = self.parse_response(response)
            
            # 记录思考
            self.memory.append({"thought": thought})
            
            # 执行动作
            if action.startswith("finish"):
                # 提取最终答案
                answer = self.extract_answer(action)
                return answer
            else:
                # 执行工具
                result = self.execute_tool(action)
                self.memory.append({"observation": result})
                
                # 更新提示
                prompt += f"\nThought: {thought}\nAction: {action}\nObservation: {result}\n"
    
    def format_tools(self):
        """格式化工具描述"""
        return "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
    
    def parse_response(self, response):
        """解析 LLM 响应，提取 Thought 和 Action"""
        lines = response.strip().split("\n")
        thought = ""
        action = ""
        
        for line in lines:
            if line.startswith("Thought:"):
                thought = line[9:].strip()
            elif line.startswith("Action:"):
                action = line[8:].strip()
        
        return thought, action
    
    def execute_tool(self, action):
        """执行工具调用"""
        # 解析工具名称和参数
        tool_name = action.split("(")[0]
        params = self.parse_params(action)
        
        # 查找并执行工具
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.execute(**params)
        
        return "Error: Unknown tool"
```

### 3.3 ReAct 的优势

| 优势 | 说明 |
|------|------|
| **可解释性** | Thought 步骤让推理过程透明 |
| **灵活性** | 可以根据观察调整策略 |
| **错误恢复** | 可以从失败的工具调用中恢复 |
| **多步推理** | 支持复杂的多步任务 |

---

## 四、Agent 架构模式

### 4.1 单 Agent 架构

```
用户请求
    ↓
┌─────────────┐
│   LLM Agent │ ← 工具列表
└──────┬──────┘
       ↓
   ┌───┴───┐
   │ 工具1 │
   │ 工具2 │
   │ 工具3 │
   └───────┘
       ↓
   最终答案
```

**适用场景**：任务简单，工具明确。

### 4.2 Multi-Agent 架构（多智能体）

```
用户请求
    ↓
┌─────────────┐
│   协调者     │
└──────┬──────┘
       ↓
   ┌───┴───────────┐
   ↓               ↓
┌──────┐      ┌──────┐
│Agent1│      │Agent2│
│(搜索)│      │(代码)│
└──────┘      └──────┘
   ↓               ↓
   └───────┬───────┘
           ↓
       最终答案
```

**多 Agent 的优势**：

| 优势 | 说明 |
|------|------|
| **专业化** | 每个 Agent 专注一个领域 |
| **并行化** | 多个 Agent 可以同时工作 |
| **可扩展性** | 新增 Agent 不影响现有系统 |
| **容错性** | 一个 Agent 失败不影响其他 |

### 4.3 AutoGPT / MetaGPT 等框架

```python
# AutoGPT 风格的 Agent
from autogpt import Agent, Command

class ResearchAgent(Agent):
    def __init__(self):
        super().__init__(
            name="研究员",
            role="搜索和分析信息",
            tools=[WebSearchTool(), SummarizeTool()]
        )
    
    def think(self, observation):
        # 基于观察进行思考
        if "需要信息" in observation:
            return "我需要搜索相关信息"
        elif "信息足够" in observation:
            return "我可以总结答案了"
    
    def act(self, thought):
        # 根据思考执行动作
        if "搜索" in thought:
            return self.tools[0].execute()
        elif "总结" in thought:
            return self.tools[1].execute()
```

---

## 五、Agent 的挑战

### 5.1 常见问题

| 问题 | 说明 | 解决方案 |
|------|------|---------|
| **幻觉调用** | 模型编造不存在的函数 | 严格的函数定义验证 |
| **参数错误** | 生成错误的参数格式 | 参数类型校验、示例引导 |
| **无限循环** | 反复调用同一个函数 | 设置最大调用次数 |
| **权限越界** | 调用不该调用的函数 | 白名单、权限控制 |
| **结果误解** | 错误理解工具返回 | 更清晰的返回格式 |

### 5.2 安全考虑

```
安全措施：

1. 输入验证
   - 验证用户输入是否安全
   - 防止注入攻击

2. 输出过滤
   - 过滤敏感信息
   - 防止信息泄露

3. 权限控制
   - 限制可调用的函数
   - 限制执行的操作

4. 沙箱执行
   - 代码在沙箱中执行
   - 限制文件系统访问

5. 审计日志
   - 记录所有工具调用
   - 便于事后审查
```

### 5.3 可靠性提升

| 方法 | 说明 |
|------|------|
| **重试机制** | 工具调用失败时重试 |
| **回退策略** | 主工具失败时尝试备选工具 |
| **验证步骤** | 执行后验证结果是否正确 |
| **人工确认** | 敏感操作需要人工确认 |
| **超时控制** | 设置执行时间限制 |

---

## 六、面试高频问题

### Q1：什么是 Agent？它和普通 LLM 有什么区别？

> Agent（智能体）是能够感知环境、做出决策、执行行动的自主系统，由 LLM + 工具 + 记忆 + 规划组成。普通 LLM 只能生成文本，无法获取实时信息或执行操作。Agent 能够调用工具获取信息（如搜索、API）、执行操作（如代码执行、文件操作），并通过观察结果进行多步推理，完成复杂任务。

### Q2：什么是 Function Calling？它是如何工作的？

> Function Calling 让大模型在回答时可以调用外部函数/API。工作流程：(1) 定义可用的函数（名称、描述、参数）；(2) 用户提问时，模型判断是否需要调用函数；(3) 如果需要，模型生成函数名和参数（JSON 格式）；(4) 系统执行函数并返回结果；(5) 模型基于结果生成最终回答。关键是函数定义要清晰，让模型理解何时调用、如何传参。

### Q3：ReAct 框架是什么？它的优势是什么？

> ReAct（Reasoning + Acting）是结合推理和行动的 Agent 框架。循环过程：Thought（思考下一步）→ Action（执行工具）→ Observation（观察结果）→ 继续循环。优势：(1) 可解释性——Thought 步骤让推理过程透明；(2) 灵活性——可以根据观察调整策略；(3) 错误恢复——可以从失败中恢复；(4) 支持多步推理。

### Q4：Multi-Agent 架构是什么？什么时候用它？

> Multi-Agent 是多个 Agent 协作完成任务的架构。每个 Agent 专注一个领域（如搜索、代码、分析），由协调者分配任务并整合结果。适用场景：(1) 任务复杂，需要多种能力；(2) 需要并行处理；(3) 需要专业化分工。优势是可扩展、可维护、容错性好。劣势是协调复杂、通信开销大。

### Q5：Agent 常见的问题有哪些？如何解决？

> 常见问题：(1) **幻觉调用**——模型编造不存在的函数，解决：严格函数定义验证；(2) **参数错误**——参数格式错误，解决：类型校验、提供示例；(3) **无限循环**——反复调用，解决：设置最大调用次数；(4) **权限越界**——调用不该调用的函数，解决：白名单控制。安全措施包括输入验证、输出过滤、沙箱执行、审计日志。

### Q6：如何设计一个好的 Agent 系统？

> 设计原则：(1) **清晰的工具定义**——每个工具的用途、参数、返回值要明确；(2) **合理的提示设计**——引导模型正确使用工具；(3) **错误处理**——工具失败时有回退策略；(4) **权限控制**——限制可调用的工具；(5) **可观察性**——记录所有决策和调用；(6) **人机协作**——敏感操作需要人工确认。架构上根据任务复杂度选择单 Agent 或 Multi-Agent。

---

## 七、总结

| 概念 | 核心要点 |
|------|---------|
| Agent | LLM + 工具 + 记忆 + 规划 |
| Function Calling | 模型调用外部函数获取信息或执行操作 |
| ReAct | 推理（Thought）+ 行动（Action）+ 观察（Observation）循环 |
| Multi-Agent | 多个专业 Agent 协作 |

**一句话记忆**：

> Agent 就是"有手有脚的 AI"——不只是动嘴（生成文本），还能动手（调用工具）。Function Calling 是"AI 的手"，ReAct 是"AI 的思考-行动循环"，Multi-Agent 是"AI 团队协作"。

---

**相关笔记**：
- [[Agentic RAG]] - Agent 增强的 RAG 应用
- [[LangGraph]] - 用状态图构建复杂 Agent 应用（核心框架）
- [[RAG]] - 检索增强生成
- [[Prompt Engineering与CoT]] - Agent 的推理基础
