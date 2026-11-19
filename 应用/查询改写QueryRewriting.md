# 查询改写 Query Rewriting：优化检索查询

> 用户的原始查询往往不够精确，包含口语化表达、缩写、歧义等。查询改写用 LLM 或规则将原始查询优化为更适合检索的形式，是提升 RAG 检索效果的重要环节。

---

## 一、为什么需要查询改写？

### 1.1 用户查询的问题

```
问题 1：口语化表达
  用户查询："RAG 是啥？"
  问题：太口语化，向量检索可能找不到相关文档
  优化：→ "什么是检索增强生成（Retrieval-Augmented Generation）？"

问题 2：缩写/简称
  用户查询："GRPO 怎么用？"
  问题：GRPO 是缩写，文档可能用全称
  优化：→ "Group Relative Policy Optimization（GRPO）算法的使用方法"

问题 3：歧义
  用户查询："Python 的 list 和 tuple 的区别"
  问题：可能有很多编程相关的文档
  优化：→ "Python 编程语言中 list（列表）和 tuple（元组）数据结构的区别"

问题 4：信息不足
  用户查询："怎么训练？"
  问题：缺少上下文，不知道训练什么
  优化：→ "如何用 PyTorch 训练深度学习模型？包括数据准备、模型定义、训练循环等步骤"
```

### 1.2 查询改写的价值

| 问题 | 改写前 | 改写后 | 效果 |
|------|--------|--------|------|
| 口语化 | "RAG 是啥" | "什么是 RAG" | 检索更精准 |
| 缩写 | "GRPO" | "Group Relative Policy Optimization" | 覆盖更广 |
| 歧义 | "Python list" | "Python 编程 list 数据结构" | 更精确 |
| 信息不足 | "怎么训练" | "如何用 PyTorch 训练模型" | 更完整 |

---

## 二、查询改写方法

### 2.1 基于 LLM 的改写

```python
import openai

def rewrite_query_with_llm(original_query):
    """用 LLM 改写查询"""
    prompt = f"""请将以下用户查询改写为更适合搜索的形式：
    
原始查询：{original_query}

改写要求：
1. 补充完整信息（如缩写展开）
2. 去除口语化表达
3. 明确查询意图
4. 保持核心语义不变

改写后的查询："""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

**示例**：

```
原始查询："RAG 是啥？"
改写后："什么是检索增强生成（Retrieval-Augmented Generation）？它的核心原理和应用场景是什么？"

原始查询："GRPO 怎么用？"
改写后："Group Relative Policy Optimization（GRPO）算法的使用方法和实现步骤"
```

### 2.2 Multi-Query 改写

```python
def multi_query_rewrite(original_query, num_queries=3):
    """生成多个改写查询"""
    prompt = f"""请将以下用户查询改写为 {num_queries} 个不同版本，每个版本从不同角度表达相同意图：

原始查询：{original_query}

请生成 {num_queries} 个改写查询，用换行分隔："""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    queries = response.choices[0].message.content.strip().split('\n')
    return queries
```

**示例**：

```
原始查询："如何优化 Transformer 的训练速度？"

改写查询 1："Transformer 模型训练加速方法有哪些？"
改写查询 2："提高 Transformer 训练效率的技术"
改写查询 3："Transformer 训练速度优化策略"
```

**Multi-Query 的好处**：

```
原始查询可能遗漏的文档，改写后能覆盖：

原始查询：["如何优化 Transformer 的训练速度？"]
  → 检索结果：[doc1, doc2, doc3]

改写查询 1：["Transformer 模型训练加速方法有哪些？"]
  → 检索结果：[doc1, doc4, doc5]

改写查询 2：["提高 Transformer 训练效率的技术"]
  → 检索结果：[doc2, doc6, doc7]

合并去重后：[doc1, doc2, doc3, doc4, doc5, doc6, doc7] ← 覆盖更广
```

### 2.3 HyDE（Hypothetical Document Embeddings）

```python
def hyde_rewrite(original_query, llm):
    """HyDE：先生成假设文档，再用假设文档检索"""
    # 1. 用 LLM 生成一个假设的回答文档
    prompt = f"""请写一段关于以下问题的详细回答（假设这是一篇文档的内容）：

问题：{original_query}

回答："""
    
    hypothetical_doc = llm.generate(prompt)
    
    # 2. 用假设文档进行向量检索（而非原始查询）
    return hypothetical_doc
```

**HyDE 原理**：

```
原始查询："什么是注意力机制？"
    ↓ LLM 生成假设文档
假设文档："注意力机制是 Transformer 的核心组件，它允许模型在处理序列时关注输入的不同部分..."
    ↓ 用假设文档检索
找到相似文档：[注意力机制详解, Self-Attention 原理, ...]

为什么有效？
- 假设文档和真实文档在向量空间中距离更近
- 比原始查询的短文本更能代表文档的语义
```

### 2.4 Step-Back Prompting

```python
def step_back_rewrite(original_query, llm):
    """Step-Back：先问一个更宽泛的问题"""
    prompt = f"""原始查询：{original_query}

请生成一个更宽泛、更高层次的问题，帮助理解原始查询的背景：

更宽泛的问题："""
    
    broad_query = llm.generate(prompt)
    return broad_query
```

**示例**：

```
原始查询："RoPE 位置编码的旋转矩阵怎么推导？"
Step-Back 查询："什么是位置编码？位置编码有哪些类型？"

用途：
- 先用 Step-Back 查询检索背景知识
- 再用原始查询检索具体细节
- 综合两部分信息回答
```

---

## 三、查询改写实现

### 3.1 使用 LangChain

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI

# 1. 创建基础检索器
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vectorstore = Chroma.from_documents(documents, embeddings)

# 2. 创建 Multi-Query 检索器
llm = ChatOpenAI(model="gpt-4")
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 3. 检索
results = multi_query_retriever.get_relevant_documents("如何优化 Transformer 的训练速度？")
```

### 3.2 自定义查询改写

```python
class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm
    
    def rewrite(self, query, strategy="expand"):
        """查询改写"""
        if strategy == "expand":
            return self.expand_query(query)
        elif strategy == "decompose":
            return self.decompose_query(query)
        elif strategy == "step_back":
            return self.step_back_query(query)
    
    def expand_query(self, query):
        """扩展查询：补充完整信息"""
        prompt = f"""请将以下查询扩展为更完整的搜索查询：

原始查询：{query}

扩展要求：
1. 展开缩写和简称
2. 补充相关背景信息
3. 明确查询意图

扩展后的查询："""
        
        return self.llm.generate(prompt)
    
    def decompose_query(self, query):
        """分解查询：拆分为多个子查询"""
        prompt = f"""请将以下复杂查询分解为多个简单子查询：

原始查询：{query}

子查询："""
        
        response = self.llm.generate(prompt)
        return response.split('\n')
    
    def step_back_query(self, query):
        """Step-Back：生成更宽泛的问题"""
        prompt = f"""请为以下查询生成一个更宽泛的背景问题：

原始查询：{query}

背景问题："""
        
        return self.llm.generate(prompt)
```

### 3.3 使用 LlamaIndex

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 创建 Query Fusion 检索器（自动查询改写）
retriever = QueryFusionRetriever(
    retrievers=[index.as_retriever()],
    num_queries=4,  # 生成 4 个改写查询
    use_async=True
)

# 检索
results = retriever.retrieve("如何优化 Transformer 的训练速度？")
```

---

## 四、查询改写最佳实践

### 4.1 策略选择

```
简单查询：
  → 扩展查询（补充完整信息）
  → 如："RAG 是啥" → "什么是检索增强生成"

复杂查询：
  → 分解查询（拆分为子查询）
  → 如："如何训练一个 RAG 系统" → ["RAG 数据准备", "RAG 模型选择", "RAG 评估方法"]

模糊查询：
  → Step-Back（生成背景问题）
  → 如："怎么优化" → "优化的方法有哪些"

需要高召回率：
  → Multi-Query（生成多个改写）
  → 如：生成 3-5 个不同版本的查询
```

### 4.2 改写质量评估

```python
def evaluate_rewriting(original_query, rewritten_query, test_docs):
    """评估改写质量"""
    # 1. 语义一致性：改写前后语义是否一致
    semantic_similarity = compute_similarity(original_query, rewritten_query)
    
    # 2. 检索效果：改写后检索是否更好
    original_results = retrieve(original_query, test_docs)
    rewritten_results = retrieve(rewritten_query, test_docs)
    
    original_relevance = evaluate_relevance(original_results)
    rewritten_relevance = evaluate_relevance(rewritten_results)
    
    # 3. 覆盖率：是否能找到更多相关文档
    coverage = len(set(rewritten_results) - set(original_results))
    
    return {
        "semantic_similarity": semantic_similarity,
        "relevance_improvement": rewritten_relevance - original_relevance,
        "additional_coverage": coverage
    }
```

### 4.3 组合使用

```python
def combined_rewriting(query, llm):
    """组合多种改写策略"""
    # 1. Multi-Query：生成多个改写版本
    multi_queries = multi_query_rewrite(query, num_queries=3)
    
    # 2. HyDE：生成假设文档
    hyde_doc = hyde_rewrite(query, llm)
    
    # 3. Step-Back：生成背景问题
    step_back_query = step_back_rewrite(query, llm)
    
    # 4. 合并所有查询
    all_queries = [query] + multi_queries + [hyde_doc, step_back_query]
    
    return all_queries
```

`★ Insight ─────────────────────────────────────`
**查询改写的本质是弥补用户表达和文档内容之间的鸿沟**。用户用口语化、简短的表达提问，而文档用正式、完整的技术语言描述。查询改写将用户的表达"翻译"成更接近文档的语言，让检索更容易找到匹配。
`─────────────────────────────────────────────────`

---

## 五、面试高频问题

### Q1：什么是查询改写？为什么需要它？

> 查询改写用 LLM 或规则将原始查询优化为更适合检索的形式。需要它的原因是：用户查询往往口语化、信息不足、包含缩写，导致向量检索效果差。改写后查询更完整、更精确，检索效果更好。

### Q2：Multi-Query 的原理是什么？

> Multi-Query 将原始查询改写为多个不同版本，每个版本从不同角度表达相同意图。然后分别用这些改写查询检索，合并去重后返回。好处是覆盖面更广，能找到原始查询可能遗漏的文档。

### Q3：HyDE 的原理是什么？

> HyDE（Hypothetical Document Embeddings）先用 LLM 生成一个假设的回答文档，然后用这个假设文档进行向量检索（而非原始查询）。因为假设文档和真实文档在向量空间中距离更近，所以检索效果更好。

### Q4：查询改写和 RAG 有什么关系？

> 查询改写是 RAG 流程中的重要环节，位于用户提问和检索之间。好的查询改写能显著提升检索质量，进而提升 RAG 的整体效果。通常和混合检索、Reranking 结合使用。

---

## 六、总结

**查询改写的核心思想**：

> 将用户口语化、不精确的查询优化为更适合检索的形式，弥补用户表达和文档内容之间的鸿沟。

**推荐方案**：

```
简单查询：扩展查询（补充完整信息）
复杂查询：分解查询（拆分为子查询）
模糊查询：Step-Back（生成背景问题）
需要高召回率：Multi-Query（生成多个改写）
追求最佳效果：HyDE（假设文档检索）
```

**关键参数**：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| Multi-Query 数量 | 3~5 | 改写查询数量 |
| HyDE 温度 | 0.7~0.9 | 生成假设文档的多样性 |
| Step-Back 层级 | 1~2 | 问题宽泛程度 |

**完整 RAG 流程**：

```
用户提问 → 查询改写（Multi-Query/HyDE）→ 混合检索 → Reranking → LLM 生成
```

---

**相关笔记**：
- [[RAG]] - RAG 总览
- [[混合检索]] - 向量检索 + 关键词检索
- [[重排序Reranking]] - 检索结果精排
- [[Prompt Engineering与CoT]] - 提示工程
