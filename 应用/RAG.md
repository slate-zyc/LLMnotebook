# RAG：检索增强生成

> 大模型的知识来自训练数据，训练数据截止后的新知识、企业私有数据、实时信息，模型统统不知道。RAG（Retrieval-Augmented Generation）通过在生成前先检索相关文档，让模型"开卷考试"而非"闭卷硬记"，是当前解决大模型知识局限性的主流方案。
>
> 本文是 RAG 的总览，详细介绍各个技术组件的笔记请参见：
> - [[Embedding模型]] - 文本向量化技术
> - [[向量数据库]] - 向量存储与检索
> - [[文档分块策略]] - Chunking 技术
> - [[混合检索]] - 向量检索 + 关键词检索
> - [[重排序Reranking]] - 检索结果精排
> - [[查询改写QueryRewriting]] - 优化检索查询

---

## 一、为什么需要 RAG？

### 1.1 大模型的知识困境

大模型有三个根本性的知识问题：

| 问题 | 说明 | 例子 |
|------|------|------|
| **知识过时** | 训练数据有截止日期 | "2026 年诺贝尔物理学奖得主是谁？" |
| **缺乏私有知识** | 无法访问企业/个人数据 | "我们公司的请假流程是什么？" |
| **幻觉** | 知识不足时会编造看似合理的答案 | 编造不存在的论文、法律条文 |

### 1.2 三种解决思路对比

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **重新训练/微调** | 把新知识注入模型参数 | 效果好 | 成本高，更新慢 |
| **长上下文** | 把所有资料塞进 prompt | 简单 | 有上下文长度限制，成本高 |
| **RAG** | 检索相关文档 + 拼到 prompt 里 | **灵活、低成本、实时更新** | 依赖检索质量 |

RAG 的核心优势：**知识更新不需要重新训练模型**——换一批文档就行。

### 1.3 RAG 的一句话定义

> RAG = **检索（Retrieve）**相关文档 + **增强（Augment）**到 prompt 中 + **生成（Generate）**回答

```
用户提问："我们公司的年假有多少天？"
    ↓
检索：在公司知识库中找到《员工手册》相关段落
    ↓
拼接：[检索到的文档] + 用户提问 → 送给大模型
    ↓
生成：根据文档内容回答"入职满一年享有 5 天年假"
```

---

## 二、RAG 的完整流程

### 2.1 总览

RAG 分为**离线阶段**（建索引）和**在线阶段**（检索+生成）：

```
┌──────────── 离线阶段（一次性准备）────────────┐
│                                             │
│  文档 → 分块 → Embedding → 存入向量数据库    │
│                                             │
└─────────────────────────────────────────────┘

┌──────────── 在线阶段（每次问答）─────────────┐
│                                             │
│  用户问题 → Embedding → 向量检索 → 拼接 → LLM │
│                                             │
└─────────────────────────────────────────────┘
```

### 2.2 离线阶段：建索引

```
原始文档（PDF、Word、网页...）
    ↓
① 文档加载（解析为纯文本）
    ↓
② 文本分块（Chunking）→ 切成一个个小段落
    ↓
③ 向量化（Embedding）→ 每个块变成一个向量
    ↓
④ 存入向量数据库 → 按向量索引存储
```

### 2.3 在线阶段：检索+生成

```
用户提问
    ↓
① 查询向量化 → 把问题也变成向量
    ↓
② 向量相似度检索 → 在数据库中找最相似的 K 个块
    ↓
③ 拼接 prompt → [检索到的文档块] + 用户问题
    ↓
④ LLM 生成回答 → 基于文档内容作答
```

---

## 三、核心组件详解

### 3.1 Embedding 模型

Embedding 模型的作用：**把文本转换为高维向量，使得语义相似的文本向量距离近。**

```
"什么是深度学习"  →  Embedding  →  [0.23, -0.15, 0.78, ...]（如 1024 维）
"深度学习的定义"  →  Embedding  →  [0.25, -0.12, 0.81, ...]（相近）
"今天天气怎么样"  →  Embedding  →  [-0.67, 0.43, 0.12, ...]（很远）
```

**常用 Embedding 模型**：

| 模型 | 维度 | 特点 |
|------|------|------|
| OpenAI text-embedding-3-large | 3072 | 效果好，付费 API |
| BGE（BAAI） | 768/1024 | 开源，中文效果好 |
| GTE（阿里） | 768 | 开源，多语言 |
| M3E | 768 | 开源，中文专精 |

> 详见 [[Embedding模型]]

### 3.2 相似度检索

检索的核心是计算**查询向量**与**文档向量**之间的相似度。

**常用相似度指标**：

| 指标 | 公式 | 适用场景 |
|------|------|---------|
| 余弦相似度 | $\cos(q, d) = \frac{q \cdot d}{\|q\| \|d\|}$ | 最常用，不受向量长度影响 |
| 内积 | $q \cdot d$ | 向量已归一化时等价于余弦 |
| 欧氏距离 | $\|q - d\|$ | 较少使用 |

**Top-K 检索**：返回相似度最高的 K 个文档块，K 通常取 3~10。

### 3.3 向量数据库

专门存储和检索向量的数据库，支持高效的近似最近邻搜索（ANN）。

| 数据库 | 特点 |
|--------|------|
| **FAISS**（Meta） | 内存级，适合中小规模，嵌入代码使用 |
| **Milvus**（Zilliz） | 分布式，支持大规模，功能全面 |
| **Chroma** | 轻量级，Python 友好，适合原型开发 |
| **Qdrant** | Rust 实现，高性能，支持过滤 |
| **Pinecone** | 全托管云服务，开箱即用 |

> 详见 [[向量数据库]]

---

## 四、分块策略（Chunking）

分块是 RAG 中**最容易被忽视但对效果影响最大**的环节。

### 4.1 为什么分块重要？

- 块太小：信息不完整，检索到的块可能缺上下文
- 块太大：包含太多无关信息，干扰 LLM 判断
- 分块位置不当：可能把完整的段落/表格切碎

### 4.2 常见分块策略

| 策略 | 方式 | 优点 | 缺点 |
|------|------|------|------|
| **固定长度分块** | 按字符/token 数切分（如 512 token） | 简单可控 | 可能截断句子/段落 |
| **按分隔符分块** | 按段落、换行符、标题切分 | 保持语义完整性 | 块大小不均匀 |
| **递归分块** | 先按大分隔符切，太大再按小分隔符切 | 兼顾大小和语义 | 实现稍复杂 |
| **语义分块** | 用 Embedding 检测语义边界 | 最智能 | 计算开销大 |
| **滑动窗口分块** | 固定窗口 + 重叠区域 | 避免边界信息丢失 | 有冗余 |

**常用方案**：递归分块 + 滑动窗口重叠

```
原文：  [段落1完整内容...][段落2完整内容...][段落3完整内容...]

分块（chunk_size=512, overlap=50）：

Chunk 1: [段落1 的前 512 个 token .............]
Chunk 2: [段落1 末尾 50 token + 段落2 前 462 token ...]  ← 50 token 重叠
Chunk 3: [段落2 末尾 50 token + 段落3 内容 .........]
```

### 4.3 父子文档检索

一种高级策略：**检索时用小块（精确匹配），返回时附带父块（完整上下文）**。

```
检索阶段：用 256 token 的小块去匹配（精确）
    ↓ 命中了某个小块
返回阶段：返回包含这个小块的 1024 token 大块（完整上下文）
    ↓
LLM 用大块生成（信息更完整）
```

> 详见 [[文档分块策略]]

---

## 五、Prompt 拼接模板

检索到文档后，需要合理拼接给 LLM。常见模板：

```python
prompt = f"""
你是一个专业的助手。请根据以下参考文档回答用户的问题。
如果参考文档中没有相关信息，请如实说明"根据已有资料无法回答"。

参考文档：
---
{retrieved_doc_1}
---
{retrieved_doc_2}
---
{retrieved_doc_3}
---

用户问题：{user_question}

请回答：
"""
```

关键设计：
- **限定回答范围**：要求基于文档回答，减少幻觉
- **承认不确定性**：允许说"不知道"，而非编造
- **分隔清晰**：用分隔符让模型区分文档和问题

---

## 六、高级 RAG 优化

### 6.1 查询改写（Query Rewriting）

用户的原始查询可能不够精确，先用 LLM 改写：

```
原始查询："RAG 是啥？"
    ↓ LLM 改写
改写查询："什么是检索增强生成（Retrieval-Augmented Generation）？它的核心原理和应用场景是什么？"
```

改写后的查询信息更丰富，检索效果更好。

> 详见 [[查询改写QueryRewriting]]

### 6.2 混合检索（Hybrid Search）

结合**向量检索**（语义匹配）和**关键词检索**（精确匹配）：

```
用户查询："Python 的 list 和 tuple 的区别"
    ↓
向量检索：找到语义相关的文档（可能提到"序列类型"、"可变性"）
关键词检索：精确匹配 "list" "tuple" 这两个词
    ↓
合并排序：取两路结果的并集，按综合得分排序
```

`★ Insight ─────────────────────────────────────`
单独用向量检索有个经典陷阱：用户问"GRPO"，向量检索可能返回包含"RLHF"的文档（语义相近但不精确）。加入关键词检索后，能确保精确匹配 "GRPO" 这个术语的文档被召回。这就是混合检索的价值——语义 + 精确互为补充。
`─────────────────────────────────────────────────`

> 详见 [[混合检索]]

### 6.3 重排序（Reranking）

检索返回 Top-K 后，用一个更精确的模型对结果重新排序：

```
向量检索 Top-10（快速但粗略）
    ↓
Reranker 模型逐对精排（慢但准确）
    ↓
返回 Top-3（最相关的文档）
```

Reranker 的输入是 `(query, document)` 对，输出一个相关性分数。常用模型：BGE-reranker、Cohere rerank。

> 详见 [[重排序Reranking]]

### 6.4 上下文压缩（Contextual Compression）

检索到的文档块可能包含大量无关信息，压缩后再送给 LLM：

```
原始块（1000 token）：包含关于年假的3个段落 + 其他无关内容
    ↓ LLM 压缩
压缩后（200 token）：只保留与"年假天数"直接相关的内容
```

### 6.5 自适应检索（Adaptive RAG）

不是所有问题都需要检索：

```
"1+1=?"        → 不检索，直接回答（常识问题）
"公司的年假政策" → 检索公司文档（私有知识）
"今天的天气"    → 检索实时数据（时效信息）
```

用 LLM 或分类器先判断是否需要检索，减少不必要的检索开销。

---

## 七、代码实现

### 7.1 基础 RAG 流程（LangChain）

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# 1. 加载文档
loader = PyPDFLoader("company_handbook.pdf")
documents = loader.load()

# 2. 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # 每个块的最大字符数
    chunk_overlap=50,     # 块之间的重叠字符数
    separators=["\n\n", "\n", "。", "，", " "]  # 按优先级尝试的分隔符
)
chunks = splitter.split_documents(documents)

# 3. 向量化
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"  # 中文 BGE 模型
)

# 4. 存入向量数据库
vectorstore = Chroma.from_documents(chunks, embeddings)

# 5. 创建检索器 + LLM 问答链
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Top-3
llm = HuggingFacePipeline.from_model_id(model_id="Qwen/Qwen2.5-7B")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 6. 问答
answer = qa_chain.invoke("公司年假有多少天？")
print(answer)
```

### 7.2 纯 Python 实现（不用框架）

```python
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# 1. 加载 Embedding 模型
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5")

def get_embedding(text):
    """将文本转换为向量"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 取 [CLS] token 的输出作为句子向量
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# 2. 对文档块向量化并存储
docs = [
    "公司员工入职满一年享有5天年假。",
    "年假可以分次使用，每次最少半天。",
    "未休完的年假可以顺延到次年3月底。",
]
doc_embeddings = [get_embedding(doc) for doc in docs]

# 3. 用户查询 → 向量 → 余弦相似度检索
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = "年假有多少天？"
query_emb = get_embedding(query)

# 4. 计算相似度并排序
scores = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

print("检索结果：")
for idx, score in ranked:
    print(f"  相似度 {score:.4f}：{docs[idx]}")
```

---

## 八、面试高频问题

### Q1：RAG 的核心流程是什么？

> RAG 分为离线和在线两个阶段。离线阶段：文档加载 → 分块 → Embedding 向量化 → 存入向量数据库。在线阶段：用户查询 → 查询向量化 → 向量相似度检索 Top-K 文档 → 拼接 prompt（文档 + 问题）→ LLM 生成回答。

### Q2：RAG 和微调的区别？分别适合什么场景？

> - **微调**：修改模型参数，将知识"写入"模型。适合需要深度理解的任务、风格迁移、行为调整。成本高，更新慢
> - **RAG**：不改模型，通过检索外部文档提供知识。适合知识频繁更新、需要引用来源、私有数据问答。成本低，更新快
>
> 两者可以结合使用：微调提升模型的"能力"，RAG 补充模型的"知识"。

### Q3：向量检索的原理是什么？

> 将文本通过 Embedding 模型映射为高维向量，语义相似的文本向量距离近。检索时将查询也转为向量，计算与所有文档向量的余弦相似度，取 Top-K。向量数据库用 ANN（近似最近邻）算法（如 HNSW、IVF）加速检索。

### Q4：分块策略怎么选？

> 常用方案是递归分块（先按段落，太大再按句子）+ 滑动窗口重叠。chunk_size 通常 256~1024 token，overlap 50~100 token。关键原则：保持语义完整性（不要切断句子/表格），块大小适中（太小信息不完整，太大包含噪声）。

### Q5：什么是混合检索？为什么需要它？

> 混合检索结合向量检索（语义匹配）和关键词检索（精确匹配）。单独用向量检索可能遗漏精确术语匹配（如搜索 "GRPO" 可能返回 "RLHF" 的文档），单独用关键词检索无法处理同义词和语义相似。两者结合后取并集重排，覆盖面更广。

### Q6：RAG 中最常见的问题是什么？怎么解决？

> 1. **检索不到相关文档**：优化分块策略、使用混合检索、查询改写
> 2. **检索到了但排名靠后**：用 Reranker 重排序
> 3. **文档相关但包含噪声**：上下文压缩，只保留相关段落
> 4. **LLM 不看文档自己编**：优化 prompt 模板，明确要求基于文档回答
> 5. **多跳问题**（需要综合多个文档）：用 Multi-hop RAG 或 Agent 式迭代检索

### Q7：什么是 Reranking？为什么需要两阶段检索？

> 第一阶段向量检索速度快但精度有限（返回 Top-10），第二阶段 Reranker 用更精确的模型（通常是 cross-encoder）对 Top-10 结果逐对精排，返回 Top-3。好处是兼顾速度和精度：向量检索负责快速召回大量候选，Reranker 负责精确筛选。

---

## 九、总结

**RAG 的核心思想一句话**：

> 不让模型"死记硬背"所有知识，而是让它在回答前先"查资料"，用检索到的文档辅助生成。

**完整的 RAG 流水线**：

```
文档 → 分块 → Embedding → 向量库
                           ↑ 存储
用户提问 → 查询改写 → 混合检索 Top-10 → Reranking Top-3 → 拼接 prompt → LLM 生成
```

**关键组件**：

| 组件 | 作用 | 常用选择 | 详见 |
|------|------|---------|------|
| Embedding 模型 | 文本 → 向量 | BGE、GTE、OpenAI | [[Embedding模型]] |
| 向量数据库 | 存储和检索向量 | FAISS、Chroma、Milvus | [[向量数据库]] |
| 分块策略 | 切分文档 | 递归分块 + 滑动窗口 | [[文档分块策略]] |
| 混合检索 | 向量 + 关键词检索 | BM25 + 向量检索 | [[混合检索]] |
| Reranker | 精排 | BGE-reranker、Cohere | [[重排序Reranking]] |
| 查询改写 | 优化查询 | Multi-Query、HyDE | [[查询改写QueryRewriting]] |
