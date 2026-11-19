# 重排序 Reranking：检索结果精排

> 向量检索速度快但精度有限，返回的 Top-K 可能不是最相关的。Reranking 用更精确的模型对 Top-K 结果重新排序，是提升 RAG 检索质量的关键环节。

---

## 一、为什么需要 Reranking？

### 1.1 两阶段检索的必要性

```
第一阶段：向量检索（快速召回）
  - 速度快：毫秒级
  - 覆盖广：从百万文档中快速找出候选
  - 精度有限：基于向量相似度，可能有噪声

第二阶段：Reranking（精确排序）
  - 速度慢：需要逐对计算
  - 精度高：用更复杂的模型评估相关性
  - 只处理候选：Top-10 → Top-3
```

### 1.2 向量检索的局限

```
用户查询："如何用 Python 实现快速排序"

向量检索 Top-5：
  1. Python 排序算法详解（向量相似度 0.91）
  2. 快速排序原理（向量相似度 0.89）
  3. Python 数据结构与算法（向量相似度 0.87）
  4. 快速排序 Python 实现（向量相似度 0.85）
  5. 算法复杂度分析（向量相似度 0.83）

问题：
  - 第 4 个结果（精确匹配）排在第 4 位
  - 第 1 个结果可能只是介绍 Python 排序，没有快速排序的代码
```

### 1.3 Reranking 的作用

```
Reranking Top-5：
  1. 快速排序 Python 实现（相关性 0.95）✓ 精确匹配
  2. Python 排序算法详解（相关性 0.82）✓ 包含快速排序
  3. 快速排序原理（相关性 0.78）✓ 原理介绍
  4. Python 数据结构与算法（相关性 0.45）✗ 相关性低
  5. 算法复杂度分析（相关性 0.32）✗ 相关性低

Reranking 重新评估后，精确匹配的文档排到了第 1 位
```

---

## 二、Reranking 原理

### 2.1 Cross-Encoder 架构

Reranking 通常使用 Cross-Encoder 模型：

```
向量检索（Bi-Encoder）：
  Query → Embedding → [0.23, -0.15, ...]
  Doc → Embedding → [0.25, -0.12, ...]
  相似度 = cosine(query_emb, doc_emb)

Reranking（Cross-Encoder）：
  [CLS] query tokens [SEP] doc tokens [SEP]
      ↓
  Transformer 编码器
      ↓
  [CLS] token 的输出
      ↓
  线性层 → 相关性分数
```

### 2.2 Bi-Encoder vs Cross-Encoder

| 方面 | Bi-Encoder | Cross-Encoder |
|------|------------|---------------|
| **输入** | Query 和 Doc 分开编码 | Query 和 Doc 拼接编码 |
| **输出** | 两个独立的向量 | 一个相关性分数 |
| **速度** | 快（可以预计算 Doc 向量） | 慢（需要逐对计算） |
| **精度** | 有限（无法捕捉细粒度交互） | 高（可以捕捉 Query-Doc 交互） |
| **用途** | 第一阶段召回 | 第二阶段精排 |

```
Bi-Encoder 计算流程：
  Query 向量（预先计算） ← 1次
  Doc 1 向量（预先计算） ← 0次（已缓存）
  Doc 2 向量（预先计算） ← 0次（已缓存）
  相似度计算 ← 向量点积，极快

Cross-Encoder 计算流程：
  [Query, Doc 1] → Transformer → 分数 1
  [Query, Doc 2] → Transformer → 分数 2
  [Query, Doc 3] → Transformer → 分数 3
  ... 每个 Doc 都需要重新计算，很慢
```

### 2.3 Reranking 流程

```
向量检索返回 Top-10：
  [doc1, doc2, doc3, ..., doc10]

Reranking 评估：
  for doc in [doc1, doc2, ..., doc10]:
      score = cross_encoder.predict(query, doc)
      doc.rerank_score = score

按 rerank_score 排序：
  [doc3, doc1, doc7, ..., doc5]  ← Top-10 重新排序

返回 Top-3：
  [doc3, doc1, doc7]
```

---

## 三、常用 Reranking 模型

### 3.1 开源模型

| 模型 | 发布方 | 特点 | 适用场景 |
|------|--------|------|---------|
| **bge-reranker-v2-m3** | BAAI | 多语言，效果好 | 中文为主 |
| **bge-reranker-v2-gemma** | BAAI | 基于 Gemma，效果最好 | 追求最佳效果 |
| **bge-reranker-large** | BAAI | 较旧版本，轻量 | 资源受限 |
| **ms-marco-MiniLM-L-6-v2** | Microsoft | 英文专精 | 英文场景 |
| **cross-encoder/ms-marco-MiniLM-L-12-v2** | Microsoft | 英文，稍大 | 英文高精度 |

### 3.2 闭源模型（API）

| 模型 | 发布方 | 特点 | 价格 |
|------|--------|------|------|
| **Cohere rerank** | Cohere | 效果好，API 简单 | 按量计费 |
| **Jina reranker** | Jina | 多语言，速度快 | 按量计费 |
| **Voyage reranker** | Voyage | OpenAI 生态 | 按量计费 |

### 3.3 模型选择建议

```
中文场景：
  → bge-reranker-v2-m3（首选）
  → bge-reranker-v2-gemma（追求最佳效果）

英文场景：
  → ms-marco-MiniLM-L-6-v2（轻量）
  → cross-encoder/ms-marco-MiniLM-L-12-v2（高精度）

多语言：
  → bge-reranker-v2-m3
  → Cohere rerank（API）

资源受限：
  → bge-reranker-large（较小）
  → ms-marco-MiniLM-L-6-v2（轻量）
```

---

## 四、代码实现

### 4.1 使用 sentence-transformers

```python
from sentence_transformers import CrossEncoder

# 加载 Reranking 模型
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)

# 查询和文档
query = "如何用 Python 实现快速排序"
documents = [
    "Python 排序算法详解",
    "快速排序原理介绍",
    "快速排序 Python 实现代码",
    "Python 数据结构与算法",
    "算法复杂度分析"
]

# 计算相关性分数
pairs = [(query, doc) for doc in documents]
scores = reranker.predict(pairs)

# 按分数排序
ranked_results = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)

print("Reranking 结果：")
for doc, score in ranked_results:
    print(f"  {score:.4f}: {doc}")
```

### 4.2 使用 LangChain

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 1. 创建基础检索器
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vectorstore = Chroma.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 2. 创建 Reranker
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

# 3. 创建带 Reranking 的检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 4. 检索
results = compression_retriever.get_relevant_documents("如何用 Python 实现快速排序")
```

### 4.3 使用 Cohere API

```python
import cohere

# 创建客户端
co = cohere.Client("YOUR_API_KEY")

# Reranking
query = "如何用 Python 实现快速排序"
documents = [
    "Python 排序算法详解",
    "快速排序原理介绍",
    "快速排序 Python 实现代码",
    ...
]

results = co.rerank(
    query=query,
    documents=documents,
    top_n=3,
    model="rerank-multilingual-v3.0"
)

# 结果
for result in results.results:
    print(f"{result.relevance_score:.4f}: {documents[result.index]}")
```

### 4.4 使用 Qdrant 内置 Reranking

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Query, Prefetch

client = QdrantClient(host="localhost", port=6333)

# 使用 Qdrant 内置的 Reranking
results = client.query_points(
    collection_name="documents",
    query=Query(
        prefetch=Prefetch(
            vector=query_embedding,
            limit=10  # 先检索 Top-10
        ),
        # 使用 Reranking 模型
        rerank={
            "model": "BAAI/bge-reranker-v2-m3",
            "query": query
        }
    ),
    limit=3  # 最终返回 Top-3
)
```

---

## 五、Reranking 最佳实践

### 5.1 候选数量选择

```
向量检索返回数量：
  - 太少（<5）：可能遗漏相关文档
  - 太多（>20）：计算开销大，收益递减
  - 推荐：10~20

Reranking 返回数量：
  - 通常 3~5 个
  - 根据 LLM 上下文窗口调整
```

### 5.2 批量处理

```python
# 批量 Reranking 提高效率
def batch_rerank(reranker, query, documents, batch_size=32):
    """批量 Reranking"""
    all_scores = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        pairs = [(query, doc) for doc in batch]
        scores = reranker.predict(pairs)
        all_scores.extend(scores)
    
    return all_scores
```

### 5.3 缓存优化

```python
# 缓存 Reranking 结果
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_rerank(query, doc_id):
    """缓存 Reranking 结果"""
    doc = get_document(doc_id)
    score = reranker.predict([(query, doc)])[0]
    return score
```

### 5.4 结合上下文压缩

```python
# Reranking + 上下文压缩
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter

# 1. 先用 EmbeddingsFilter 过滤
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.5
)

# 2. 再用 CrossEncoderReranker 精排
compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)

# 3. 组合成管道
pipeline = DocumentCompressorPipeline(
    transformers=[embeddings_filter, compressor]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=base_retriever
)
```

`★ Insight ─────────────────────────────────────`
**Reranking 的本质是用精度换速度**：Cross-Encoder 可以捕捉 Query 和 Doc 之间的细粒度交互，但需要逐对计算，速度慢。所以先用向量检索快速召回候选，再用 Reranking 精排。这是经典的"召回-精排"两阶段范式，在搜索引擎中广泛使用。
`─────────────────────────────────────────────────`

---

## 六、面试高频问题

### Q1：什么是 Reranking？为什么需要它？

> Reranking 用更精确的模型（通常是 Cross-Encoder）对向量检索返回的 Top-K 结果重新排序。需要它的原因是：向量检索速度快但精度有限，Reranking 可以捕捉 Query 和 Doc 之间的细粒度交互，提升排序质量。

### Q2：Bi-Encoder 和 Cross-Encoder 有什么区别？

> - **Bi-Encoder**：Query 和 Doc 分别编码成向量，用向量相似度计算相关性。速度快，可以预计算 Doc 向量，但精度有限
> - **Cross-Encoder**：Query 和 Doc 拼接后一起输入 Transformer，输出相关性分数。精度高，可以捕捉细粒度交互，但需要逐对计算，速度慢
> 
> 所以用 Bi-Encoder 做召回，Cross-Encoder 做精排。

### Q3：Reranking 的计算开销大吗？

> 相对较大。假设向量检索返回 Top-10，Reranking 需要计算 10 次 Cross-Encoder 推理。但因为只处理少量候选（通常 10~20 个），总体延迟可控（通常 <100ms）。

### Q4：如何选择 Reranking 模型？

> 选择依据：
> 1. **语言**：中文首选 BGE-reranker 系列，英文选 ms-marco-MiniLM
> 2. **资源**：资源受限选 bge-reranker-large，预算充足选 bge-reranker-v2-gemma
> 3. **场景**：通用场景选 bge-reranker-v2-m3，多语言选 Cohere rerank
> 4. **评估**：在自己的数据集上测试 nDCG@10

---

## 七、总结

**Reranking 的核心思想**：

> 用更精确的模型（Cross-Encoder）对向量检索返回的候选结果重新排序，提升检索质量。

**推荐方案**：

```
默认选择：bge-reranker-v2-m3
追求最佳：bge-reranker-v2-gemma
英文场景：ms-marco-MiniLM-L-12-v2
API 调用：Cohere rerank
```

**关键参数**：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| 向量检索 Top-K | 10~20 | 候选数量 |
| Reranking Top-N | 3~5 | 最终返回数量 |
| max_length | 512 | 最大输入长度 |

**完整 RAG 流程**：

```
文档 → 分块 → Embedding → 向量库
                           ↑ 存储
用户提问 → Embedding → 向量检索 Top-10 → Reranking Top-3 → LLM 生成
```

---

**相关笔记**：
- [[RAG]] - RAG 总览
- [[Embedding模型]] - 文本向量化
- [[混合检索]] - 向量检索 + 关键词检索
- [[文档分块策略]] - 分块技术
