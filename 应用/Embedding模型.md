# Embedding 模型：文本向量化技术

> 向量化是 RAG 的核心基础——把文本转换成计算机能理解的数字向量。Embedding 模型的质量直接决定检索效果：好的 Embedding 让语义相似的文本距离近，不同的文本距离远。

---

## 一、为什么需要 Embedding？

### 1.1 文本的数字化表示

计算机无法直接理解文本，需要将文本转换为数值向量：

```
"什么是深度学习"  →  [0.23, -0.15, 0.78, 0.12, ...]  （1024维向量）
```

### 1.2 语义空间

Embedding 的核心思想：**语义相似的文本，向量距离近**。

```
语义空间示意（2D 投影）：

        ↑ 语义维度2
        |
  "深度学习" •────• "机器学习"
        |
        |         • "量子计算"
        |
────────┼─────────────────→ 语义维度1
        |
  "今天天气" •────• "下雨了吗"
        |

特点：
- "深度学习" 和 "机器学习" 距离近（语义相似）
- "深度学习" 和 "今天天气" 距离远（语义不同）
```

### 1.3 Embedding 的应用

| 应用场景 | 说明 |
|---------|------|
| **语义搜索** | 根据语义而非关键词匹配文档 |
| **RAG 检索** | 为向量数据库提供可检索的文档表示 |
| **文本聚类** | 将相似文本自动分组 |
| **相似度计算** | 判断两段文本是否表达相同含义 |
| **推荐系统** | 基于语义相似度推荐内容 |

---

## 二、Embedding 模型原理

### 2.1 Transformer 编码器架构

现代 Embedding 模型基于 Transformer 编码器（如 BERT）：

```
输入文本："什么是深度学习"
    ↓
Tokenizer 分词：[CLS] 什么 是 深度 学习 [SEP]
    ↓
Transformer 编码器（多层自注意力）
    ↓
输出每个 token 的上下文向量
    ↓
池化（Pooling）：取 [CLS] 或 平均所有 token
    ↓
最终句子向量：[0.23, -0.15, 0.78, ...]
```

### 2.2 两种池化策略

| 策略 | 方式 | 特点 |
|------|------|------|
| **CLS Pooling** | 取 [CLS] token 的输出 | 适合分类任务 |
| **Mean Pooling** | 所有 token 向量取平均 | 适合语义相似度 |

**Mean Pooling 示例**：

```
Token 向量：
  [CLS] → [0.1, 0.2, 0.3]
  什么   → [0.4, 0.5, 0.6]
  是     → [0.7, 0.8, 0.9]
  深度   → [1.0, 1.1, 1.2]
  学习   → [1.3, 1.4, 1.5]

Mean Pooling = (0.1+0.4+0.7+1.0+1.3)/5, (0.2+0.5+0.8+1.1+1.4)/5, ...
             = [0.7, 0.8, 0.9]
```

### 2.3 对比学习训练

Embedding 模型通过对比学习（Contrastive Learning）训练：

```
训练目标：相似文本的向量距离近，不相似的距离远

正样本对（应该相似）：
  "什么是深度学习" ↔ "深度学习的定义"

负样本对（应该不相似）：
  "什么是深度学习" ↔ "今天天气怎么样"

损失函数：InfoNCE / Triplet Loss / Contrastive Loss
```

---

## 三、常用 Embedding 模型对比

### 3.1 开源模型

| 模型 | 发布方 | 维度 | 特点 | 适用场景 |
|------|--------|------|------|---------|
| **BGE-large-zh-v1.5** | BAAI | 1024 | 中文效果最好，MTEB 中文榜 Top | 中文为主的应用 |
| **BGE-M3** | BAAI | 1024 | 多语言，支持 100+ 语言 | 多语言应用 |
| **GTE-large-zh** | 阿里 | 1024 | 中文效果好，训练数据多 | 中文通用场景 |
| **GTE-Qwen2** | 阿里 | 768/1536 | 基于 Qwen2，长文本支持好 | 长文本场景 |
| **M3E-large** | Moka | 1024 | 中文专精，轻量级 | 资源受限场景 |
| **nomic-embed-text-v1.5** | Nomic | 768 | 轻量级，768维 | 英文为主 |

### 3.2 闭源模型（API）

| 模型 | 发布方 | 维度 | 特点 | 价格 |
|------|--------|------|------|------|
| **text-embedding-3-large** | OpenAI | 3072 | 效果最好，支持维度缩减 | $0.13/1M tokens |
| **text-embedding-3-small** | OpenAI | 1536 | 性价比高 | $0.02/1M tokens |
| **embed-english-v3.0** | Cohere | 1024 | 英文效果好 | 按量计费 |
| **Cohere Embed v3** | Cohere | 1024 | 多语言，支持压缩 | 按量计费 |

### 3.3 中文模型性能排名（MTEB 中文榜）

```
1. BGE-large-zh-v1.5      ★★★★★  最推荐
2. GTE-large-zh            ★★★★☆  阿里出品
3. M3E-large               ★★★★   轻量级首选
4. text-embedding-3-large  ★★★★   闭源最强
```

`★ Insight ─────────────────────────────────────`
**选模型的黄金法则**：中文场景首选 BGE 系列（BAAI 出品），多语言场景选 BGE-M3，资源受限选 M3E，预算充足选 OpenAI text-embedding-3。不要只看排行榜，实际测试自己的数据集才是关键。
`─────────────────────────────────────────────────`

---

## 四、Embedding 模型使用技巧

### 4.1 文本预处理

```python
# 不同模型的输入格式不同，需要遵循规范

# BGE 模型：需要添加指令前缀
query_text = "query: " + user_query
doc_text = "passage: " + document

# OpenAI 模型：直接输入
embedding = openai.embeddings.create(
    model="text-embedding-3-large",
    input=text
)
```

### 4.2 维度选择

```python
# OpenAI 支持维度缩减（MRL 技术）
embedding = openai.embeddings.create(
    model="text-embedding-3-large",
    dimensions=1024,  # 从 3072 缩减到 1024
    input=text
)

# 权衡：
# - 3072 维：效果最好，存储/计算成本高
# - 1024 维：效果稍差，成本降低 2/3
# - 256 维：效果明显下降，成本极低
```

### 4.3 批量处理

```python
# 批量编码提高效率
texts = ["文本1", "文本2", "文本3", ...]

# BGE 模型批量处理
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
embeddings = model.encode(
    texts,
    batch_size=32,
    max_length=512
)
```

### 4.4 Query vs Document 编码

```python
# BGE 模型区分 query 和 document
query_embedding = model.encode("query: " + query)
doc_embedding = model.encode("passage: " + doc)

# 为什么区分？
# - Query 通常较短，需要更关注关键词
# - Document 通常较长，需要更关注整体语义
```

---

## 五、Embedding 模型评估

### 5.1 评估基准

| 基准 | 说明 | 关注点 |
|------|------|--------|
| **MTEB** | 多任务 Embedding 基准 | 综合性能 |
| **C-MTEB** | 中文 MTEB | 中文场景 |
| **STS-B** | 语义文本相似度 | 相似度计算 |
| **NLI** | 自然语言推理 | 逻辑推理 |

### 5.2 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **nDCG@K** | 归一化折扣累积增益 | 排序质量 |
| **MRR** | 平均倒数排名 | 第一个相关结果的位置 |
| **Recall@K** | 前 K 个结果的召回率 | 覆盖率 |
| **MAP** | 平均精度均值 | 整体排序质量 |

### 5.3 自定义评估

```python
# 评估 Embedding 模型在自己数据集上的表现
import numpy as np
from sklearn.metrics import ndcg_score

def evaluate_embedding(model, test_data):
    """
    test_data: [{"query": "...", "relevant_docs": [...], "all_docs": [...]}]
    """
    queries = [item["query"] for item in test_data]
    query_embeddings = model.encode(queries)
    
    # 计算检索结果
    results = []
    for i, item in enumerate(test_data):
        doc_embeddings = model.encode(item["all_docs"])
        scores = np.dot(doc_embeddings, query_embeddings[i])
        ranked_indices = np.argsort(scores)[::-1]
        
        # 计算 nDCG
        relevance = [1 if idx in item["relevant_docs"] else 0 
                     for idx in ranked_indices]
        results.append(ndcg_score([relevance], [scores]))
    
    return np.mean(results)
```

---

## 六、面试高频问题

### Q1：Embedding 模型的作用是什么？

> Embedding 模型将文本转换为高维向量，使得语义相似的文本向量距离近。这样计算机就能通过计算向量距离来判断文本的语义相似度，是语义搜索和 RAG 的基础。

### Q2：如何选择 Embedding 模型？

> 选择依据：
> 1. **语言**：中文首选 BGE 系列，多语言选 BGE-M3
> 2. **资源**：资源受限选 M3E，预算充足选 OpenAI
> 3. **场景**：通用场景选 BGE-large，长文本选 GTE-Qwen2
> 4. **评估**：在自己的数据集上测试，不要只看排行榜

### Q3：Query 和 Document 的编码有什么区别？

> BGE 等模型区分 query 和 document 编码：
> - Query 编码时加 "query: " 前缀，模型会更关注关键词匹配
> - Document 编码时加 "passage: " 前缀，模型会更关注整体语义
> 
> 这种区分能提升检索效果，因为查询和文档的语义特征不同。

### Q4：维度越高效果越好吗？

> 不一定。维度越高，表达能力越强，但计算和存储成本也越高。实际应用中需要权衡：
> - 3072 维：效果最好，成本最高
> - 1024 维：效果接近，成本降低 2/3
> - 256 维：效果明显下降，成本极低
> 
> 建议先用 1024 维，效果不够再增加。

---

## 七、总结

**Embedding 模型的核心思想**：

> 把文本变成向量，让语义相似的文本距离近，不同的距离远。

**选择建议**：

```
中文场景：BGE-large-zh-v1.5（首选）
多语言：BGE-M3
资源受限：M3E-large
预算充足：OpenAI text-embedding-3-large
```

**关键参数**：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| 维度 | 768~1024 | 平衡效果和成本 |
| max_length | 512 | 大多数场景够用 |
| batch_size | 32~64 | 平衡速度和内存 |

---

**相关笔记**：
- [[RAG]] - RAG 总览
- [[向量数据库]] - 向量存储和检索
- [[混合检索]] - 向量检索 + 关键词检索
