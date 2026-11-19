# Tokenizer 分词器与 BPE 算法

> 你可能觉得大模型的魔法在 Transformer 架构或训练策略上，但模型处理的最小单位并不是"字"或"词"，而是 **token**。一段文本在送入模型之前，必须先经过 Tokenizer 切分成 token 序列。分词方式直接影响模型的词汇表大小、序列长度、对新词的处理能力，甚至多语言性能。
>
> 本文从"为什么需要 Tokenizer"出发，讲清三种分词粒度的取舍，重点拆解 **BPE（Byte Pair Encoding）** 算法的完整流程，并对比 WordPiece、SentencePiece 等主流方案。最后整理了面试高频问题。

---
## 一、为什么需要 Tokenizer？

### 1.1 模型只能处理数字

神经网络不能直接处理文本，只能处理数值。Tokenizer 的核心任务就是**将原始文本转换为模型可以处理的 token ID 序列**。

```
原始文本："我爱深度学习"
    ↓ Tokenizer
Token IDs：[10892, 5398, 3748, 2983]
    ↓ Embedding
向量序列：[v₁, v₂, v₃, v₄]  ← 进入 Transformer
```

### 1.2 分词粒度的三种选择

| 方案 | 切分单位 | 词汇表大小 | 示例 | 代表 |
|------|---------|-----------|------|------|
| **词级（Word-level）** | 完整单词 | 极大（10万+） | `"playing"` → `["playing"]` | 早期 NLP |
| **字符级（Character-level）** | 单个字符 | 很小（几百） | `"playing"` → `["p","l","a","y","i","n","g"]` | CharRNN |
| **子词级（Subword）** | 介于词和字符之间 | 适中（3万~6万） | `"playing"` → `["play", "ing"]` | **BPE、WordPiece** |

三种粒度各有致命问题：

| 方案      | 优点         | 致命缺陷                  |
| ------- | ---------- | --------------------- |
| 词级      | 语义完整       | 遇到未登录词（OOV）直接失败；词汇表膨胀 |
| 字符级     | 无 OOV 问题   | 序列太长，计算量大；单个字符语义太弱    |
| **子词级** | **兼顾两者优点** | 需要训练分词算法（如 BPE）       |

`★ Insight ─────────────────────────────────────`
现代大模型（GPT、LLaMA、Qwen、DeepSeek）**全部使用子词级分词**。子词的核心思想是：常见词保持完整（如 `"the"`、`"是"`），罕见词拆成子词片段（如 `"ChatGPT"` → `["Chat", "G", "PT"]`）。这样既控制了词汇表大小，又解决了 OOV 问题。
`─────────────────────────────────────────────────`

---

## 二、BPE 算法详解

BPE（Byte Pair Encoding，字节对编码）是最主流的子词分词算法，GPT 系列、LLaMA、Qwen 等模型都使用它。

### 2.1 BPE 的核心思想

> BPE 的思路非常直觉：**反复合并语料中出现频率最高的相邻字符对，直到词汇表达到目标大小。**

它是一个**从下往上**的贪心算法——从单个字符开始，逐步合并出更大的子词单元。

### 2.2 完整算法流程

**输入**：
- 训练语料（大量文本）
- 目标词汇表大小 $V$（如 32000）

**输出**：
- 合并规则表（merge rules）
- 词汇表（vocabulary）

**步骤**：

```
Step 1：初始化词汇表 = 所有单个字符（+ 特殊 token）

Step 2：统计所有相邻字符对的出现频率

Step 3：找到频率最高的字符对 (a, b)

Step 4：将 (a, b) 合并为新符号 "ab"，加入词汇表

Step 5：在语料中执行合并（所有 "a b" 替换为 "ab"）

Step 6：重复 Step 2~5，直到词汇表大小达到 V
```

### 2.3 具体例子

假设训练语料只有以下单词（用 `_` 表示空格）：

```
"low" 出现 5 次，"lower" 出现 2 次，"newest" 出现 6 次，"widest" 出现 3 次
```

**初始状态**（按字符切分，附频率）：

```
词频表：
  l o w </w>          × 5
  l o w e r </w>      × 2
  n e w e s t </w>    × 6
  w i d e s t </w>    × 3

词汇表：{l, o, w, e, r, n, s, t, i, d, </w>}
```

**第 1 轮合并**：

```
统计相邻字符对频率：
  (e, s) = 6 + 3 = 9 次   ← 最高！
  (s, t) = 6 + 3 = 9 次
  (e, w) = 6 次
  (l, o) = 5 + 2 = 7 次
  ...

合并 (e, s) → "es"

更新后：
  l o w </w>          × 5
  l o w e r </w>      × 2
  n e w es t </w>     × 6    ← "es" 合并了
  w i d es t </w>     × 3

词汇表新增：es
```

**第 2 轮合并**：

```
统计相邻字符对频率：
  (es, t) = 6 + 3 = 9 次   ← 最高！
  (l, o) = 5 + 2 = 7 次
  ...

合并 (es, t) → "est"

更新后：
  l o w </w>          × 5
  l o w e r </w>      × 2
  n e w est </w>      × 6
  w i d est </w>      × 3

词汇表新增：est
```

**第 3 轮合并**：

```
合并 (l, o) → "lo"  （频率 7 次最高）

更新后：
  lo w </w>           × 5
  lo w e r </w>       × 2
  ...
```

**持续合并**，直到词汇表达到目标大小。最终学到的合并规则类似：

```
合并规则（按学习顺序）：
  1. (e, s) → es
  2. (es, t) → est
  3. (l, o) → lo
  4. (lo, w) → low
  5. (n, e) → ne
  6. (ne, w) → new
  7. (new, est) → newest
  ...
```

### 2.4 推理时如何分词？

训练完成后，用学到的合并规则对新文本分词：

```
输入："lowest"

Step 1：按字符切分 → [l, o, w, e, s, t, </w>]

Step 2：按合并规则依次应用：
  规则 (e, s) → es    → [l, o, w, es, t, </w>]
  规则 (es, t) → est   → [l, o, w, est, </w>]
  规则 (l, o) → lo     → [lo, w, est, </w>]
  规则 (lo, w) → low    → [low, est, </w>]

最终分词结果：["low", "est", "</w>"]
```

注意 `"lowest"` 被拆成了 `"low"` + `"est"`，而不是保持完整——因为 BPE 是基于频率的统计方法，`"lowest"` 在训练语料中出现次数不够多，没能学到完整的合并。

`★ Insight ─────────────────────────────────────`
BPE 的优雅之处在于：**分词结果完全由训练语料的统计规律决定**。高频词会学到完整的合并（保持完整），低频词只学到部分合并（被拆成子词）。这种自适应机制让 BPE 无需人工定义任何分词规则，就能自动平衡词汇表大小和序列长度。
`─────────────────────────────────────────────────`

---

## 三、Tokenizer 的训练过程

### 3.1 训练流程

```
原始语料（TB 级文本）
    ↓
预处理（统一大小写？去重？添加特殊 token？）
    ↓
统计字符/字节频率，初始化基础词汇表
    ↓
反复执行 BPE 合并，直到词汇表大小 = V
    ↓
输出：词汇表 + 合并规则表
```

### 3.2 使用 HuggingFace Tokenizers 训练

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# 1. 创建 BPE Tokenizer（基于字节级）
tokenizer = Tokenizer(BPE(unk_token=None))  # Byte-BPE 没有 UNK
tokenizer.pre_tokenizer = ByteLevel()        # 以字节为初始单元

# 2. 配置训练器
trainer = BpeTrainer(
    vocab_size=32000,              # 目标词汇表大小
    min_frequency=2,               # 最小频率阈值
    special_tokens=[               # 特殊 token
        "<|pad|>", "<|eos|>",
        "<|bos|>"
    ],
    show_progress=True
)

# 3. 训练（喂入语料文件列表）
tokenizer.train(
    files=["corpus_part1.txt", "corpus_part2.txt"],
    trainer=trainer
)

# 4. 保存
tokenizer.save("my_tokenizer.json")

# 5. 使用
encoded = tokenizer.encode("Hello, 你好！")
print(encoded.ids)       # token ID 列表
print(encoded.tokens)    # token 字符串列表
```

### 3.3 使用 SentencePiece 训练

```python
import sentencepiece as spm

# 一行命令训练
spm.SentencePieceTrainer.train(
    input='corpus.txt',            # 训练语料
    model_prefix='my_tokenizer',   # 输出文件前缀
    vocab_size=32000,              # 词汇表大小
    model_type='bpe',              # 算法类型：bpe / unigram / char / word
    character_coverage=0.9995,     # 字符覆盖率（中文建议 0.9995）
    pad_id=0,                      # pad token ID
    eos_id=1,                      # eos token ID
    bos_id=2,                      # bos token ID
    unk_id=3,                      # unk token ID
)

# 加载并使用
sp = spm.SentencePieceProcessor(model_file='my_tokenizer.model')
print(sp.encode("Hello, 你好！", out_type=str))
# ['▁Hello', ',', '▁', '你', '好', '！']
print(sp.encode("Hello, 你好！", out_type=int))
# [15068, 3, 4, 234, 89, 18]
```

---

## 四、特殊 Token

大模型的 Tokenizer 都会定义特殊 token 来控制模型行为：

| 特殊 Token | 作用 | 示例 |
|---|---|---|
| `<|eos|>` / `<|endoftext|>` | 序列结束标志 | 生成到此 token 时停止 |
| `<|pad|>` | 填充 token | 批量推理时对齐序列长度 |
| `<|bos|>` / `<s>` | 序列开始标志 | 标记输入的起始 |
| `<|unk|>` | 未知 token | 无法分词时的兜底（Byte-BPE 不需要） |
| `[CLS]` / `[SEP]` | BERT 专用 | 句子分类和句子对分隔 |
| `<|im_start|>` / `<|im_end|>` | 对话格式标记 | ChatML 格式的角色分隔 |
这些特殊 token 会被加入词汇表，但**不会参与 BPE 合并训练**。

---

## 五、影响分词效果的关键因素

### 5.1 词汇表大小

| 词汇表大小 | 序列长度 | 模型 Embedding 层 | 代表模型 |
|-----------|---------|-------------------|---------|
| 30K | 较长 | 小 | T5, LLaMA 1 |
| 32K~64K | 适中 | 中等 | GPT-2, LLaMA 2, Qwen |
| 100K+ | 较短 | 大 | GPT-4, Claude |

**权衡**：
- 词汇表**越大** → 每个 token 承载更多信息 → 序列更短 → 计算更快
- 但 Embedding 层参数 = 词汇表大小 × 隐藏维度，词汇表太大会增加模型参数和显存

### 5.2 训练语料的语言分布

```
如果训练语料 90% 是英文，10% 是中文：
  → BPE 会优先学习英文的常见字符组合
  → 中文分词效率低（一个汉字可能需要 2~3 个 token）

如果训练语料包含充足中文：
  → 高频汉字会被合并为单个 token
  → 中文分词效率高
```

这就是为什么**同一个中文句子，不同模型的 token 数量差异很大**：

```
"我爱深度学习"

GPT-2：  [我, 爱, 深, 度, 学, 习] → 6+ tokens（中文效率差）
Qwen：   [23451, 31067, 18293, 8923] → 4 tokens（中文效率好）
```

`★ Insight ─────────────────────────────────────`
Qwen 和 DeepSeek 等国产模型特别注重中文分词效率。Qwen 的 Tokenizer 使用了约 15 万个 merge 操作，其中包含大量中文字符组合，确保常用中文词能被高效编码。这直接影响中文推理的速度和成本——token 越少，计算量越小。
`─────────────────────────────────────────────────`

### 5.3 Tokenizer 与模型的匹配

> Tokenizer 是在预训练之前单独训练的，但它的词汇表必须与模型的 Embedding 层匹配。

```
Tokenizer 词汇表大小 = V
  → Embedding 矩阵形状：(V, d_model)  ← 每个 token 一个向量
  → 最后一层线性层形状：(d_model, V)  ← 预测下一个 token
```

如果更换 Tokenizer，Embedding 层和输出层需要重新初始化或训练。

---

## 六、常见 Tokenizer 对比

| 模型          | Tokenizer 类型           | 词汇表大小    | 特点               |
| ----------- | ---------------------- | -------- | ---------------- |
| GPT-1       | BPE                    | 40,478   | 字符级初始单元          |
| GPT-2/3/4   | Byte-Level BPE         | 50,257   | 字节级，无 OOV        |
| BERT        | WordPiece              | 30,522   | 中文按字切分           |
| LLaMA 2     | SentencePiece (BPE)    | 32,000   | 支持多语言            |
| LLaMA 3     | Tiktoken (BPE)         | 128,256  | **超大词汇表**，多语言效率高 |
| Qwen 2.5    | Tiktoken (BPE)         | 151,643  | 超大词汇表，中文效率极高     |
| DeepSeek V3 | HuggingFace Tokenizers | 128,000+ | 多语言支持好           |

**趋势**：新模型的词汇表越来越大（从 3 万到 15 万），因为更大的词汇表能更高效地编码多语言文本，减少序列长度，加速推理。

---

## 七、面试高频问题

### Q1：什么是 Tokenizer？为什么大模型需要它？

> Tokenizer 是将原始文本转换为 token ID 序列的组件。大模型的神经网络只能处理数值，不能直接处理文本，所以必须先通过 Tokenizer 将文本切分为 token（子词单元），再映射为 ID 序列送入模型。分词方式直接影响词汇表大小、序列长度、对新词的处理能力。

### Q2：BPE 算法的原理是什么？

> BPE 从单个字符开始，反复合并语料中出现频率最高的相邻字符对，直到词汇表达到目标大小。训练时记录合并规则；推理时按字符切分，再按合并规则从左到右依次应用。高频词会学到完整合并（保持为一个 token），低频词只学到部分合并（被拆为几个子词 token），天然平衡了词汇表大小和 OOV 问题。

### Q3：Byte-Level BPE 是什么？为什么 GPT-2 使用它？

> Byte-Level BPE 以字节（256 种可能值）为初始单元，而非 Unicode 字符。优势：初始词汇表只有 256 个，任何文本（含所有语言、emoji、特殊符号）都能用字节表示，**天然不存在 OOV 问题**。GPT-2 使用它是为了处理互联网上多语言、多格式的文本，而无需预先定义字符集。代价是非 ASCII 字符（如中文）初始需要更多 token，但 BPE 会学习常见字节组合来缓解。

### Q4：词汇表大小怎么选？越大越好吗？

> 不是。词汇表大小是效率和参数的权衡：
> - **越大**：每个 token 承载更多信息，序列更短，推理更快；但 Embedding 层参数 = V × d_model，会增大模型
> - **越小**：Embedding 层更小，模型参数更少；但序列更长，计算量更大
>
> LLaMA 3 和 Qwen 2.5 选择 12~15 万的超大词汇表，是为了提升多语言（尤其中文）的编码效率，减少序列长度，这在推理成本上有显著收益。

### Q5：为什么同一个中文句子，不同模型的 token 数差异很大？

> 因为 Tokenizer 是基于训练语料的统计规律学习的。如果训练语料中文占比低，BPE 会优先学习英文常见组合，中文汉字无法高效合并，每个字可能需要 2~3 个 token。反之，如果中文语料充足（如 Qwen、DeepSeek），高频中文词和汉字会被合并为单个 token，编码效率大幅提升。

### Q6：Tokenizer 和模型是怎么配合的？

> Tokenizer 是在预训练前单独训练的，训练完成后词汇表大小 $V$ 固定。模型的 Embedding 层形状为 $(V, d_{model})$，最后一层线性层形状为 $(d_{model}, V)$，与词汇表一一对应。推理时：文本 → Tokenizer → token IDs → 查 Embedding 表 → 送入 Transformer → 输出 logits → 取 argmax 得到下一个 token ID → 解码为文本。

### Q7：特殊 token 有哪些？各自的作用是什么？

> | Token | 作用 |
> |------|------|
> | `<\|eos\|>` / `<|endoftext|>` | 序列结束，生成到此停止 |
> | `<\|pad\|>` | 填充，对齐批量中的序列长度 |
> | `<\|bos\|>` | 序列开始标记 |
> | `<\|im_start\|>` / `<\|im_end\|>` | ChatML 格式的对话角色边界 |
>
> 特殊 token 加入词汇表但不参与 BPE 训练。它们控制模型的输入格式和生成行为。

---

## 八、总结

**Tokenizer 的核心作用一句话总结**：

> 将文本切分为 token 序列并映射为 ID，是文本进入模型的第一道关。分词效率直接影响模型的推理速度和多语言能力。

**关键要点**：

| 要点 | 说明 |
|------|------|
| 子词分词 | 现代大模型的标准，平衡词汇表大小和 OOV |
| BPE 算法 | 自底向上贪心合并，高频完整保留，低频拆为子词 |
| Byte-Level | 以字节为初始单元，天然无 OOV，GPT 系列使用 |
| 词汇表权衡 | 越大序列越短但参数越多，新模型倾向更大词汇表 |
| 语料决定效果 | Tokenizer 质量取决于训练语料的语言覆盖和多样性 |

**从文本到模型的完整链路**：

```
"我爱深度学习"
  → Tokenizer 分词 → [23451, 31067, 18293, 8923]
  → 查 Embedding 表 → [v₁, v₂, v₃, v₄]
  → Transformer 处理 → [h₁, h₂, h₃, h₄]
  → 线性层 + Softmax → 下一个 token 的概率分布
  → 采样 / Greedy → 选出下一个 token ID
  → Tokenizer 解码 → 输出文本
```
