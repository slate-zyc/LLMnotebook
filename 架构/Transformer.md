# Transformer 架构详解

> 前面写的 LoRA、RLHF、GRPO，都是在"怎么调模型"。但 Transformer 本身是怎么工作的？Self-Attention 为什么能捕捉长距离依赖？位置编码为什么是必要的？本文从零讲透 Transformer 的每个组件，帮你补上所有后续知识的地基。
>
> Transformer 由 Google 在 2017 年提出（论文 *Attention Is All You Need*），彻底取代了 RNN/LSTM，成为当今几乎所有大模型的基座架构。

---

## 一、从 RNN 到 Transformer

### 1.1 RNN 的问题

在 Transformer 出现之前，序列建模主要靠 RNN（循环神经网络）：

```
输入：x₁ → x₂ → x₃ → x₄ → x₅
       ↓     ↓     ↓     ↓     ↓
      h₁ → h₂ → h₃ → h₄ → h₅   （逐步传递）
```

RNN 的核心问题：

| 问题        | 说明                              |
| --------- | ------------------------------- |
| **串行计算**  | 必须按顺序处理，x₅ 要等 x₁~x₄ 算完才能开始，无法并行 |
| **长距离遗忘** | 信息通过隐状态逐步传递，经过很多步后早期信息会衰减       |
| **训练慢**   | 串行 + 梯度需要沿时间步反向传播，GPU 利用率低      |

### 1.2 Transformer 的解决思路

> 一句话：**让序列中每个位置都能直接"看到"其他所有位置，不需要逐步传递。**

```
Transformer：每个位置同时处理，两两之间直接计算关联度

x₁ ←→ x₂ ←→ x₃ ←→ x₄ ←→ x₅
  ↘   ↗   ↘   ↗   ↘
   全部直接连接，没有顺序依赖
```

这解决了 RNN 的所有问题：
- **可并行**：所有位置同时计算
- **无遗忘**：任意两个位置之间距离都是 1
- **训练快**：充分利用 GPU 并行能力

---

## 二、Self-Attention（自注意力机制）

Self-Attention 是 Transformer 最核心的组件，理解了它就理解了 Transformer 的一半。

### 2.1 直觉理解

> Self-Attention 的本质：**对于序列中的每个 token，计算它和所有其他 token 的"关联度"，然后用关联度加权求和得到新的表示。**

类比阅读理解：
> 句子："**小明**把球传给了**小红**，**她**踢进了球门。"
>
> 当模型处理"她"这个 token 时，Self-Attention 会计算"她"和句中所有词的关联度，发现"她"和"小红"关联度最高，于是把"小红"的信息加权聚合过来，模型就知道"她"指的是小红。

### 2.2 Q、K、V 三元组

Self-Attention 的核心机制是为每个 token 生成三个向量：

| 向量 | 全称 | 含义 | 类比 |
|------|------|------|------|
| **Q**（Query） | 查询 | "我在找什么信息" | 搜索框里的关键词 |
| **K**（Key） | 键 | "我能提供什么信息" | 文章的标题/标签 |
| **V**（Value） | 值 | "我的具体内容" | 文章的实际内容 |

**计算过程**：

对于输入序列 $X$（每个 token 的 embedding），通过三个权重矩阵变换得到 Q、K、V：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

其中 $W_Q, W_K, W_V$ 就是你在 LoRA 笔记里提到的**可训练权重矩阵**，也是 LoRA 重点微调的目标。

### 2.3 Attention 的计算公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

分步拆解：

```
步骤1：QK^T → 计算每对 token 之间的相似度（原始分数）
        shape: (seq_len, seq_len)

步骤2：除以 √d_k → 缩放，防止点积值过大导致 softmax 梯度消失
        d_k 是每个头的维度

步骤3：softmax → 转换为概率分布（注意力权重），每行和为 1
        shape: (seq_len, seq_len)

步骤4：乘以 V → 用注意力权重对 Value 加权求和
        shape: (seq_len, d_k)
```

**具体例子**（简化为 3 个 token）：

```
假设：Q = [q₁, q₂, q₃],  K = [k₁, k₂, k₃],  V = [v₁, v₂, v₃]

1. 相似度矩阵 QK^T：
        k₁    k₂    k₃
   q₁ [ 2.1   0.5   0.3 ]
   q₂ [ 0.4   1.8   0.2 ]
   q₃ [ 0.1   0.6   1.9 ]

2. softmax 后（注意力权重）：
        k₁    k₂    k₃
   q₁ [ 0.75  0.15  0.10 ]   ← q₁ 主要关注 k₁（self）
   q₂ [ 0.15  0.75  0.10 ]   ← q₂ 主要关注 k₂（self）
   q₃ [ 0.08  0.18  0.74 ]   ← q₃ 主要关注 k₃（self）

3. 加权求和：
   output₁ = 0.75·v₁ + 0.15·v₂ + 0.10·v₃
   output₂ = 0.15·v₁ + 0.75·v₂ + 0.10·v₃
   output₃ = 0.08·v₁ + 0.18·v₂ + 0.74·v₃
```

### 2.4 为什么要除以 $\sqrt{d_k}$？

当 $d_k$（每个头的维度）很大时，$QK^T$ 的点积值会很大：

- $Q$ 和 $K$ 的每个元素独立同分布，均值 0 方差 1
- $d_k$ 个元素的点积均值为 0，**方差为 $d_k$**
- 值太大 → softmax 会变得非常"尖锐"（接近 one-hot）→ 梯度接近 0 → 训练不动

除以 $\sqrt{d_k}$ 把方差拉回 1，保证 softmax 处于梯度良好的区域。

---

## 三、Multi-Head Attention（多头注意力）

### 3.1 为什么要多头？

单头 Attention 只能从一个角度（一种关联模式）看待 token 之间的关系。多头让模型同时从**多个角度**捕捉不同类型的依赖关系。

> 类比：读一句话时，你同时关注语法结构、语义相似性、指代关系等多种模式。多头 Attention 就是让模型同时做多件事。

### 3.2 计算方式

假设 8 个头，$d_{model} = 512$：

```
每个头的维度 d_k = d_model / h = 512 / 8 = 64

Head₁: Attention(XW_Q¹, XW_K¹, XW_V¹) → output₁ (seq_len × 64)
Head₂: Attention(XW_Q², XW_K², XW_V²) → output₂ (seq_len × 64)
...
Head₈: Attention(XW_Q⁸, XW_K⁸, XW_V⁸) → output₈ (seq_len × 64)

拼接：[output₁; output₂; ...; output₈] → (seq_len × 512)
线性变换：concat × W_O → 最终输出 (seq_len × 512)
```

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O$$

`★ Insight ─────────────────────────────────────`
这里就和 LoRA 笔记里的 `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` 对应上了：
- `q_proj` = $W_Q$（将输入投影为 Q）
- `k_proj` = $W_K$（将输入投影为 K）
- `v_proj` = $W_V$（将输入投影为 V）
- `o_proj` = $W_O$（多头拼接后的输出投影）

LoRA 对这些矩阵做低秩分解，用极少的参数影响模型的注意力模式。
`─────────────────────────────────────────────────`

---

## 四、位置编码（Positional Encoding）

### 4.1 为什么需要位置编码？

Self-Attention 的计算只看 token 内容（Q、K），**不包含位置信息**。如果交换两个 token 的顺序，Attention 的输出不变（只是行/列顺序变了）。

```
"我爱你" 和 "你爱我" 在没有位置编码时，对 Self-Attention 来说是同一组 token
```

但语言中顺序很重要，所以需要显式注入位置信息。

### 4.2 正弦位置编码（原始 Transformer）

原始论文用正弦和余弦函数生成位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中 $pos$ 是位置，$i$ 是维度索引。

**直觉理解**：不同维度编码不同频率的位置信号，就像时钟的秒针、分针、时针用不同速度转动，组合起来可以唯一表示任何时刻。

**优点**：可以推广到训练时没见过的序列长度（因为是函数，不是查表）。

### 4.3 RoPE（旋转位置编码）

现代大模型（LLaMA、Qwen、DeepSeek）普遍使用 **RoPE（Rotary Positional Embedding）**：

核心思想：**不把位置编码加到 token embedding 上，而是将位置信息"旋转"到 Q 和 K 向量中。**

$$q'_m = R_m \cdot q_m, \quad k'_n = R_n \cdot k_n$$

其中 $R_m$ 是旋转矩阵，取决于位置 $m$。

**关键优势**：

| 特性 | 正弦位置编码 | RoPE |
|------|------------|------|
| 位置信息注入方式 | 加到 embedding | 旋转 Q、K |
| 相对位置 | 间接支持 | **直接支持**（$q_m^T k_n$ 只依赖 $m-n$） |
| 外推性（超长序列） | 较差 | 较好 |
| 计算效率 | 高 | 高 |

`★ Insight ─────────────────────────────────────`
RoPE 的核心好处：两个 token 的注意力分数只取决于它们的**相对距离** $m-n$，而不取决于各自的绝对位置。这让模型更容易学到"距离近的 token 关联更强"这种模式。这也是为什么 Qwen、LLaMA 等模型都选择 RoPE。更多位置编码细节见 [[RoPE旋转位置编码]]，注意力机制的各种高效变体见 [[注意力机制]]。
`─────────────────────────────────────────────────`

---

## 五、Transformer 的完整结构

### 5.1 原始 Encoder-Decoder 架构

```
┌─────────────────────────────────────────────────────┐
│                   Encoder（编码器）                    │
│  ┌────────────────────────────────────────────────┐ │
│  │  输入 Embedding + 位置编码                        │ │
│  └────────────────────┬───────────────────────────┘ │
│                       ▼                              │
│  ┌────────────────────────────────────────────────┐ │
│  │  Multi-Head Self-Attention                     │ │ ← 可以看到所有输入
│  │  + 残差连接 + LayerNorm                        │ │
│  └────────────────────┬───────────────────────────┘ │
│                       ▼                              │
│  ┌────────────────────────────────────────────────┐ │
│  │  Feed-Forward Network (FFN)                    │ │
│  │  + 残差连接 + LayerNorm                        │ │
│  └────────────────────┬───────────────────────────┘ │
│                       ▼                              │
│              × N 层堆叠                               │
└───────────────────────┬──────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│                   Decoder（解码器）                    │
│  ┌────────────────────────────────────────────────┐ │
│  │  输出 Embedding + 位置编码                        │ │
│  └────────────────────┬───────────────────────────┘ │
│                       ▼                              │
│  ┌────────────────────────────────────────────────┐ │
│  │  Masked Multi-Head Self-Attention               │ │ ← 只能看到之前的 token
│  │  + 残差连接 + LayerNorm                        │ │
│  └────────────────────┬───────────────────────────┘ │
│                       ▼                              │
│  ┌────────────────────────────────────────────────┐ │
│  │  Cross-Attention（关注 Encoder 输出）            │ │ ← 交叉注意力
│  │  + 残差连接 + LayerNorm                        │ │
│  └────────────────────┬───────────────────────────┘ │
│                       ▼                              │
│  ┌────────────────────────────────────────────────┐ │
│  │  Feed-Forward Network (FFN)                    │ │
│  │  + 残差连接 + LayerNorm                        │ │
│  └────────────────────┬───────────────────────────┘ │
│                       ▼                              │
│              × N 层堆叠                               │
└───────────────────────┬──────────────────────────────┘
                        ▼
                  输出概率分布
```

### 5.2 Decoder-Only 架构（现代 LLM）

GPT、LLaMA、Qwen、DeepSeek 等现代大模型都用 **Decoder-Only** 架构——只有 Decoder，没有 Encoder。

```
┌──────────────────────────────────────┐
│          Decoder-Only LLM             │
│                                      │
│  输入：你是一个有用的助手             │
│    ↓                                 │
│  Token Embedding + RoPE 位置编码     │
│    ↓                                 │
│  ┌────────────────────────────┐      │
│  │  Masked Self-Attention     │ × N  │
│  │  + 残差 + RMSNorm          │ 层   │
│  ├────────────────────────────┤      │
│  │  FFN (SwiGLU)              │      │
│  │  + 残差 + RMSNorm          │      │
│  └────────────────────────────┘      │
│    ↓                                 │
│  输出：下一个 token 的概率分布        │
└──────────────────────────────────────┘
```

**为什么不用 Encoder-Decoder？**

| 架构 | 代表模型 | 适用场景 |
|------|---------|---------|
| Encoder-Only | BERT | 理解类任务（分类、NER） |
| Encoder-Decoder | T5、原始 Transformer | 翻译、摘要（输入输出都是文本） |
| **Decoder-Only** | GPT、LLaMA、Qwen | **生成类任务（对话、写作、推理）** |

Decoder-Only 的优势：结构简单、可扩展性好、所有能力统一在一个架构里。

### 5.3 现代 LLM 的关键改进

| 组件 | 原始 Transformer | 现代 LLM（LLaMA/Qwen） |
|------|-----------------|----------------------|
| 位置编码 | 正弦位置编码 | **RoPE**（旋转位置编码） |
| LayerNorm | Post-LN（在残差之后） | **Pre-RMSNorm**（在残差分支内，用 RMSNorm） |
| FFN | ReLU 激活 | **SwiGLU** 激活 |
| Attention | 标准 MHA | **GQA**（分组查询注意力，见下文） |

---

## 六、GQA（分组查询注意力）

### 6.1 从 MHA 到 GQA

标准 MHA（Multi-Head Attention）：每个头有独立的 Q、K、V。

MQA（Multi-Query Attention）：所有头共享一组 K、V，只有 Q 各自独立。推理快但质量有损失。

**GQA（Grouped-Query Attention）**是折中方案：将头分成若干组，组内共享 K、V。

```
MHA（8头）：  Q₁K₁V₁  Q₂K₂V₂  Q₃K₃V₃  Q₄K₄V₄  ...  ← 8 组 KV，质量最好
GQA（8头2组）：Q₁K₁V₁  Q₂K₁V₁  Q₃K₁V₁  Q₄K₁V₁  ...  ← 2 组 KV，折中
MQA（8头）：  Q₁KV    Q₂KV    Q₃KV    Q₄KV    ...  ← 1 组 KV，最快
```

### 6.2 GQA 的价值

GQA 的核心收益在**推理阶段的 KV Cache**：

| 方案 | KV Cache 大小 | 推理速度 |
|------|-------------|---------|
| MHA | 最大 | 最慢 |
| **GQA** | **减少到 1/h（h=组数）** | **快** |
| MQA | 最小 | 最快 |

LLaMA 2 70B、Qwen 2.5 等模型都采用 GQA，兼顾质量和效率。

---

## 七、残差连接与 LayerNorm

### 7.1 残差连接（Residual Connection）

每个子层的输出 = 子层输入 + 子层计算结果：

$$\text{output} = x + \text{SubLayer}(x)$$

**作用**：
- 解决深层网络的梯度消失问题（梯度可以直接"跳过"子层传回去）
- 让网络更容易学习"微小的调整"（子层只需学增量 $\Delta x$）

### 7.2 LayerNorm vs RMSNorm

**LayerNorm**（原始 Transformer）：对每个 token 的所有维度计算均值和方差，归一化。

**RMSNorm**（现代 LLM）：只计算均方根（RMS），不减去均值，更简单更快：

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i} x_i^2}$$

实验证明 RMSNorm 效果与 LayerNorm 相当，但计算更快。

### 7.3 Pre-Norm vs Post-Norm

```
Post-Norm（原始）：  x → SubLayer(x) → LayerNorm(x + SubLayer(x))
Pre-Norm（现代）：   x → LayerNorm(x) → SubLayer(LayerNorm(x)) → x + result
```

现代 LLM 用 Pre-Norm：训练更稳定，不需要 warmup。

---

## 八、FFN（前馈网络）

每个 Transformer 层除了 Attention，还有一个 FFN：

### 8.1 原始 FFN

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

将维度从 $d_{model}$ 扩展到 $4d_{model}$ 再压缩回来。

### 8.2 SwiGLU（现代 LLM）

$$\text{SwiGLU}(x) = (xW_1 \odot \text{Swish}(xW_3)) W_2$$

三个权重矩阵，用 Swish 激活做门控。效果比 ReLU 好。

`★ Insight ─────────────────────────────────────`
 LoRA 笔记里提到"激进策略对所有线性层应用 LoRA（含 MLP）"，这里的 MLP 就是 FFN。Attention 层管"关注哪里"，FFN 层管"记住什么知识"。FFN 的参数量通常占模型总参数的 2/3，所以对 MLP 层加 LoRA 会显著增加可训练参数量。
`─────────────────────────────────────────────────`

---

## 九、面试高频问题

### Q1：Self-Attention 的计算复杂度是多少？为什么？

> $O(n^2 \cdot d)$，其中 $n$ 是序列长度，$d$ 是隐藏维度。$QK^T$ 的矩阵乘法是 $(n \times d) \times (d \times n) = O(n^2 d)$，这是 Self-Attention 最大的计算瓶颈。相比之下，RNN 是 $O(n \cdot d^2)$，短序列时 Transformer 更快，长序列时 RNN 更快。

### Q2：为什么 Self-Attention 能捕捉长距离依赖？

> RNN 中两个距离为 $n$ 的 token 需要经过 $n$ 步信息传递，每一步都有信息衰减。Self-Attention 中任意两个 token 之间直接计算关联度，距离永远是 1，不存在衰减问题。

### Q3：Masked Self-Attention 是什么？为什么 Decoder 需要它？

> Masked Self-Attention 在计算注意力权重时，将当前位置之后的位置设为 $-\infty$（softmax 后变为 0），确保生成第 $i$ 个 token 时只能看到前 $i-1$ 个 token。原因：训练时如果能看到后面的 token（"偷看答案"），模型就学不到真正的生成能力。

### Q4：多头注意力中，每个头的维度怎么算？为什么？

> 每个头的维度 $d_k = d_{model} / h$，其中 $h$ 是头数。例如 $d_{model} = 512, h = 8$，则 $d_k = 64$。这样多头的总计算量与单头大 Attention 基本相同（$h$ 个 $d_k$ 维的小 Attention vs 1 个 $d_{model}$ 维的大 Attention），但能捕捉更多样化的模式。

### Q5：RoPE 相比正弦位置编码的优势是什么？

> 1. **直接编码相对位置**：$q_m^T k_n$ 的值只依赖于相对距离 $m-n$，正弦编码只能间接表达相对位置
> 2. **外推性更好**：处理超过训练长度的序列时，RoPE 的衰减特性更平滑
> 3. **不需要额外参数**：正弦编码需要加到 embedding 上（增加了一个可学习的成分），RoPE 直接旋转 Q、K

### Q6：Encoder-Decoder、Decoder-Only、Encoder-Only 分别适合什么任务？

> - **Encoder-Only（BERT）**：理解类任务——分类、NER、句子相似度。能看到完整输入，双向理解
> - **Encoder-Decoder（T5）**：序列到序列任务——翻译、摘要。Encoder 理解输入，Decoder 生成输出
> - **Decoder-Only（GPT）**：生成类任务——对话、写作、推理。单向自回归生成，结构最简单，可扩展性最强

### Q7：GQA 相比 MHA 的优势是什么？

> GQA 将 Query 头分成若干组，组内共享 K、V。主要优势在推理阶段：KV Cache 的大小减少到 MHA 的 $1/g$（$g$ 是组数），显存占用大幅降低，推理速度更快，而质量损失很小。LLaMA 2 70B、Qwen 2.5 等都采用 GQA。

---

## 十、总结

**Transformer 的核心组件一句话总结**：

| 组件 | 作用 |
|------|------|
| Self-Attention | 让每个 token 关注所有 token，捕捉全局依赖 |
| 多头 Attention | 从多个角度同时关注不同模式 |
| 位置编码（RoPE） | 注入位置信息，让模型知道 token 的顺序 |
| 残差连接 | 解决深层梯度消失，让网络学习增量 |
| RMSNorm | 稳定训练，加速收敛 |
| FFN（SwiGLU） | 存储知识，进行非线性变换 |

**一张图总结数据流**：

```
Token IDs
  → Embedding + RoPE
    → [Masked Self-Attention → 残差+RMSNorm → FFN → 残差+RMSNorm] × N
      → 线性层
        → softmax
          → 下一个 token 的概率
```
