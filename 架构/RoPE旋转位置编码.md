# RoPE 旋转位置编码

## 一、为什么需要位置编码

Transformer 的自注意力机制本身是**置换不变的（permutation invariant）**——打乱输入 token 的顺序，输出不变。但自然语言是有序的，"狗咬人"和"人咬狗"语义完全不同。因此需要引入**位置编码**来注入序列的位置信息。

位置编码方案的演进：

| 方案 | 代表模型 | 核心思路 |
|------|---------|---------|
| 绝对位置编码 | BERT, GPT-1/2 | 给每个位置生成一个固定向量，加到 token embedding 上 |
| 相对位置编码 | Transformer-XL, T5 | 在注意力计算中加入 token 之间的相对距离信息 |
| **旋转位置编码（RoPE）** | LLaMA, GPT-NeoX, Qwen | 将位置信息编码为旋转操作，通过乘法注入 |

---

## 二、RoPE 的核心思想

> **一句话概括：通过旋转矩阵将位置信息编码到 query/key 向量中，使得两个向量的点积只依赖于它们的相对位置差。**

### 2.1 二维直觉

先从二维情况理解。设 query 向量为 $q = (q_0, q_1)$，key 向量为 $k = (k_0, k_1)$。

位置 $m$ 的编码通过**旋转**实现：

$$
R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
$$

其中 $\theta$ 是一个固定的旋转角度。

对 query 和 key 分别施加旋转：

$$
\tilde{q}_m = R_m q, \quad \tilde{k}_n = R_n k
$$

计算内积：

$$
\tilde{q}_m^T \tilde{k}_n = (R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k
$$

关键性质：$R_m^T R_n = R_{n-m}$，即两个旋转矩阵的"内积"只与**位置差 $(n-m)$** 有关。

```
★ Insight ─────────────────────────────────────
这个等式是 RoPE 的灵魂：通过旋转矩阵的乘法性质，将绝对位置的编码
自然地转化为相对位置的信息——不需要额外的距离矩阵或偏置项。
─────────────────────────────────────────────────
```

### 2.2 高维推广

实际场景中 query/key 维度 $d$ 通常是 64/128/256，不是二维。RoPE 将 $d$ 维向量**两两配对**，每对使用不同的旋转角，等价于多个二维旋转的直积：

$$
R_m = \text{diag}(R_{m,\theta_1}, R_{m,\theta_2}, \ldots, R_{m,\theta_{d/2}})
$$

其中每对的旋转角为：

$$
\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1
$$

---

## 三、数学公式推导

### 3.1 旋转的显式形式

对维度偶数下标 $2i$ 和奇数下标 $2i+1$ 配对，旋转后的向量为：

$$
\tilde{q}_m^{[2i]} = q_{2i} \cos(m\theta_i) - q_{2i+1} \sin(m\theta_i)
$$

$$
\tilde{q}_m^{[2i+1]} = q_{2i} \sin(m\theta_i) + q_{2i+1} \cos(m\theta_i)
$$

这等价于将复数 $q_{2i} + j \cdot q_{2i+1}$ 乘以 $e^{jm\theta_i}$。

### 3.2 内积的相对位置性质

$$
\langle \tilde{q}_m, \tilde{k}_n \rangle = \sum_{i=0}^{d/2-1} \left[ q_{2i}k_{2i} + q_{2i+1}k_{2i+1} \right] \cos((n-m)\theta_i) + \left[ q_{2i}k_{2i+1} - q_{2i+1}k_{2i} \right] \sin((n-m)\theta_i)
$$

可以看出，点积结果只和相对位置 $m - n$ 有关，和绝对位置 $m, n$ 无关。

### 3.3 实际计算优化

实际实现中不需要构造旋转矩阵，直接对向量分量做**两两分组然后旋转**即可：

```python
def apply_rope(x, positions, dim):
    """
    x: (batch, seq_len, d_head)
    positions: (seq_len,)  每个 token 的位置索引
    """
    # 计算每对维度的旋转角
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # (d/2,)

    # 计算 cos 和 sin
    angles = positions[:, None] * theta[None, :]  # (seq_len, d/2)
    cos = angles.cos()  # (seq_len, d/2)
    sin = angles.sin()  # (seq_len, d/2)

    # 将 x 的偶数维和奇数维分开
    x_even = x[..., 0::2]  # (batch, seq_len, d/2)
    x_odd = x[..., 1::2]   # (batch, seq_len, d/2)

    # 旋转
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd  = x_even * sin + x_odd * cos

    # 交错合并回去
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)
    return x_rotated
```

```
★ Insight ─────────────────────────────────────
注意：RoPE 只对 query 和 key 施加旋转，不对 value 施加。
因为注意力的最终输出是 value 的加权和，位置信息
已经通过 softmax 权重（由旋转后的 q·k 决定）编码进去了。
─────────────────────────────────────────────────
```

---

## 四、与其他位置编码方案的对比

### 4.1 vs 绝对位置编码

| 维度 | 绝对位置编码 | RoPE |
|------|------------|------|
| 编码方式 | 位置向量**加**到 embedding 上 | 旋转操作**乘**到 q/k 上 |
| 相对位置 | 无法直接表达 | 天然支持（点积只依赖相对距离） |
| 外推性 | 泛化到更长序列效果差 | 可通过调整旋转角扩展到更长序列 |
| 计算开销 | 需要额外的位置 embedding 矩阵 | 仅需对 q/k 做逐元素运算 |

### 4.2 vs 相对位置编码

| 维度 | 相对位置编码 | RoPE |
|------|------------|------|
| 实现方式 | 修改注意力分数矩阵 | 修改 q/k 向量 |
| 与注意力机制耦合度 | 高（需要改 attention 计算） | 低（只需在 q/k 线性变换后加一步） |
| 推理效率 | KV Cache 复杂 | KV Cache 简单（key 已编码好位置） |

---

## 五、RoPE 在主流模型中的应用

| 模型 | RoPE 变体 | 上下文长度 |
|------|----------|-----------|
| LLaMA 1/2 | 基础 RoPE + NTK Scaling | 4K / 2-128K |
| LLaMA 3 | RoPE + 500K base + 长上下文微调 | 128K |
| Qwen 2/3 | YaRN | 32K-128K |
| Mistral | Sliding Window + RoPE | 32K |
| DeepSeek-V2/V3 | 基础 RoPE（仅部分注意力头） | 128K |

---

## 六、代码实现要点

### 6.1 KV Cache 中的位置处理

自回归生成时，每一步只需要对**当前新 token** 的 q/k 做旋转。已缓存的 key 不需要重新计算旋转——因为旋转是在 cache 之前就施加好了的。

```python
# 推理时：只需对新 token 位置做 RoPE
new_q = apply_rope(new_q, position=current_pos)  # 只旋转当前步
# cached_k 已经是旋转后的，直接拼接即可
```

### 6.2 注意 head 维度的分组

旋转操作是**两两配对**，所以 `d_head` 必须是偶数。典型值：d_head = 64 或 128。

### 6.3 precompute cos/sin

对于静态长度的场景（如训练时 batch 固定），可以预计算 cos/sin 表避免重复计算：

```python
# 预计算到最大长度
max_len = 4096
theta = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
positions = torch.arange(max_len)
angles = positions[:, None] * theta[None, :]  # (max_len, d_head // 2)
cos_cached = angles.cos()
sin_cached = angles.sin()
```

---

## 七、常见面试问题

**Q1：RoPE 为什么能编码相对位置信息？**

因为旋转矩阵满足 $R_m^T R_n = R_{n-m}$，两个旋转后的向量的内积只依赖于位置差。

**Q2：为什么只对 q 和 k 做旋转，不对 v 做？**

因为注意力机制中只有 Q 和 K 参与点积计算，V 只是被加权求和，注意力输出是 value 的加权和，位置信息已经通过 softmax 权重（由旋转后的 q·k 决定）编码。对 v 做旋转反而会破坏信息。

**Q3：RoPE 的旋转角 $\theta_i = 10000^{-2i/d}$ 中 10000 这个数是怎么来的？**

来自原始论文的实验选择，类似于 Sinusoidal Positional Encoding 中的 10000。它控制了能编码的"有效位置范围"。增大这个 base 可以支持更长序列（LLaMA 3 使用了 500000）。

**Q4：RoPE 和 ALiBi 的区别是什么？**

- RoPE：修改 q/k 向量本身（旋转）
- ALiBi：在注意力分数上直接加一个与距离成比例的偏置（$-|m-n| \cdot m_h$），不需要修改向量
- ALiBi 实现更简单，但 RoPE 配合 NTK Scaling 在长度外推上更灵活

**Q5：sin 和 cos 的计算精度对 RoPE 有影响吗？**

有。在 FP16 下，当位置很大时 cos/sin 的精度会下降，导致位置编码退化。这是使用 FP32 计算 RoPE 的原因——LLaMA 等模型会在 RoPE 计算时切换到 FP32 精度。

**Q6：填空题 - Transformer 的 Self-Attention 本身是 ______ 不变的，所以需要位置编码。**

**置换（Permutation）** 不变的。即打乱 token 顺序，Self-Attention 的输出不变。

**Q7：判断题 - RoPE 引入了额外的可学习参数。（对/错？）**

**错。** RoPE 的旋转频率是预计算的固定值，不需要学习。

**Q8：为什么 RoPE 的注意力分数只依赖相对位置？**

因为旋转矩阵有性质 $R_m^T R_n = R_{n-m}$，所以 Query 和 Key 的内积 $(R_m q)^T(R_n k) = q^T R_{n-m} k$ 只取决于 $(n-m)$。

**Q9：计算题 - MiniMind 的 head_dim=96，RoPE 将其分成多少组？**

head_dim=96，两两分组，共 **48 组**。

**Q10：思考题 - 如果把 rope_theta 从 1e6 减小到 1e4，对模型的长文本能力会有什么影响？**

theta 减小会让旋转频率增大、旋转变快，模型在短距离上的位置区分更敏感，但在长距离上会出现"旋转重复"现象，降低长文本处理能力。
