# KV Cache

> 推理优化.md 讲了量化、剪枝等压缩技术，但还有一个**推理阶段的核心优化**没讲——**KV Cache**。大模型生成文本时是一个词一个词生成的，每生成一个新词都要用到之前所有词的 Key 和 Value。如果每次都重新计算，效率极低。KV Cache 就是把计算过的 K、V 缓存起来，避免重复计算。

---

## 一、为什么需要 KV Cache

### 1.1 自回归生成的过程

大模型生成文本是**逐 token 生成**的：

```
输入: "你好"
Step 1: "你好" → 模型 → "世"    （生成第1个token）
Step 2: "你好世" → 模型 → "界"  （生成第2个token）
Step 3: "你好世界" → 模型 → "！" （生成第3个token）
...
```

### 1.2 没有 KV Cache 的情况

每个 step 都要对**整个序列**重新计算 Attention：

```
Step 1: 计算 Attention("你好")           → 得到 K₁, V₁
Step 2: 计算 Attention("你好世")         → 重新计算 K₁, V₁，再算 K₂, V₂
Step 3: 计算 Attention("你好世界")       → 重新计算 K₁, V₁, K₂, V₂，再算 K₃, V₃
...
```

**问题**：每一步都在重复计算之前的 K、V，计算量随序列长度二次增长。

### 1.3 有 KV Cache 的情况

把每一步计算的 K、V 缓存起来，下一步**只计算新 token 的 K、V**：

```
Step 1: 计算 Attention("你好")
        → 缓存 K₁, V₁
        → 得到 "世"

Step 2: 只计算 Attention("世" | K₁, V₁)
        → 用缓存的 K₁, V₁，只算 K₂, V₂
        → 缓存 [K₁, K₂], [V₁, V₂]
        → 得到 "界"

Step 3: 只计算 Attention("界" | K₁K₂, V₁V₂)
        → 用缓存的 K₁K₂, V₁V₂，只算 K₃, V₃
        → 得到 "！"
```

**效果**：每步只计算 1 个 token 的 K、V，计算量从 $O(n^2)$ 降到 $O(n)$。

`★ Insight ─────────────────────────────────────`
**类比理解**：想象你在查字典。没有 KV Cache 时，每查一个词都要从头翻完整本字典；有 KV Cache 时，你把查过的内容记在便签上，下次直接看便签就行。KV Cache 就是那张"便签"。
`─────────────────────────────────────────────────`

---

## 二、KV Cache 的工作原理

### 2.1 Attention 计算回顾

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

对于新生成的 token $q_n$，需要和**所有之前的** $k_1, k_2, ..., k_n$ 计算注意力：

$$\text{output}_n = \text{softmax}\left(\frac{q_n \cdot [k_1, k_2, ..., k_n]^T}{\sqrt{d_k}}\right) [v_1, v_2, ..., v_n]$$

### 2.2 缓存内容

每一步缓存的是**所有 token 的 K、V 向量**：

```
Step 1 后的缓存：
  K_cache = [k₁]     shape: (1, n_heads, d_k)
  V_cache = [v₁]     shape: (1, n_heads, d_v)

Step 2 后的缓存：
  K_cache = [k₁, k₂] shape: (2, n_heads, d_k)
  V_cache = [v₁, v₂] shape: (2, n_heads, d_v)

Step 3 后的缓存：
  K_cache = [k₁, k₂, k₃] shape: (3, n_heads, d_k)
  V_cache = [v₁, v₂, v₃] shape: (3, n_heads, d_v)
```

### 2.3 Prefill vs Decode

大模型推理分为两个阶段：

| 阶段 | 做什么 | 计算特点 | KV Cache 状态 |
|------|--------|---------|--------------|
| **Prefill（预填充）** | 处理输入 prompt | 并行计算所有 token 的 K、V | 一次性填充整个缓存 |
| **Decode（解码）** | 逐 token 生成 | 每步只计算 1 个 token | 每步追加 1 个 K、V |

```
Prefill 阶段：
  输入: [你, 好, 世, 界]（4 个 token）
  计算: 并行计算 4 个 token 的 Q、K、V
  缓存: K_cache = [k₁, k₂, k₃, k₄], V_cache = [v₁, v₂, v₃, v₄]

Decode 阶段：
  Step 5: q₅ × [k₁...k₅] → 生成新 token → 追加 k₅, v₅ 到缓存
  Step 6: q₆ × [k₁...k₆] → 生成新 token → 追加 k₆, v₆ 到缓存
  ...
```

### 2.4 伪代码实现

```python
class KVCache:
    def __init__(self, max_seq_len, n_heads, d_k):
        self.K = torch.zeros(max_seq_len, n_heads, d_k)
        self.V = torch.zeros(max_seq_len, n_heads, d_k)
        self.cur_len = 0
    
    def update(self, k_new, v_new):
        """追加新的 K、V 到缓存"""
        batch_size = k_new.shape[0]
        self.K[self.cur_len:self.cur_len+1] = k_new
        self.V[self.cur_len:self.cur_len+1] = v_new
        self.cur_len += 1
    
    def get(self):
        """返回缓存的 K、V"""
        return self.K[:self.cur_len], self.V[:self.cur_len]

# 推理过程
kv_cache = KVCache(max_len=2048, n_heads=32, d_k=64)

# Prefill：处理输入
q, k, v = model.embed_and_project(input_ids)
kv_cache.update(k, v)
output = attention(q, k, v)

# Decode：逐 token 生成
for _ in range(max_new_tokens):
    q, k_new, v_new = model.embed_and_project(last_token)
    kv_cache.update(k_new, v_new)
    k_cached, v_cached = kv_cache.get()
    output = attention(q, k_cached, v_cached)
    next_token = sample(output)
```

---

## 三、KV Cache 的显存占用

### 3.1 显存计算公式

每个 token 的 KV Cache 大小：

$$\text{KV Cache per token} = 2 \times n\_layers \times n\_heads \times d\_k \times \text{bytes\_per\_param}$$

其中：
- 2：K 和 V 各一份
- $n\_layers$：Transformer 层数
- $n\_heads$：注意力头数
- $d\_k$：每个头的维度
- bytes_per_param：数据类型字节数（FP16=2, FP32=4）

### 3.2 常见模型的 KV Cache 大小

| 模型 | 参数 | 层数 | 头数 | d_k | FP16 KV Cache/序列 |
|------|------|------|------|-----|-------------------|
| LLaMA 7B | 7B | 32 | 32 | 128 | 1 KB/token → 2 MB/2048 tokens |
| LLaMA 70B | 70B | 80 | 64 | 128 | 8 KB/token → 16 MB/2048 tokens |
| GPT-4 级 | ~1.8T | ~120 | ~128 | 128 | ~200 KB/token → 400 MB/2048 tokens |

### 3.3 批量推理的显存压力

```
单条序列（2048 tokens）的 KV Cache：
  LLaMA 7B: 2 MB
  LLaMA 70B: 16 MB

批量推理（batch_size=32）：
  LLaMA 7B: 64 MB
  LLaMA 70B: 512 MB

长序列（8192 tokens）+ 批量（batch=32）：
  LLaMA 7B: 256 MB
  LLaMA 70B: 2 GB ← 已经很可观了！
```

`★ Insight ─────────────────────────────────────`
**KV Cache 是推理显存的主要瓶颈**：模型参数只需要加载一次，但 KV Cache 随序列长度和 batch size 线性增长。这就是为什么"长上下文"模型需要特殊的显存优化——128K context 的 KV Cache 可能比模型参数本身还大。
`─────────────────────────────────────────────────`

---

## 四、KV Cache 优化技术

### 4.1 GQA（Grouped-Query Attention）

**问题**：标准 MHA 每个头都有独立的 K、V，KV Cache 大小 = $n\_heads \times d\_k$。

**GQA 解决方案**：多个 Query 头共享一组 K、V。

```
MHA（8 头）：
  Q: 8 个头独立
  K: 8 个头独立
  V: 8 个头独立
  KV Cache: 8 × d_k

GQA（8 头，2 组 KV）：
  Q: 8 个头独立
  K: 2 组（每 4 个 Q 头共享 1 个 K 头）
  V: 2 组
  KV Cache: 2 × d_k → 节省 75%

MQA（8 头，1 组 KV）：
  Q: 8 个头独立
  K: 1 个头（所有 Q 头共享）
  V: 1 个头
  KV Cache: 1 × d_k → 节省 87.5%
```

**显存节省**：

| 方案 | KV Cache 大小 | 质量损失 |
|------|-------------|---------|
| MHA | $n\_heads \times d\_k$ | 无 |
| GQA（8头2组） | $n\_heads/4 \times d\_k$ | 极小 |
| MQA | $1 \times d\_k$ | 有一定损失 |

### 4.2 MQA（Multi-Query Attention）

```python
# MQA 实现示例
class MultiQueryAttention(nn.Module):
    def forward(self, x, kv_cache=None):
        q = self.W_q(x)  # (batch, seq, n_heads * d_k)
        k = self.W_k(x)  # (batch, seq, 1 * d_k)  ← 只有 1 个头
        v = self.W_v(x)  # (batch, seq, 1 * d_k)
        
        # 如果是 Decode 阶段，追加到缓存
        if kv_cache is not None:
            k = torch.cat([kv_cache.k, k], dim=1)
            v = torch.cat([kv_cache.v, v], dim=1)
        
        # K、V 广播到所有 Q 头
        k = k.repeat(1, 1, self.n_heads, 1)  # 广播
        v = v.repeat(1, 1, self.n_heads, 1)
        
        return attention(q, k, v)
```

### 4.3 PagedAttention（vLLM）

**问题**：KV Cache 需要连续显存，但生成长度不确定，导致显存碎片化。

**PagedAttention 解决方案**：借鉴操作系统虚拟内存的分页思想。

```
传统方式：预分配最大长度的连续显存
  [最大2048 tokens] → 实际用了 100 tokens → 浪费 1948 tokens 的显存

PagedAttention：把 KV Cache 分成固定大小的"页"
  页 1: [token 1-16]   ← 已使用
  页 2: [token 17-32]  ← 已使用
  页 3: [token 33-48]  ← 未使用（不占用显存）
  ...
  需要时动态分配新页，不需要时立即释放
```

**优势**：

| 问题 | 传统方式 | PagedAttention |
|------|---------|----------------|
| 显存碎片 | 严重（预分配） | 无（动态分配） |
| 显存利用率 | 50-70% | 95%+ |
| batch size | 受限于预分配 | 可以更大 |

### 4.4 滑动窗口注意力（Sliding Window Attention）

**思想**：每个 token 只关注最近的 $W$ 个 token，而不是全部历史。

```
标准 Attention：
  token₁₀ 关注 [token₁, token₂, ..., token₁₀]

滑动窗口（W=4）：
  token₁₀ 关注 [token₇, token₈, token₉, token₁₀]

KV Cache 大小：固定为 W × d_k，不随序列增长
```

**代价**：丢失长距离依赖信息。

**改进：Longformer 的 Dilated Sliding Window**：

```
扩大窗口范围，但跳过一些 token（扩张注意力）

窗口大小 W=4，扩张率 d=2：
  token₁₀ 关注 [token₂, token₄, token₆, token₈, token₁₀]
                    ↑     ↑     ↑     ↑     ↑
                  跳过   跳过   跳过   跳过   当前
```

### 4.5 StreamingLLM（流式推理）

**思想**：KV Cache 只保留开头几个 token（Attention Sink）+ 最近的 $W$ 个 token。

```
完整 KV Cache: [t₁, t₂, t₃, ..., t₉₈, t₉₉, t₁₀₀]
                     ↓
StreamingLLM:  [t₁, t₂, ..., t₉₅, t₉₆, t₉₇, t₉₈, t₉₉, t₁₀₀]
                ↑___开头4个___↑    ↑_____最近96个_____↑
                (Attention Sink)
```

**为什么保留开头几个 token？**

开头的 token 充当"Attention Sink"——所有 token 都会对它们有一定的注意力权重（类似"默认关注"），如果删掉开头，注意力分布会崩溃。

---

## 五、KV Cache 与量化

### 5.1 KV Cache 量化

把 KV Cache 从 FP16 量化到 INT8 或 INT4，进一步节省显存：

| 精度 | 每 token 显存（LLaMA 70B） | 质量影响 |
|------|---------------------------|---------|
| FP16 | 8 KB | 基准 |
| INT8 | 4 KB | 极小 |
| INT4 | 2 KB | 有一定影响 |

### 5.2 KV Cache 量化实现

```python
# 量化 KV Cache 的简单实现
def quantize_kv_cache(k_cache, v_cache, bits=8):
    """将 KV Cache 量化到指定位数"""
    if bits == 8:
        # INT8 量化
        k_scale = k_cache.abs().max() / 127
        k_quant = (k_cache / k_scale).round().to(torch.int8)
        v_scale = v_cache.abs().max() / 127
        v_quant = (v_cache / v_scale).round().to(torch.int8)
        return k_quant, v_quant, k_scale, v_scale
    elif bits == 4:
        # INT4 量化（需要分组量化）
        ...

def dequantize_kv_cache(k_quant, v_quant, k_scale, v_scale):
    """反量化"""
    return k_quant.float() * k_scale, v_quant.float() * v_scale
```

---

## 六、面试高频问题

### Q1：什么是 KV Cache？为什么需要它？

> KV Cache 是推理阶段缓存历史 token 的 Key 和 Value 向量的技术。自回归生成时，每生成一个新 token 都需要和所有之前的 token 计算 Attention。如果每次都重新计算 K、V，计算量随序列长度二次增长。KV Cache 把计算过的 K、V 缓存起来，每步只计算新 token 的 K、V，计算量从 $O(n^2)$ 降到 $O(n)$，推理速度大幅提升。

### Q2：KV Cache 的显存占用如何计算？

> 每个 token 的 KV Cache 大小 = 2 × n_layers × n_heads × d_k × bytes_per_param。其中 2 代表 K 和 V 各一份。例如 LLaMA 7B（32 层，32 头，d_k=128），FP16 下每个 token 需要 1 KB。2048 tokens 的序列需要 2 MB。批量推理时显存压力线性增长，batch=32 需要 64 MB。

### Q3：GQA 如何减少 KV Cache？

> GQA（Grouped-Query Attention）让多个 Query 头共享一组 K、V。标准 MHA 每个头都有独立的 K、V，KV Cache 大小与头数成正比。GQA 将头分组（如 8 个 Q 头分 2 组，每 4 个 Q 头共享 1 个 KV 头），KV Cache 减少到原来的 1/4。LLaMA 2 70B、Qwen 2.5 都用 GQA，兼顾质量和效率。

### Q4：PagedAttention 解决了什么问题？

> PagedAttention 解决的是 KV Cache 的显存碎片化问题。传统方式预分配最大序列长度的连续显存，但实际生成长度往往远小于最大值，导致大量显存浪费。PagedAttention 借鉴操作系统分页思想，把 KV Cache 分成固定大小的"页"，动态分配和释放，显存利用率从 50-70% 提升到 95%+。vLLM 使用这个技术。

### Q5：Prefill 和 Decode 阶段有什么区别？

> **Prefill 阶段**：处理输入 prompt，一次性并行计算所有 token 的 Q、K、V，填充整个 KV Cache。计算密集，GPU 利用率高。**Decode 阶段**：逐 token 生成，每步只计算 1 个 token 的 Q、K、V，追加到 KV Cache。访存密集（需要读取整个 KV Cache），GPU 利用率低。Decode 阶段是推理的主要瓶颈。

### Q6：为什么 Decode 阶段 GPU 利用率低？

> Decode 阶段每步只生成 1 个 token，计算量很小（几个矩阵-向量乘法），但需要读取整个 KV Cache（访存量大）。这导致计算和访存的比例（算术强度）很低，GPU 的计算单元大量空闲等待数据。这种模式称为"Memory-bound"，受限于显存带宽而非计算能力。优化方向：减少 KV Cache 大小（GQA、量化）、增加 batch size（提高并行度）。

---

## 七、总结

| 概念 | 核心要点 |
|------|---------|
| KV Cache | 缓存历史 K、V，避免重复计算 |
| Prefill | 处理输入，并行计算，填充缓存 |
| Decode | 逐 token 生成，只计算新 token 的 K、V |
| GQA/MQA | 多 Q 头共享 KV，减少缓存大小 |
| PagedAttention | 动态分页，消除显存碎片 |
| 滑动窗口 | 只保留最近 W 个 token 的 KV |
| KV 量化 | INT8/INT4 量化 KV Cache，进一步省显存 |

**一句话记忆**：

> KV Cache 就是"记住你查过的东西"——生成第 N 个 token 时，不用从头重新算前 N-1 个 token 的 K、V，直接用缓存的结果。GQA 是"几个人共用一本字典"，PagedAttention 是"按需借书，用完归还"。
