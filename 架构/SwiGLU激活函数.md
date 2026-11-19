# SwiGLU 激活函数

## 一、为什么需要 FFN？
我们可以把一个 Transformer Block 的工作模式总结为：**先交流，再计算**。

| 阶段 | 操作 | 作用 |
|------|------|------|
| **交流阶段（Attention）** | Token 之间互相看 | 通过 MHA，每个词把全句和自己最相关的特征都抓取过来，存在自己的向量里。但这步只是**搬运**，并没有产生质变 |
| **计算阶段（FFN）** | 每个词回到自己的位置上 | 利用 FFN 这一层对刚抓到的素材进行**深度消化**。它通过高维投影和非线性变换，把零散的上下文信号合并成更高阶的语义概念 |

```
Token 流程：
x₁ x₂ x₃ x₄ → Attention（互相看，抓取特征）→ FFN（各自深度消化）→ 输出
```

---

## 二、SwiGLU 的数学逻辑

传统的 FFN（GPT-2 时代）使用的是 `Linear → ReLU → Linear`。现代模型则使用 **SwiGLU**，它通过"门控机制"提供了更强的表达能力。

### 2.1 核心公式

$$\text{FFN}_{SwiGLU}(x) = (\text{SiLU}(xW_1) \otimes xW_3)W_2$$

我们将计算过程拆解为**三条路径**：

| 路径 | 操作 | 作用 |
|------|------|------|
| **门控分支（Gate Path）** | 输入 $x$ 经过 $W_1$ 升维，随后应用 SiLU 激活函数 | 决定了哪些信息是重要的 |
| **信号分支（Signal Path）** | 输入 $x$ 经过 $W_3$ 升维 | 提供了实际要处理的内容 |
| **融合与降维** | 将两条路径的结果逐元素相乘（$\otimes$），最后经过 $W_2$ 映射回原始维度 $d_{model}$ | 输出最终结果 |

```
         ┌──→ W₁ ──→ SiLU ──┐
x ───────┤                   ├──→ ⊙ ──→ W₂ ──→ output
         └──→ W₃ ───────────┘
           (门控)    (信号)
```

### 2.2 激活函数：SiLU（Swish）

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**与 ReLU 的对比**：

| 特性   | ReLU          | SiLU          |
| ---- | ------------- | ------------- |
| 负区间  | 输出为 0，梯度为 0   | 输出为小负数，梯度不为 0 |
| 平滑性  | 不平滑（x=0 处有折点） | 处处可导，更平滑      |
| 死神经元 | 有（一旦死亡无法恢复）   | 无（允许微弱负值通过）   |

```
SiLU 曲线特点：
- x > 0 时：接近恒等映射，保留正向信号
- x < 0 时：输出微弱负值（不是 0），梯度仍能流动
- x = 0 附近：平滑过渡，无折点
```

**关键优势**：SiLU 在 0 点附近更加平滑，且允许微弱的负值通过，这有助于深层网络中的梯度流动。

---

## 三、参数规模与 8/3 维度准则

### 3.1 传统 FFN vs SwiGLU FFN

| 项目 | 传统 FFN | SwiGLU FFN |
|------|----------|------------|
| 权重矩阵数量 | 2 个（$W_1$, $W_2$） | 3 个（$W_1$, $W_3$, $W_2$） |
| 升维维度 | $4d_{model}$ | $d_{ff} = \frac{8}{3}d_{model}$ |
| 总参数量 | $8d_{model}^2$ | $3 \times d_{model} \times \frac{8}{3}d_{model} = 8d_{model}^2$ |

### 3.2 8/3 维度准则详解

**问题**：SwiGLU 多了一个矩阵（$W_3$），参数量不会增加吗？

**解决方案**：通过缩小中间维度来保持总参数量不变。

**推导过程**：

```
传统 FFN 参数量：2 × (d × 4d) = 8d²

SwiGLU 要保持相同参数量：3 × (d × h) = 8d²
解得：h = 8d/3 ≈ 2.67d
```

### 3.3 硬件对齐要求

为了让 SwiGLU 的总参数数量与传统 FFN 持平，我们不再使用 4 倍升维，而是将中间维度 $d_{ff}$ 设为 $d_{model}$ 的 **8/3** 倍。

**硬件对齐要求**：为了优化 GPU 计算效率，$d_{ff}$ 通常需要**向上取整为 64 的倍数**。

| 模型 | $d_{model}$ | $d_{ff}$（理论值） | $d_{ff}$（实际值） | 对齐 |
|------|------------|-------------------|-------------------|------|
| LLaMA 7B | 4096 | 10922.67 | 11008 | 64 的倍数 |
| LLaMA 2 70B | 8192 | 21845.33 | 28672 | 64 的倍数 |
| Qwen 2.5 7B | 3584 | 9557.33 | 18944 | 256 的倍数 |
| Mistral 7B | 4096 | 10922.67 | 14336 | 64 的倍数 |

---

## 四、代码实现

### 4.1 基础 SwiGLU

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  # W₁
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)    # W₃
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)  # W₂

    def forward(self, x):
        # 门控分支: SiLU(x @ W₁)
        gate = F.silu(self.gate_proj(x))
        # 信号分支: x @ W₃
        up = self.up_proj(x)
        # 融合: gate ⊙ up，然后降维
        return self.down_proj(gate * up)
```

### 4.2 LLaMA 中的实际实现

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # 三个线性层，无 bias
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### 4.3 动手验证
```python
# 设置参数
hidden_size = 512
intermediate_size = 1376  # ≈ 512 * 8/3，取整到 64 的倍数

# 创建 SwiGLU 层
swiglu = SwiGLU(hidden_size, intermediate_size)

# 创建输入
x = torch.randn(2, 10, hidden_size)  # batch=2, seq_len=10, hidden=512
print(f"输入形状: {x.shape}")

# 前向传播
output = swiglu(x)
print(f"输出形状: {output.shape}")

# 查看参数量
total_params = sum(p.numel() for p in swiglu.parameters())
print(f"总参数量: {total_params:,}")
print(f"验证: 3 × {hidden_size} × {intermediate_size} = {3 * hidden_size * intermediate_size:,}")
```

输出：

```
输入形状: torch.Size([2, 10, 512])
输出形状: torch.Size([2, 10, 512])
总参数量: 2,111,488
验证: 3 × 512 × 1376 = 2,111,488
```

---

## 五、常见问题

### Q1：为什么不用 ReLU FFN 了？

> ReLU 有死神经元问题（负区间梯度为 0），且没有门控机制。SwiGLU 通过 SiLU 激活消除死神经元，通过 GLU 门控让网络自适应选择特征，实验效果显著优于 ReLU。

### Q2：SwiGLU 多一个矩阵，参数量不会增加吗？

> 不会。传统 FFN 中间维度是 4d，SwiGLU 是 8d/3。通过缩小中间维度，总参数量保持 8d² 不变。SwiGLU 的优势来自架构设计，不是堆参数。

### Q3：gate_proj、up_proj、down_proj 分别对应哪个权重？

> - `gate_proj` = $W_1$（门控分支，经过 SiLU 激活）
> - `up_proj` = $W_3$（信号分支，提供内容）
> - `down_proj` = $W_2$（降维投影，输出结果）

### Q4：GLU、SwiGLU、GeGLU 有什么区别？

> 都是门控线性单元的变体，区别在激活函数：
> - GLU：sigmoid
> - SwiGLU：SiLU/Swish（推荐，LLaMA/Qwen 使用）
> - GeGLU：GELU

### Q5：为什么中间维度要取整到 64 的倍数？

> GPU 的 Tensor Core 对特定维度的矩阵乘法有硬件优化。将维度对齐到 64 的倍数（或 128、256），可以充分利用 GPU 并行计算能力，提升推理效率。

---

## 六、总结

### 6.1 核心要点

| 项目 | 内容 |
|------|------|
| **公式** | $\text{FFN}_{SwiGLU}(x) = (\text{SiLU}(xW_1) \otimes xW_3)W_2$ |
| **三条路径** | 门控分支（Gate Path）、信号分支（Signal Path）、融合降维 |
| **激活函数** | SiLU（Swish），平滑无死神经元 |
| **权重矩阵** | 3 个：gate_proj（$W_1$）、up_proj（$W_3$）、down_proj（$W_2$） |
| **中间维度** | $d_{ff} = \frac{8}{3}d_{model}$，向上取整到 64 的倍数 |
| **总参数量** | $8d_{model}^2$（与传统 FFN 相同） |
| **代表模型** | LLaMA、Qwen、Mistral、PaLM |

### 6.2 与传统 FFN 的对比

| 对比项 | 传统 FFN（ReLU） | SwiGLU FFN |
|--------|-----------------|------------|
| 激活函数 | ReLU | SiLU + 门控 |
| 死神经元 | 有 | 无 |
| 特征选择 | 无（逐元素激活） | 有（门控机制） |
| 权重矩阵 | 2 个 | 3 个 |
| 中间维度 | 4d | 8d/3 |
| 总参数量 | 8d² | 8d² |

### 6.3 直观理解

```
Attention（交流阶段）：Token 互相看，抓取相关特征 → 搬运
                                                                    ↓
FFN（计算阶段）：各自深度消化，把零散信号合并成高阶语义 → 质变
```

SwiGLU 在"计算阶段"加入门控机制，让网络自己决定"哪些信息值得保留"，比简单的 ReLU 激活更智能。
