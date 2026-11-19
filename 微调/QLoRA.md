# QLoRA：在消费级显卡上微调大模型

> 微调一个 65B 参数的模型需要多少显存？答案是：只需要**一块 48GB 的 GPU**。
>
> QLoRA（Quantized Low-Rank Adaptation）是 LoRA 的"显存极限压缩版"——先将基础模型量化到 4-bit，再在上面叠加 LoRA 适配器进行训练。它由华盛顿大学的 Tim Dettmers 等人于 2023 年提出，论文：*QLoRA: Efficient Finetuning of Quantized LLMs*。
>
> 本文从 LoRA 的局限性出发，讲解 QLoRA 的三大核心创新、显存对比、与 LoRA 的区别，并给出实战配置。适合已有 LoRA 基础、希望在有限硬件上微调大模型的读者。

---

## 一、从 LoRA 到 QLoRA：为什么还需要更进一步？

### 1.1 LoRA 的显存瓶颈

回顾 [[LoRA]] 的做法：冻结原始权重，只训练低秩矩阵 $BA$。这确实大幅减少了训练参数量，但有一个容易被忽视的问题：

**推理时，原始模型仍然以 FP16/BF16（16-bit）加载在显存中。**

| 模型规模 | FP16 模型权重占用 | LoRA 训练开销 | 合计显存 |
|---------|----------------|--------------|---------|
| 7B      | ~14 GB         | ~2-4 GB      | ~16-18 GB |
| 13B     | ~26 GB         | ~4-6 GB      | ~30-32 GB |
| 33B     | ~66 GB         | ~8-12 GB     | ~74-78 GB |
| 65B     | ~130 GB        | ~12-16 GB    | ~142-146 GB |

对于 13B 以上的模型，即使是 LoRA，单张消费级显卡（24GB）也扛不住。

### 1.2 QLoRA 的核心思路

QLoRA 的思路很直接：**既然原始权重只是"摆在那里"不更新，为什么不用更少的位数来存储它？**

```
LoRA:   原始权重(FP16) + LoRA 适配器(FP16) → 显存大，但精度高
QLoRA:  原始权重(4-bit) + LoRA 适配器(BF16) → 显存小，精度靠适配器弥补
```

`★ Insight ─────────────────────────────────────`
**为什么 4-bit 量化后还能正常训练？** 关键在于：反向传播的梯度只流过 LoRA 适配器，不流过被量化的原始权重。原始权重的"数值精度"只影响前向传播时的激活值——4-bit 的精度损失通过 LoRA 的训练来补偿。这就像用一张模糊的地图（量化权重）搭配精确的导航仪（LoRA 适配器），最终照样能到达目的地。
`─────────────────────────────────────────────────`

---

## 二、QLoRA 的三大核心创新

QLoRA 不是简单地"先量化再加 LoRA"，它引入了三个关键技术来保证量化后的训练质量：

### 2.1 NormalFloat 4-bit 量化（NF4）

普通的 INT4 量化是将数值均匀映射到 16 个整数上。但神经网络的权重分布通常近似**正态分布**（均值为 0，标准差为 $\sigma$），在正态分布的"尾部"（远离均值的区域）INT4 会浪费很多量化级别。

**NF4 的做法：** 利用正态分布的分位数（quantile）来设计量化级别，使得每个级别覆盖相同的概率质量。

```
标准正态分布 N(0,1) 的分位数划分（4-bit，16个级别）：

INT4:  [-7, -6, -5, ..., 5, 6, 7]    ← 均匀分布，尾部浪费级别
NF4:   [-2.15, -1.38, -0.97, ..., 1.38, 2.15]  ← 分位数划分，每个级别概率相等
```

具体步骤：
1. 从标准正态分布 $N(0,1)$ 中取 $2^k + 1$ 个分位数边界
2. 相邻分位数的中点即为量化级别
3. 将权重归一化到 $[-1, 1]$ 区间后进行映射

**NF4 的优势：** 对正态分布的数据，NF4 比 INT4 的信息论最优性更好（理论上接近无信息损失的最优 4-bit 量化）。

### 2.2 双重量化（Double Quantization, DQ）

量化本身也需要存储一些额外信息（比如每个数据块的缩放因子 / scale）。对于 4-bit 量化，通常每 64 个参数需要一个 FP32 的 scale：

$$\text{额外开销} = \frac{32 \text{ bits}}{64 \text{ params}} = 0.5 \text{ bits/param}$$

双重量化的思路：**对这些 scale 也做量化。**

```
第一层量化：权重 → 4-bit (NF4)，需要 scale₁
第二层量化：scale₁ → 8-bit (FP8)，需要 scale₂

额外开销从 0.5 bits/param 降到约 0.127 bits/param
```

对于 65B 参数的模型，这大约节省了 **~3 GB 显存**。

### 2.3 分页优化器（Paged Optimizers）

训练过程中，优化器状态（如 Adam 的动量和方差）是显存的"大头"。当某个 mini-batch 的序列特别长时，可能会出现显存峰值超出 GPU 容量的情况。

**分页优化器**借用操作系统的虚拟内存思想：
- 优化器状态在 GPU 显存和 CPU 内存之间自动交换
- GPU 显存不够时，自动将部分状态"换出"到 CPU
- 需要时再"换入"

```
┌─────────────┐     自动换入/换出     ┌─────────────┐
│  GPU 显存    │ ◄─────────────────► │  CPU 内存    │
│  (活跃状态)   │                     │  (暂存状态)   │
└─────────────┘                     └─────────────┘
```

> 实际上，分页优化器只在极端情况下（长序列导致的显存峰值）才会触发交换，对训练速度影响很小。

---

## 三、显存对比：QLoRA vs LoRA vs Full Fine-Tuning

以 LLaMA 系列模型为例（单 GPU）：

| 微调方法 | 7B | 13B | 33B | 65B |
|---------|-----|------|------|------|
| Full FT (FP16) | ~56 GB | ~104 GB | ~264 GB | ~520 GB |
| LoRA (FP16) | ~16 GB | ~30 GB | ~74 GB | ~142 GB |
| **QLoRA (NF4)** | **~5-8 GB** | **~10-14 GB** | **~20-26 GB** | **~40-48 GB** |

**这意味着：**
- 7B 模型 → 普通游戏本（RTX 3060 12GB）就能训
- 13B 模型 → 单张 RTX 4090（24GB）足够
- 65B 模型 → 单张 A6000（48GB）可以搞定

`★ Insight ─────────────────────────────────────`
**QLoRA 的显存节省来自哪里？** 主要是三个部分：(1) 模型权重从 16-bit 降到 4-bit，节省约 75%；(2) 双重量化减少了 scale 的开销；(3) 分页优化器缓解峰值。但注意，**激活值（activations）的显存开销并没有减少**，所以 batch size 和序列长度仍然是显存的主要瓶颈之一。
`─────────────────────────────────────────────────`

---

## 四、QLoRA 的训练流程

### 4.1 前向传播

```
输入 x
  │
  ├──────────────────┐
  ▼                  ▼
 量化反量化 W(FP16)    W(NF4) + scale → dequantize → W(FP16)
  └──────┬───────────┘
         ▼
     h = W(FP16) · x + B · (A · x)    ← LoRA 路径用 BF16 计算
```

关键点：在计算时，4-bit 的权重被**反量化（dequantize）**回 BF16，然后与输入相乘。LoRA 的 $A$、$B$ 矩阵始终用 BF16 存储和更新。

### 4.2 反向传播

```
损失 L
  │
  ▼
梯度只流过 LoRA 适配器 (B, A)
  │
  ▼
更新 B 和 A（BF16），原始权重 W 不更新
```

梯度**不流过**被量化的原始权重——这是 QLoRA 能工作的根本原因。

### 4.3 训练步骤伪代码

```python
# 1. 加载预训练模型，并量化到 4-bit NF4
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,                    # 启用 4-bit 量化
    bnb_4bit_quant_type="nf4",            # 使用 NF4 量化
    bnb_4bit_use_double_quant=True,       # 启用双重量化
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算时用 BF16
)

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # 缩放因子
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. 用 PEFT 注入 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622%

# 4. 训练（原始权重 4-bit 冻结，只更新 LoRA 参数）
trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        bf16=True,
        optim="paged_adamw_8bit",         # 分页优化器
        ...
    ),
    train_dataset=dataset,
)
trainer.train()
```

---

## 五、QLoRA vs LoRA：怎么选？

| 维度 | LoRA | QLoRA |
|------|------|-------|
| **基础模型精度** | FP16/BF16 | 4-bit (NF4) |
| **显存需求** | 较高 | 降低 ~60-75% |
| **训练速度** | 较快 | 略慢（反量化有开销） |
| **最终效果** | 略优 | 几乎持平（论文证明差距极小） |
| **适用场景** | 显存充足 | 显存受限 |

**选择建议：**
- 如果显存够用 → 用 LoRA，速度更快、效果略好
- 如果显存不够 → 用 QLoRA，效果损失可忽略
- 想用大模型（>13B）→ QLoRA 几乎是唯一选择（除非你有多卡集群）

> **论文的关键结论：** QLoRA 在 Vicuna benchmark 上微调的 Guanaco-65B 达到了 ChatGPT 99.3% 的水平——用的只是一块 48GB 的 A6000。

---

## 六、实战配置建议

### 6.1 关键超参数

| 超参数                           | 推荐值          | 说明                        |
| ----------------------------- | ------------ | ------------------------- |
| `r` (rank)                    | 8 ~ 64       | 越大容量越高，但显存开销也越大。一般从 16 开始 |
| `lora_alpha`                  | 2r           | 经验值，控制 LoRA 路径的贡献权重       |
| `target_modules`              | q/k/v/o_proj | 注意力层是最常用的，也可加 MLP 层       |
| `bnb_4bit_quant_type`         | "nf4"        | 一定选 NF4，不要用普通的 INT4       |
| `bnb_4bit_use_double_quant`   | True         | 能省则省                      |
| `gradient_accumulation_steps` | 4-16         | 模拟更大 batch size，缓解显存压力    |

### 6.2 不同显存的模型选择参考

| 可用显存 | 推荐模型 | 配置要点 |
|---------|---------|---------|
| 8 GB | 7B (QLoRA) | batch_size=1, gradient_accumulation=16 |
| 16 GB | 7B (QLoRA) | batch_size=4, 可加更多 target_modules |
| 24 GB | 13B (QLoRA) | batch_size=4, r=32 |
| 48 GB | 33B-65B (QLoRA) | batch_size=4-8, r=64 |

---

## 七、常见问题

### Q1：QLoRA 训练后的模型怎么推理？
训练完成后，你可以：
1. **合并权重**：将 LoRA 适配器合并回基础模型，导出为完整模型（会变回 FP16 大小）
2. **分开加载**：加载 4-bit 基础模型 + LoRA 适配器，推理时实时合并

```python
# 合并导出
model = model.merge_and_unload()
model.save_pretrained("merged-model")
```

### Q2：NF4 和 INT4 到底差多少？
在论文实验中，NF4 在下游任务上比 INT4 提升了约 **0.5-1.0** 个百分点。虽然差距看起来不大，但它是"免费的午餐"——不需要额外的计算或显存。

### Q3：QLoRA 能用于多 GPU 训练吗？
可以。QLoRA 兼容 HuggingFace 的 `accelerate` 库和 DeepSpeed，支持数据并行和模型并行。但注意，量化的部分在多卡场景下的分片策略需要额外配置。

### Q4：QLoRA 微调后的模型效果会比全量微调差很多吗？
不会。论文实验显示，QLoRA 在多个 benchmark 上与 16-bit 全量微调的效果**几乎一致**。在 MMLU 上差距小于 1 个百分点。

---

## 八、总结

```
全量微调 ──(冻结权重)──► LoRA ──(量化基础模型)──► QLoRA
                              16-bit              4-bit
                              ~16GB (7B)          ~6GB (7B)
```

QLoRA 的本质是：**用精度换空间，用 LoRA 补精度**。它让"用消费级硬件微调超大模型"从不可能变成了现实，是目前开源社区微调大模型的主流方案（LLaMA Factory、Axolotl 等框架的默认配置就是 QLoRA）。

---

## 参考

- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021
- HuggingFace PEFT 文档: https://huggingface.co/docs/peft
- bitsandbytes 库: https://github.com/TimDettmers/bitsandbytes
