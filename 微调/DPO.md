# DPO：直接偏好优化

> DPO（Direct Preference Optimization）把[[RLHF]]中"先训奖励模型、再跑 PPO"的两步流程压缩成一步——跳过奖励模型，直接用偏好数据训练策略模型。它让对齐训练从"四个模型的强化学习工程"变成了"两个模型的监督学习"，极大降低了门槛。
>
> 本文从 DPO 的数学推导出发，深入讲解隐式奖励的直觉、β 的作用、常见训练陷阱、DPO 的局限性，以及 SimPO/KTO/IPO 等变种。适合已了解 [[RLHF]] 基础流程、希望深入理解 DPO 原理与实践的读者。

---

## 一、DPO 要解决什么问题？

回顾 [[RLHF]] 的标准流程：

```
SFT 模型 → 训练 Reward Model → PPO 优化（需4个模型）
```

这条链路有三个痛点：

| 痛点 | 具体问题 |
|------|---------|
| **工程复杂** | 需要同时维护 4 个模型（策略、参考、奖励、价值），代码量大 |
| **显存巨大** | 4 个模型加载到 GPU，显存需求约为 SFT 的 4 倍 |
| **训练不稳定** | PPO 本身是在线强化学习，超参敏感，容易 reward hacking |

DPO 的核心问题：**能不能不训练奖励模型，直接从偏好数据学到最优策略？**

---

## 二、数学推导：从 RLHF 到 DPO

### 2.1 RLHF 的优化目标

在 [[RLHF]] 中，PPO 最大化以下目标：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi_\theta(y|x)} [r(x, y)] - \beta \cdot \text{KL}(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x))$$

即：在不偏离参考模型太远（KL 约束）的前提下，最大化奖励。

### 2.2 最优策略的解析解

上述带 KL 约束的优化问题有**解析解**（closed-form solution）：

$$\pi^*(y|x) = \frac{1}{Z(x)} \cdot \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(x,y)}{\beta}\right)$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \cdot \exp(r(x,y)/\beta)$ 是配分函数。

**关键一步：反解奖励函数**

对上式两边取对数并整理：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

这就是 DPO 的核心公式——**奖励可以用策略模型和参考模型的 log 概率比来表示**。

`★ Insight ─────────────────────────────────────`
**为什么这个变换如此重要？** 传统 RLHF 中，奖励是一个独立的神经网络，需要额外训练。而 DPO 证明了：最优策略和最优奖励之间存在一一对应关系，知道了最优策略就能算出最优奖励，反过来也一样。因此我们不需要显式地训练奖励模型——策略模型本身就隐含了奖励信息。
`─────────────────────────────────────────────────`

### 2.3 消除配分函数

偏好数据使用 Bradley-Terry 模型建模：人类更偏好 $y_w$ 而非 $y_l$ 的概率为：

$$p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

将 $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$ 代入：

$$p(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

注意 $\beta \log Z(x)$ 项**恰好抵消了**——这就是 DPO 的魔法：不需要计算配分函数。

### 2.4 DPO 损失函数

最大化偏好概率，等价于最小化负对数似然：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \underbrace{\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}}_{\text{chosen 的隐式奖励}} - \beta \underbrace{\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}}_{\text{rejected 的隐式奖励}} \right) \right]$$

展开来看，每个部分的含义：

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 策略模型（正在训练） |
| $\pi_{\text{ref}}$ | 参考模型（冻结的 SFT 模型） |
| $y_w$ | 人类偏好的回答（chosen） |
| $y_l$ | 人类不偏好的回答（rejected） |
| $\beta$ | 温度系数，控制偏离参考模型的程度 |
| $\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ | 隐式奖励：策略模型对参考模型的 log 概率比 |

---

## 三、直觉理解

### 3.1 隐式奖励是怎么工作的？

```
                        参考模型 π_ref
                      ╱               ╲
好回答 y_w →  log P(y_w)  =  -5.2      → 差值 = β × (0.3) = 隐式奖励
策略模型 π_θ  log P(y_w)  =  -4.9

                        参考模型 π_ref
                      ╱               ╲
差回答 y_l →  log P(y_l)  =  -6.0      → 差值 = β × (-0.5) = 隐式惩罚
策略模型 π_θ  log P(y_l)  =  -6.5
```

DPO 的逻辑：**如果策略模型对好回答的概率提升得比差回答多，就说明它学对了。**

### 3.2 β 的作用

$\beta$ 控制两件事的平衡：

| β 值 | 行为 | 效果 |
|------|------|------|
| β 很小（如 0.01） | 弱化 KL 约束，策略模型可以大幅偏离参考模型 | 更激进地学习偏好，但可能过拟合或丧失语言质量 |
| β 很大（如 0.5） | 强化 KL 约束，策略模型被"拴"在参考模型附近 | 更保守，偏好学习不够充分 |

**经验值**：β = 0.1 是最常见的默认值。训练时一般从 0.1 开始，根据效果调整。

类比：
> β 就像弹簧的弹性系数——β 大，弹簧紧，模型不容易偏离；β 小，弹簧松，模型可以自由探索偏好空间。

---

## 四、DPO 的训练流程

### 4.1 完整流程

训练前准备同一问题下的 chosen 和 rejected 两条回答。标注、去重和划分可参考[[偏好数据构建]]。

```
┌──────────────────────────────────────────────────────┐
│  1. 准备 SFT 模型 (π_ref) 和偏好数据                    │
│     每条数据：(prompt, chosen 回答, rejected 回答)      │
│                        ↓                             │
│  2. 深拷贝 SFT 模型 → 策略模型 (π_θ)，开始训练          │
│     π_ref 冻结，π_θ 可训练                             │
│                        ↓                             │
│  3. 对每个样本，分别计算：                               │
│     - π_θ 对 chosen 的 log 概率                       │
│     - π_θ 对 rejected 的 log 概率                     │
│     - π_ref 对 chosen 的 log 概率（无梯度）             │
│     - π_ref 对 rejected 的 log 概率（无梯度）           │
│                        ↓                             │
│  4. 计算 DPO Loss，反向传播，更新 π_θ                   │
│                        ↓                             │
│  5. 重复 3-4 直到收敛                                  │
└──────────────────────────────────────────────────────┘
```

### 4.2 关键实现细节

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_logps_chosen, policy_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """
    DPO 损失函数

    所有输入都是标量张量（batch 内每个样本的序列级 log 概率）
    """
    # 隐式奖励
    chosen_rewards = beta * (policy_logps_chosen - ref_logps_chosen)
    rejected_rewards = beta * (policy_logps_rejected - ref_logps_rejected)

    # DPO loss：让 chosen 的隐式奖励 > rejected 的隐式奖励
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


def compute_sequence_log_prob(model, input_ids, attention_mask):
    """
    计算模型对完整序列的 log 概率（只在 response 部分计算）

    实践中的关键：log 概率只在 chosen/rejected 的 token 上计算，
    prompt 部分不参与。这是因为 DPO 的偏好信号来自回答，不是 prompt。
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    # 用 attention_mask 只对 response 部分求和
    response_mask = attention_mask[:, 1:]  # 对齐 shift
    masked_log_probs = token_log_probs * response_mask
    return masked_log_probs.sum(dim=-1)
```

### 4.3 TRL 库的简化写法

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-sft-model")
tokenizer = AutoTokenizer.from_pretrained("your-sft-model")

training_args = DPOConfig(
    output_dir="./dpo-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,          # DPO 的 lr 通常比 SFT 低很多
    beta=0.1,                    # DPO 的 β
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=pref_dataset,   # 需包含 chosen/rejected 列
    tokenizer=tokenizer,
)

trainer.train()
```

`★ Insight ─────────────────────────────────────`
**为什么 DPO 的学习率要比 SFT 低很多？** SFT 的 lr 通常在 1e-5 ~ 5e-5，而 DPO 常用 1e-7 ~ 5e-7，低了 1-2 个数量级。原因是 DPO 的优化目标更"微妙"——它不是学新知识，而是微调已有的概率分布让好回答的概率相对更高。过大的 lr 会导致模型"矫枉过正"，语言能力崩塌。
`─────────────────────────────────────────────────`

---

## 五、常见训练陷阱

### 5.1 隐式奖励的长度偏差

DPO 的 log 概率比存在一个问题：**较长的回答天然有更负的 log 概率**（每个 token 的概率 < 1，连乘后越来越小）。

```
短回答："是的。"                    → log P ≈ -5
长回答："是的，这是一个很好的问题…"  → log P ≈ -200
```

这意味着长回答的隐式奖励会被系统性地压低，即使它质量更好。

**解决方案：长度归一化**

$$r_{\text{norm}}(x, y) = \beta \cdot \frac{1}{|y|} \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

### 5.2 胜率坍缩（Win Rate Collapse）

模型可能很快学会"总是生成 chosen 风格的回答"，导致训练集上胜率 100%，但：
- 生成的回答变得单一
- 在分布外的 prompt 上表现退化

**缓解方法**：
- 降低学习率
- 早期停止（early stopping）
- 增大 β 值，加强 KL 约束

### 5.3 参考模型的质量瓶颈

DPO 的参考模型是 SFT 模型。如果 SFT 模型的质量不够好，DPO 能学到的东西就很有限——因为 DPO 只能调整"策略模型相对于参考模型的偏好排序"，不能突破参考模型的能力上限。

**实践建议**：花时间把 SFT 做好，比调 DPO 的超参回报更高。

---

## 六、DPO 的局限性

| 局限 | 说明 |
|------|------|
| **无法处理非对比偏好** | DPO 需要 chosen/rejected 的对比对，无法直接利用"单条好回答"的数据 |
| **离线学习** | DPO 用的是静态偏好数据集，不像 PPO 那样可以在线采样和探索 |
| **超越 SFT 能力边界难** | DPO 更像是"重新排序"已有的能力，而非学新东西 |
| **对数据质量敏感** | 如果偏好标注不一致或有噪声，DPO 容易过拟合到错误信号 |

---

## 七、DPO 的变种

### 7.1 SimPO（Simple Preference Optimization）

SimPO 去掉了参考模型，改用**序列平均 log 概率**作为隐式奖励：

$$r_{\text{SimPO}}(x, y) = \frac{1}{|y|} \sum_t \log \pi_\theta(y_t | x, y_{<t})$$

| 对比 | DPO | SimPO |
|------|-----|-------|
| 隐式奖励 | $\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ | 序列平均 log 概率 |
| 需要参考模型 | ✅ | ❌ |
| 显存需求 | 2 倍 SFT | 1 倍 SFT |
| 长度偏差 | 有（需要额外处理） | 内置了长度归一化 |

### 7.2 KTO（Kahneman-Tversky Optimization）

KTO 不需要对比对，只需要知道每条回答"好"还是"不好"：

| 对比 | DPO | KTO |
|------|-----|-----|
| 数据需求 | (prompt, chosen, rejected) 三元组 | (prompt, 回答, 好/坏标签) 二元组 |
| 数据利用效率 | 每条数据需要两个回答 | 每条数据只需一个回答 |
| 适用场景 | 有对比标注 | 只有单条标注（如点赞/点踩） |

### 7.3 IPO（Identity Preference Optimization）

IPO 改变了 DPO 的目标函数，直接优化偏好的"差异"而非 Bradley-Terry 概率：

$$\mathcal{L}_{\text{IPO}} = \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta} \right)^2$$

IPO 的优势是更不容易过拟合，适合偏好噪声较大的场景。

### 7.4 ORPO（Odds Ratio Preference Optimization）

ORPO 连参考模型都不需要，用**赔率比（odds ratio）**替代 log 概率比：

$$\mathcal{L}_{\text{ORPO}} = -\log \sigma \left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right)$$

其中 $\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}$。

ORPO 可以**在 SFT 阶段直接加入偏好优化**，将两步合并为一步。

### 7.5 变种总览

```
                    DPO
                   ╱   ╲
            需要 ref    去掉 ref
            ╱               ╲
       IPO / KTO          SimPO / ORPO
       (改 loss)          (改奖励定义)
```

---

## 八、DPO vs 各方案选择指南

| 场景             | 推荐方案       | 原因            |
| -------------- | ---------- | ------------- |
| 显存充足，追求最佳效果    | PPO (RLHF) | 上限最高，但工程复杂    |
| 显存有限，有对比偏好数据   | DPO        | 简单高效，效果接近 PPO |
| 只有点赞/点踩，无对比对   | KTO        | 能利用非对比数据      |
| 想 SFT + 对齐一步完成 | ORPO       | 省去单独的对齐阶段     |
| 偏好数据噪声大        | IPO        | 更鲁棒，不容易过拟合    |
| 想极致省显存         | SimPO      | 不需要参考模型       |

---

## 九、常见问题

### Q1：DPO 训练时参考模型可以和策略模型用同一个吗？

可以，但效果会变差。标准做法是深拷贝 SFT 模型，一份冻结做 $\pi_{\text{ref}}$，一份可训练做 $\pi_\theta$。如果用同一个模型，训练过程中参考分布也在变，会导致目标不稳定。

### Q2：DPO 需要多少偏好数据？

经验数据：
- 最少：几百条高质量对比对就能看到效果
- 一般：1k-10k 条
- 充足：10k-100k 条

数据质量远比数量重要——1000 条一致性强的标注胜过 10000 条噪声大的标注。

### Q3：DPO 微调后的模型可以再做 DPO 吗？

可以，这叫**迭代 DPO**（Iterative DPO）：
1. 用当前模型生成新的回答
2. 人类标注新的偏好对
3. 用新的偏好数据继续 DPO

迭代 DPO 在一定程度上弥补了 DPO"离线学习"的局限。

### Q4：DPO 能和 LoRA/QLoRA 结合吗？

完全可以，而且是主流做法：

```python
# QLoRA + DPO：4-bit 量化 + LoRA 适配器 + DPO 对齐
from peft import get_peft_model, LoraConfig
from trl import DPOTrainer, DPOConfig

model = AutoModelForCausalLM.from_pretrained(
    "base-model", load_in_4bit=True, ...
)
lora_config = LoraConfig(r=16, target_modules=[...])
model = get_peft_model(model, lora_config)

# DPO Trainer 会自动处理：只有 LoRA 参数参与梯度计算
trainer = DPOTrainer(model=model, args=dpo_config, ...)
```

这是目前开源社区最流行的组合：**QLoRA + DPO = 在单张消费级显卡上完成对齐训练**。

---

## 十、总结

**DPO 的本质**：把[[RLHF]]中的"强化学习问题"转化为"监督学习问题"，通过数学推导消除了对显式奖励模型的需求。

```
RLHF:  数据 → Reward Model → PPO (4个模型，强化学习)
DPO:   数据 → 直接优化策略 (2个模型，监督学习)
```

**一句话记住 DPO**：策略模型和奖励模型是一体两面——知道了策略就能算出奖励，所以不需要单独训奖励模型。

**当前趋势**：DPO 及其变种（SimPO、KTO、ORPO）已成为开源对齐训练的主流方案。但如果资源充足且追求极致效果，PPO 仍有其价值。

---

## 参考

- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, 2023
- Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Feedback*, 2023 (IPO)
- Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, 2024
- Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*, 2024
- Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model*, 2024
