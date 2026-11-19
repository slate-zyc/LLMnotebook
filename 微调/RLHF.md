# RLHF：基于人类反馈的强化学习

> SFT 教会模型"怎么回答"，但回答得好不好、人类喜不喜欢，SFT 管不了。RLHF（Reinforcement Learning from Human Feedback）通过人类偏好信号来进一步优化模型，让模型的回答更符合人类期望。它是 ChatGPT 等对话模型惊艳效果背后的关键技术。
>
> 本文从"SFT 为什么不够"出发，讲解 RLHF 的三阶段流程、Reward Model、PPO 和 DPO 的原理，并整理面试高频问题。

---

## 一、为什么需要 RLHF？

### 1.1 SFT 的局限性

SFT 的目标是让模型学会"模仿"标注数据中的回答。但它有两个根本问题：

| 问题 | 说明 |
|------|------|
| **只能学"像不像"，不能学"好不好"** | SFT 的 loss 只看模型输出和标准答案的 token 是否匹配，无法衡量回答的"质量"、"有用性"、"安全性" |
| **数据覆盖有限** | SFT 数据量有限，模型没见过的场景下容易胡说八道（幻觉） |

举个例子：

```
问题：解释量子力学

SFT 标准答案：量子力学是研究微观粒子运动规律的物理学分支...

模型输出 A（token 匹配度高但胡说）：量子力学是一种利用量子效应进行计算的力学分支...
模型输出 B（token 匹配度低但更好）：简单说，微观世界里的粒子行为和我们日常经验完全不同...
```

SFT 的 loss 会认为 A 比 B 好（因为 A 更像标准答案），但实际上 B 的表达更清晰、更适合普通人理解。**SFT 无法区分"像"和"好"。**

### 1.2 RLHF 怎么解决？

核心思路：**用人类的"偏好"来替代 token 级别的精确匹配。**

> 不再要求模型输出和标准答案一模一样，而是让模型学会"人类更喜欢哪种回答"。

```
SFT：  你的回答和标准答案越像 → loss 越低
RLHF：你的回答人类越喜欢 → reward 越高
```

---

## 二、RLHF 的三阶段流程

RLHF 的完整流程分为三个阶段：

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ 阶段1: SFT   │ ──→ │ 阶段2: 训练奖励模型  │ ──→ │ 阶段3: PPO优化 │
│ 教模型怎么回答 │     │ 学会判断回答好坏     │     │ 用奖励引导模型  │
└─────────────┘     └──────────────────┘     └─────────────┘
```

### 阶段 1：SFT（监督微调）

这一步你已经很熟悉了——用高质量的"指令-回答"对训练模型，让它学会基本的对话能力。

- 产出：一个能回答问题但质量参差不齐的 SFT 模型
- 作用：为后续阶段提供一个"起点"，避免从零开始

### 阶段 2：训练奖励模型（Reward Model）

收集人类对模型不同回答的**偏好排序**，训练一个打分模型。

### 阶段 3：PPO 优化

用奖励模型作为"裁判"，通过强化学习让 SFT 模型学会生成更高分的回答。

---

## 三、奖励模型（Reward Model）

### 3.1 核心思想

奖励模型（RM）的本质是一个**打分器**：输入"问题 + 回答"，输出一个标量分数，表示这个回答有多好。

```
RM(问题, 回答) → 分数（越高越好）
```

### 3.2 数据收集过程

候选如何比较、平局如何处理，以及如何划分训练和验证数据，见[[偏好数据构建]]。

1. 给 SFT 模型同一个问题，让它生成 **多个回答**（如 4 个）
2. 人类标注员对这些回答进行 **排序**（哪个最好，哪个最差）
3. 排序结果转化为**对比对**（pairwise comparison）

**示例：**

```
问题：如何提高睡眠质量？

回答 A：每天固定时间睡觉和起床，保持规律作息。        ← 人类排序：第1
回答 B：睡觉前不要看手机，保持卧室黑暗安静。            ← 人类排序：第2
回答 C：你可以试试吃安眠药，效果很好。                  ← 人类排序：第3
回答 D：提高睡眠质量的方法有很多。                      ← 人类排序：第4

对比对：A > B, A > C, A > D, B > C, B > D, C > D
```

### 3.3 Reward Model 的训练

RM 通常基于 SFT 模型初始化（去掉最后一层的 token 预测头，换成一个标量输出头），然后用 **Bradley-Terry 模型 + 交叉熵损失** 训练：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]$$

其中：

| 符号 | 含义 |
|------|------|
| $x$ | 输入问题（prompt） |
| $y_w$ | 人类更偏好的回答（winner） |
| $y_l$ | 人类不太喜欢的回答（loser） |
| $r_\theta(x, y)$ | 奖励模型对"问题 $x$ + 回答 $y$"的打分 |
| $\sigma$ | Sigmoid 函数，将差值映射到 $[0, 1]$ |

**直觉理解**：让"好回答的分数"比"差回答的分数"高，差距越大，loss 越低。

### 3.4 Reward Model 的架构

```
┌───────────────────────────┐
│      SFT 模型（预训练）      │  ← 权重初始化来源
│  ┌─────────────────────┐  │
│  │  Transformer Layers  │  │  ← 保留所有层
│  └──────────┬──────────┘  │
│             ▼              │
│  ┌─────────────────────┐  │
│  │  Linear(d_model → 1) │  │  ← 替换原来的 token 预测头
│  └──────────┬──────────┘  │
│             ▼              │
│         标量分数 r          │  ← 输出：这个回答有多好
└───────────────────────────┘
```

关键点：RM 和 SFT 模型共享 Transformer 结构，但输出从"每个 token 的概率分布"变成了"一个标量分数"。

---

## 四、PPO（近端策略优化）

### 4.1 什么是 PPO？

PPO（Proximal Policy Optimization）是 RLHF 的核心优化算法。它是一种**策略梯度**方法，通过奖励模型的分数来引导模型生成更好的回答。

先理解几个关键概念：

| 概念 | 含义 | 类比 |
|------|------|------|
| **策略（Policy）** | 模型生成回答的方式，即模型本身 $\pi_\theta$ | 学生的答题方法 |
| **奖励（Reward）** | 奖励模型对回答的打分 $r_\theta(x, y)$ | 老师的打分 |
| **目标** | 最大化生成回答的期望奖励 | 让学生学会得高分 |

### 4.2 PPO 的目标函数

$$\mathcal{L}_{PPO} = \mathbb{E}_{x, y \sim \pi_{\theta}} \left[ r_\theta(x, y) - \beta \cdot \text{KL}\left( \pi_\theta(y|x) \| \pi_{\text{SFT}}(y|x) \right) \right]$$

这个公式有两部分：

| 部分 | 含义 | 作用 |
|------|------|------|
| $r_\theta(x, y)$ | 奖励模型的分数 | 鼓励模型生成高分回答 |
| $-\beta \cdot \text{KL}(\pi_\theta \| \pi_{SFT})$ | KL 惩罚项 | 防止模型偏离 SFT 模型太远 |

### 4.3 为什么需要 KL 惩罚？

这是 RLHF 中最容易被忽略但非常关键的一点。

**没有 KL 约束会发生什么？**

模型会为了拿高分而"钻漏洞"：
- 生成奖励模型喜欢但人类不喜欢的"对抗样本"
- 退化成重复输出某些高分短语
- 丧失语言流畅性和多样性

**KL 惩罚的作用**：让优化后的模型不要离 SFT 模型太远，保证基本的语言能力不退化。

类比：
> 学生（模型）为了拿高分（奖励），开始写套路作文（钻漏洞）。老师（KL 惩罚）说："你可以改进，但不能变成另一个人。"

### 4.4 PPO 的完整流程

```
┌──────────────────────────────────────────────────────────┐
│                    PPO 训练循环                            │
│                                                          │
│  1. 采样：策略模型 π_θ 生成回答 y                           │
│                 ↓                                        │
│  2. 打分：奖励模型 r_θ 给 (x, y) 打分                      │
│                 ↓                                        │
│  3. 约束：计算 KL(π_θ || π_SFT)，防止偏离太远              │
│                 ↓                                        │
│  4. 优势估计：计算这个回答比"平均"好多少                      │
│                 ↓                                        │
│  5. 更新：用 PPO 算法更新策略模型 π_θ                       │
│                 ↓                                        │
│  重复 1-5，直到模型学会生成高分回答                          │
└──────────────────────────────────────────────────────────┘
```

### 4.5 四个模型的角色

PPO 训练时需要同时维护四个模型，这也是它显存开销巨大的原因：

| 模型 | 角色 | 是否训练 |
|------|------|---------|
| **策略模型（Policy）** | 要优化的模型，生成回答 | ✅ 训练中 |
| **参考模型（Reference）** | SFT 模型的冻结副本，计算 KL 惩罚 | ❌ 冻结 |
| **奖励模型（Reward）** | 给回答打分 | ❌ 冻结（阶段2已训练好） |
| **价值模型（Value）** | 估计当前状态的价值（用于优势函数） | ✅ 训练中 |

> 这四个模型都需要加载到显存中，所以 RLHF 的显存需求约为 SFT 的 **4 倍**。这也是为什么后来出现了 DPO 等替代方案。

---

## 五、DPO（直接偏好优化）

> DPO 内容详见 [[DPO]]，此处为概述。

### 5.1 DPO 的核心思想

PPO 太复杂了——需要四个模型、奖励模型训练、强化学习循环。DPO（Direct Preference Optimization）提出：**能不能跳过奖励模型，直接用偏好数据优化策略模型？**

DPO 的核心洞察：**奖励模型的最优解可以用策略模型来表示。**

具体来说，DPO 证明了以下等价关系：

> 在 Bradley-Terry 模型下，最大化奖励模型的偏好概率，等价于最小化一个只涉及策略模型和参考模型的 loss。

### 5.2 DPO 的损失函数

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

直觉理解：

| 行为 | 效果 |
|------|------|
| 策略模型对好回答的概率比参考模型高 | 奖励，降低 loss |
| 策略模型对差回答的概率比参考模型高 | 惩罚，增加 loss |

### 5.3 DPO vs PPO

| 维度     | PPO           | DPO          |     |
| ------ | ------------- | ------------ | --- |
| 需要奖励模型 | ✅ 是           | ❌ 否          |     |
| 模型数量   | 4 个           | 2 个（策略 + 参考） |     |
| 显存需求   | 极高（约 4 倍 SFT） | 低（约 2 倍 SFT） |     |
| 训练稳定性  | 不稳定（强化学习）     | 稳定（监督学习）     |     |
| 代码复杂度  | 高             | 低            |     |
| 效果     | 上限可能更高        | 略逊于 PPO（有争议） |     |
| 适用场景   | 资源充足、追求极致效果   | 资源有限、快速迭代    |     |

**DPO 的本质**：把"强化学习问题"转化成了"监督学习问题"，用偏好对比
对直接训练模型，大幅简化了流程。

### 5.4 DPO 的训练流程

```
┌─────────────────────────────────┐
│  1. 准备偏好数据                   │
│     (prompt, 好回答, 差回答)        │
│              ↓                   │
│  2. 策略模型 π_θ 计算两个回答的概率  │
│              ↓                   │
│  3. 参考模型 π_ref 计算两个回答的概率 │
│              ↓                   │
│  4. 计算 DPO Loss                │
│              ↓                   │
│  5. 反向传播更新 π_θ               │
└─────────────────────────────────┘
```

---

## 六、代码实现

### 6.1 Reward Model 实现

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class RewardModel(nn.Module):
    """
    奖励模型：基于 SFT 模型，将 token 预测头替换为标量打分头
    """
    def __init__(self, sft_model_path):
        super().__init__()
        # 加载 SFT 模型作为 backbone
        self.backbone = AutoModelForCausalLM.from_pretrained(sft_model_path)
        # 获取隐藏层维度
        hidden_size = self.backbone.config.hidden_size
        # 替换输出头：从 vocab_size 个 token 概率 → 1 个标量分数
        self.score_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        # 获取 Transformer 的隐藏状态
        outputs = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # 取最后一个 token 的隐藏状态（对应回答的结尾）
        last_hidden = outputs.hidden_states[-1]
        # 找到每个序列最后一个非 padding token 的位置
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden = last_hidden[
            torch.arange(last_hidden.size(0)), sequence_lengths
        ]
        # 输出标量分数
        return self.score_head(last_token_hidden).squeeze(-1)

# 训练 Reward Model 的 Loss
def reward_model_loss(chosen_rewards, rejected_rewards):
    """
    Bradley-Terry 模型损失
    chosen_rewards: 人类更偏好的回答的分数
    rejected_rewards: 人类不太喜欢的回答的分数
    """
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    return loss
```

### 6.2 DPO 实现

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

def dpo_loss(policy_model, reference_model, batch, beta=0.1):
    """
    DPO 损失函数

    Args:
        policy_model: 策略模型（正在训练的模型）
        reference_model: 参考模型（冻结的 SFT 模型）
        batch: 包含 prompt、chosen（好回答）、rejected（差回答）
        beta: 温度系数，控制偏离参考模型的程度
    """
    # 1. 策略模型计算两个回答的 log 概率
    policy_chosen_logps = compute_log_prob(policy_model, batch["chosen"])
    policy_rejected_logps = compute_log_prob(policy_model, batch["rejected"])

    # 2. 参考模型计算两个回答的 log 概率（不计算梯度）
    with torch.no_grad():
        ref_chosen_logps = compute_log_prob(reference_model, batch["chosen"])
        ref_rejected_logps = compute_log_prob(reference_model, batch["rejected"])

    # 3. 计算隐式奖励：策略模型概率相对于参考模型的变化
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # 4. DPO Loss：让好回答的隐式奖励高于差回答
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


def compute_log_prob(model, input_ids):
    """计算模型对给定序列的 log 概率"""
    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :]   # 去掉最后一个位置
    targets = input_ids[:, 1:]           # 去掉第一个位置（shift）
    log_probs = F.log_softmax(logits, dim=-1)
    # 取每个目标 token 的 log 概率并求和
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=-1)
```

---

## 七、RLHF 的实际效果与局限

### 7.1 效果对比

```
基础模型（Pretrain） < SFT 模型 < RLHF 模型
```

RLHF 带来的改进主要体现在：
- **有用性**：回答更有针对性，不再答非所问
- **安全性**：减少有害、偏见、违法内容的输出
- **诚实性**：减少幻觉，不确定时会说"我不确定"
- **格式**：输出格式更规范，适合下游使用

### 7.2 局限性

1. **人类偏好主观**：不同标注员的偏好不一致，导致奖励模型有噪声
2. **奖励欺骗（Reward Hacking）**：模型可能学会讨好奖励模型而不是真正变好
3. **分布偏移**：PPO 优化后的模型分布可能与奖励模型训练时的分布差距过大
4. **成本高**：需要大量人类标注 + 四个模型的显存开销

---

## 八、面试高频问题

### Q1：RLHF 的核心思想是什么？为什么 SFT 不够？

> RLHF 的核心是用人类偏好信号替代 token 级别的精确匹配来优化模型。SFT 只能学"像不像"标准答案，无法衡量回答的"质量"、"安全性"等主观维度。RLHF 通过训练一个奖励模型来模拟人类偏好，再用强化学习优化生成策略，让模型学会"人类喜欢什么"而不仅仅是"数据里有什么"。

### Q2：RLHF 的三个阶段分别是什么？

> 1. **SFT 阶段**：用高质量指令-回答对微调模型，获得基本对话能力
> 2. **Reward Model 阶段**：收集人类对多个回答的排序，训练一个打分模型
> 3. **PPO 阶段**：用奖励模型的分数作为奖励信号，通过 PPO 算法优化策略模型，同时用 KL 惩罚防止偏离 SFT 模型太远

### Q3：PPO 训练时为什么需要 KL 惩罚？

> 防止模型为了追求高分而"走极端"。没有 KL 约束时，模型可能：(1) 生成奖励模型喜欢但人类不喜欢的对抗样本；(2) 退化为重复输出某些高分短语；(3) 丧失语言流畅性和多样性。KL 惩罚确保优化后的模型不要离 SFT 模型太远，保持基本的语言能力。

### Q4：PPO 训练需要哪四个模型？为什么显存开销大？

> 1. **策略模型（Policy）**：要优化的模型，正在训练
> 2. **参考模型（Reference）**：SFT 模型的冻结副本，计算 KL 惩罚
> 3. **奖励模型（Reward）**：阶段 2 训练好的打分器，冻结
> 4. **价值模型（Value）**：估计当前状态的价值，用于优势函数估计，正在训练
>
> 四个模型都需要加载到显存中，所以 RLHF 的显存需求约为 SFT 的 4 倍。这也是 DPO 等替代方案出现的原因。

### Q5：DPO 和 PPO 的区别是什么？DPO 的核心优势是什么？

> PPO 需要先训练奖励模型，再用强化学习优化，流程复杂、显存需求大、训练不稳定。DPO 的核心洞察是：可以把奖励模型隐式地用策略模型表示，从而跳过奖励模型训练，直接用偏好数据（好回答 vs 差回答）通过监督学习优化模型。
>
> 优势：只需 2 个模型（策略 + 参考）、显存需求减半、训练稳定、代码简单。

### Q6：Reward Model 是怎么训练的？

> 基于 SFT 模型初始化，将输出头从 token 概率预测改为标量打分。训练数据是人类对多个回答的排序，转化为 pairwise comparison（好回答 vs 差回答的对比对）。损失函数是 Bradley-Terry 模型：$L = -E[\log \sigma(r_{good} - r_{bad})]$，即让好回答的分数高于差回答。

### Q7：什么是 Reward Hacking？怎么缓解？

> Reward Hacking 指模型学会了"讨好"奖励模型而不是真正变好。例如生成一些看起来分数很高但实际无意义的回答。缓解方法：(1) KL 惩罚防止模型偏离太远；(2) 定期更新奖励模型；(3) 使用 ensemble 的多个奖励模型取平均；(4) 人工定期评估校准。

### Q8：RLHF、DPO、RLAIF 分别是什么？有什么关系？

> - **RLHF**：用人类偏好训练奖励模型，再用 PPO 优化。最经典但最复杂。
> - **DPO**：跳过奖励模型，直接用偏好数据通过监督学习优化。是 RLHF 的简化版。
> - **RLAIF**（Reinforcement Learning from AI Feedback）：用 AI（如 GPT-4）代替人类来提供偏好标注，降低标注成本。流程与 RLHF 相同，只是标注来源不同。

---

## 九、总结

**大模型对齐的完整流程**：

```
Pretrain（预训练） → SFT（监督微调） → RLHF/DPO（偏好对齐）
  学会说话           学会听话            学会说好话
```

**三个关键概念一句话总结**：

1. **Reward Model**：用人类偏好排序训练一个打分器，判断回答好坏
2. **PPO**：用奖励模型的分数引导模型生成更好的回答，KL 惩罚防止走极端
3. **DPO**：跳过奖励模型，直接用偏好数据训练模型，简化 RLHF 流程

**选择建议**：资源充足追求效果用 PPO，资源有限快速迭代用 DPO。目前工业界 DPO 及其变种（如 SimPO、KTO）因简洁高效已逐渐成为主流。
