# GRPO：组相对策略优化

> PPO 是 RLHF 的经典算法，但需要同时维护四个模型，显存开销巨大。GRPO（Group Relative Policy Optimization）由 DeepSeek 团队在 2024 年提出，核心创新是：**用"组内相对排名"替代"绝对价值估计"，从而去掉价值模型**。它在数学推理等任务上表现出色，是 DeepSeek-R1 等模型的核心训练算法。
>
> 本文从 PPO 的痛点出发，讲解 GRPO 的核心思想、优势计算方式、代码实现，并与 PPO/DPO 对比。

---

## 一、PPO 的问题：为什么要提出 GRPO？

### 1.1 PPO 的痛点

回顾 PPO 训练需要四个模型：

| 模型 | 角色 |
|------|------|
| 策略模型（Policy） | 生成回答 |
| 参考模型（Reference） | 计算 KL 惩罚 |
| 奖励模型（Reward） | 给回答打分 |
| 价值模型（Value） | 估计状态价值，用于优势函数 |

**价值模型是最昂贵的部分**：
- 和策略模型一样大（通常同等参数量）
- 也需要训练更新
- 四个 7B 模型 = 需要约 **120GB+ 显存**

### 1.2 GRPO 的核心思路

> 既然价值模型的作用是估计"某个回答比平均好多少"，那能不能**直接用组内样本的统计量**来代替这个估计？

```
PPO：  需要一个神经网络来估计"平均表现是多少" → 价值模型
GRPO：直接生成一组回答，用这组回答的平均分当基准 → 不需要价值模型
```

`★ Insight ─────────────────────────────────────`
GRPO 的直觉类比：PPO 像考试后由老师（价值模型）告诉你"全班平均分是多少"；GRPO 像你和其他几个同学同时答题，直接看你们这一组的平均分来判断自己是好是差。省掉了"老师"这个角色。
`─────────────────────────────────────────────────`

---

## 二、GRPO 的算法流程

### 2.1 完整流程

```
对于每个 prompt（问题）x：

  1. 采样：策略模型 π_θ 对 x 生成 G 个回答 {y₁, y₂, ..., yG}
              ↓
  2. 打分：奖励模型对每个回答打分 {r₁, r₂, ..., rG}
              ↓
  3. 归一化：计算组内相对优势 {A₁, A₂, ..., AG}
              ↓
  4. 更新：用 GRPO Loss 更新策略模型
              ↓
  重复以上步骤
```

### 2.2 关键步骤：组内相对优势

这是 GRPO 最核心的创新。对于一个 prompt $x$，生成 $G$ 个回答后，第 $i$ 个回答的优势为：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, r_2, ..., r_G\})}{\text{std}(\{r_1, r_2, ..., r_G\})}$$

**直觉理解**：

| 情况 | 优势值 | 含义 |
|------|--------|------|
| 该回答是组内最好的 | 正数（如 +1.5） | "你比平均好很多，继续保持" |
| 该回答是组内中等的 | 接近 0 | "你和平均差不多" |
| 该回答是组内最差的 | 负数（如 -1.2） | "你比平均差，需要改进" |

**为什么除以标准差？**

- 不同问题的奖励分数范围可能差异很大（简单题大家都是高分，难题大家都是低分）
- 除以标准差后，优势值始终在一个相对稳定的范围内
- 这叫做**基线归一化（baseline normalization）**

### 2.3 GRPO 的目标函数

$$\mathcal{L}_{GRPO} = -\mathbb{E}_{x, \{y_i\} \sim \pi_\theta} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min\left( \frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)} \hat{A}_i, \text{clip}(\cdot) \hat{A}_i \right) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{ref}) \right) \right]$$

看起来很复杂，拆开来看：

| 部分 | 含义 | 与 PPO 的关系 |
|------|------|-------------|
| $\frac{\pi_\theta}{\pi_{old}} \hat{A}_i$ | 策略梯度项：用相对优势引导更新 | 类似 PPO 的策略梯度 |
| $\text{clip}(\cdot)$ | PPO 的裁剪机制，限制更新幅度 | 与 PPO 完全相同 |
| $-\beta \cdot \text{KL}$ | KL 惩罚，防止偏离参考模型太远 | 与 PPO 相同 |
| $\hat{A}_i$ 由组内统计量计算 | **核心区别**：不需要价值模型 | PPO 用价值模型计算 |

---

## 三、GRPO vs PPO vs DPO

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| **模型数量** | 4 个 | 2 个 | 3 个 |
| **需要奖励模型** | ✅ 是 | ❌ 否 | ✅ 是 |
| **需要价值模型** | ✅ 是 | ❌ 否 | ❌ 否 |
| **优势计算** | GAE（价值模型估计） | 不需要 | 组内统计量归一化 |
| **数据利用方式** | 每个 prompt 生成 1 个回答 | 需要预标注好/坏对比对 | 每个 prompt 生成 G 个回答 |
| **显存需求** | 极高（4x） | 低（2x） | 中等（3x） |
| **训练稳定性** | 不稳定 | 稳定 | 较稳定 |
| **适合场景** | 资源充足 | 快速迭代 | 中等资源，推理类任务 |

### 需要的模型对比

```
PPO:   [策略] [参考] [奖励] [价值]  ← 4 个模型
GRPO:  [策略] [参考] [奖励]         ← 3 个模型（去掉了价值模型）
DPO:   [策略] [参考]               ← 2 个模型（去掉了奖励模型和价值模型）
```

---

## 四、GRPO 在 DeepSeek-R1 中的应用

### 4.1 DeepSeek-R1 的训练流程

DeepSeek-R1 的训练不依赖人类标注数据，而是用 **规则奖励**（如数学题的答案正确性、代码的测试通过率）直接训练：

```
Base Model → SFT（少量冷启动数据）
    ↓
GRPO 训练（用规则奖励，不需要人类标注的 Reward Model）
    ↓
SFT（拒绝采样 + 混合数据）
    ↓
GRPO 训练（进一步对齐）
    ↓
DeepSeek-R1
```

### 4.2 规则奖励 vs 奖励模型

DeepSeek-R1 在 GRPO 阶段使用**规则奖励**而非神经网络奖励模型：

| 奖励类型 | 例子 | 优点 | 缺点 |
|---------|------|------|------|
| 神经网络 RM | 训练一个打分模型 | 灵活，能处理开放式回答 | 可能被 hack，需要额外训练 |
| **规则奖励** | 数学题答案对了 +1 分，错了 -1 分 | 简单可靠，不会被 hack | 只适合有标准答案的任务 |

`★ Insight ─────────────────────────────────────`
DeepSeek-R1 的突破在于：数学推理和代码任务有天然的"正确答案"，可以直接用规则（答案对不对、代码能不能通过测试）当奖励信号，完全不需要训练奖励模型。这让 GRPO 的三个模型进一步简化——实际只需要策略模型 + 参考模型 + 规则函数。
`─────────────────────────────────────────────────`

---

## 五、代码实现

### 5.1 GRPO 核心算法

```python
import torch
import torch.nn.functional as F

def grpo_loss(policy_model, reference_model, reward_fn, prompt_ids,
              G=8, beta=0.01, clip_eps=0.2):
    """
    GRPO 损失函数

    Args:
        policy_model: 策略模型（正在训练）
        reference_model: 参考模型（冻结）
        reward_fn: 奖励函数（可以是 RM 或规则函数）
        prompt_ids: 输入 prompt 的 token ids
        G: 每个 prompt 生成的回答数量
        beta: KL 惩罚系数
        clip_eps: PPO 裁剪范围
    """
    # 1. 采样：策略模型生成 G 个回答
    responses = []
    for _ in range(G):
        response = policy_model.generate(prompt_ids, max_new_tokens=512)
        responses.append(response)

    # 2. 打分：用奖励函数给每个回答打分
    rewards = torch.tensor([reward_fn(prompt_ids, r) for r in responses])

    # 3. 计算组内相对优势
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 4. 计算策略模型和参考模型对每个回答的 log 概率
    policy_logps = torch.tensor([
        compute_sequence_logprob(policy_model, prompt_ids, r) for r in responses
    ])
    ref_logps = torch.tensor([
        compute_sequence_logprob(reference_model, prompt_ids, r) for r in responses
    ])

    # 5. 策略梯度损失（带 PPO 裁剪）
    old_logps = policy_logps.detach()  # 当前策略作为"旧策略"
    ratio = torch.exp(policy_logps - old_logps)

    surr1 = ratio * advantages                     # 未裁剪
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages  # 裁剪
    policy_loss = -torch.min(surr1, surr2).mean()

    # 6. KL 惩罚
    kl_penalty = (policy_logps - ref_logps).mean()

    # 7. 总损失
    loss = policy_loss + beta * kl_penalty
    return loss


def compute_sequence_logprob(model, prompt_ids, response_ids):
    """计算模型对整个回答序列的 log 概率"""
    full_ids = torch.cat([prompt_ids, response_ids])
    logits = model(full_ids.unsqueeze(0)).logits[0]
    # 只取回答部分的 logits 和 targets
    response_logits = logits[len(prompt_ids) - 1:-1]
    response_targets = response_ids
    log_probs = F.log_softmax(response_logits, dim=-1)
    token_logps = log_probs.gather(1, response_targets.unsqueeze(1)).squeeze(1)
    return token_logps.sum()
```

### 5.2 规则奖励函数示例（数学题）

```python
def math_rule_reward(prompt, response):
    """
    数学题的规则奖励函数
    - 答案完全正确：+1.0
    - 答案错误：-1.0
    - 格式奖励（有推理过程）：+0.1
    """
    # 提取模型给出的最终答案
    predicted_answer = extract_answer(response)
    ground_truth = get_ground_truth(prompt)

    reward = 0.0

    # 答案正确性
    if check_answer(predicted_answer, ground_truth):
        reward += 1.0
    else:
        reward -= 1.0

    # 格式奖励：鼓励模型输出推理过程
    if has_thinking_process(response):
        reward += 0.1

    return reward
```

---

## 六、面试高频问题

### Q1：GRPO 的核心思想是什么？和 PPO 的主要区别？

> GRPO 的核心创新是用**组内相对排名**来计算优势值，替代 PPO 中价值模型的角色。对于每个 prompt，GRPO 生成 G 个回答，用奖励函数打分后，通过组内均值和标准差归一化得到相对优势。主要区别：PPO 需要 4 个模型（含价值模型），GRPO 只需要 3 个（去掉了价值模型），显存需求降低约 25%。

### Q2：GRPO 为什么不需要价值模型？

> PPO 需要价值模型来估计"当前状态的期望回报"作为基线（baseline），再用实际回报减去基线得到优势。GRPO 的替代方案是：对同一个 prompt 生成多个回答，用这组回答的奖励均值作为基线，标准差做归一化。组内统计量天然起到了基线的作用，不需要额外的神经网络来估计。

### Q3：GRPO 中 G（组大小）怎么选？

> G 越大，组内统计量（均值、标准差）越稳定，优势估计越准确，但计算成本也越高。常用 G = 8 ~ 64。DeepSeek 在实践中通常使用 G = 16 或更大。如果奖励函数的方差很大（如开放式回答），需要更大的 G 来获得稳定的估计。

### Q4：DeepSeek-R1 为什么选择 GRPO 而不是 PPO 或 DPO？

> 1. **显存效率**：相比 PPO 去掉了价值模型，显存需求更低
> 2. **规则奖励适配**：R1 的核心场景是数学推理和代码，有明确的正确答案，可以用规则函数代替奖励模型。GRPO 配合规则奖励，连奖励模型的训练都省了
> 3. **训练稳定性**：组内相对排名比绝对奖励值更稳定，不容易被单个异常样本干扰
> 4. **DPO 的局限**：DPO 需要预先标注好/坏对比对，而 GRPO 可以在线采样、在线评估，更适合大规模自博弈训练

### Q5：GRPO 和 DPO 分别适合什么场景？

> - **DPO**：适合有现成偏好数据、追求简单快速迭代的场景。不需要奖励模型，训练最简单
> - **GRPO**：适合有明确奖励信号（规则奖励或训练好的 RM）、需要在线探索的场景。特别是数学推理、代码生成等有标准答案的任务

### Q6：GRPO 的 KL 惩罚和 PPO 的有什么区别？

> PPO 用参考模型的输出概率和策略模型逐 token 计算 KL 散度，通常在每个 token 位置都计算。GRPO 的 KL 惩罚计算方式类似，但在 DeepSeek-R1 的实现中，有时会省略显式的 KL 惩罚，改用其他约束手段（如限制生成长度、格式约束等），因为规则奖励本身已经很强地约束了输出。

---

## 七、总结

**GRPO 的核心思想一句话**：

> 对同一个问题生成多个回答，用组内平均分当基线、标准差当归一化因子，计算每个回答的相对优势，从而**去掉价值模型**，降低显存开销。

**三种对齐算法的选择**：

```
资源充足、追求极致 → PPO（4 个模型）
中等资源、有标准答案 → GRPO（3 个模型）
资源有限、快速迭代 → DPO（2 个模型）
```

**演进趋势**：

```
PPO（2017）→ DPO（2023）→ GRPO（2024）
  最强但最贵    简化 RL 为 SL    去掉价值模型
```

GRPO 是 DeepSeek-R1 这类推理模型的核心训练算法，理解它有助于理解当前大模型对齐技术的最新进展。
