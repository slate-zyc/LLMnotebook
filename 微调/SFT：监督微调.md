
---
## 一、什么是 SFT？
---

SFT 全称 Supervised Fine-Tuning（监督微调），是大语言模型对齐流程中，继预训练之后的关键一步。
它的核心逻辑是：在已经预训练好的基础模型上，用人工标注的「输入 - 输出」对进行有监督训练，让模型快速适配特定任务，对齐人类指令格式。

---
## 二、SFT 的本质
---
SFT 的本质，其实是完整延续了预训练阶段的核心逻辑 —— "预测下一个 token" 的语言建模目标：

预训练时，模型学习的是 "预测下一个 token" 的通用语言规律，其更像是在进行根据前文填空或者是句子续写的游戏。

SFT 阶段，开始真正教会模型"回答"问题，而不再是简单的续写。
举个例子：

预训练模型的输出（只会续写，不会回答）
输入："什么是人工智能"
输出：
"是一门研究和开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的新的技术科学。人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等..."
（这是典型的百科词条续写，因为它在预训练时见过无数次这样的文本格式，根本不知道 "这是一个需要回答的问题"）

经过 SFT 的模型输出（学会了怎么回答人类）
输入："请用大白话解释一下什么是人工智能？"
输出：
"简单来说，人工智能就是让机器学会像人一样思考和做事的技术。比如你现在正在聊天的这个机器人、手机上的人脸识别解锁、外卖软件给你推荐的餐厅、导航帮你规划的最快路线，背后都有人工智能在发挥作用。"

##  三、SFT 监督微调与 Pre-train 的区别

| 对比维度       | 预训练的训练文本                                          | SFT 的训练文本                                                      |
| ---------- | ------------------------------------------------- | -------------------------------------------------------------- |
| **文本来源**   | 互联网上能爬取到的所有公开文本<br><br>（网页、书籍、论文、代码、论坛、微博、新闻、小说等） | 人工标注 / 高质量蒸馏的**专属对话数据**<br><br>（没有任何公开来源，是大模型公司的核心资产）          |
| **文本格式**   | 无任何统一格式，杂乱无章<br><br>（可以是一句话、一段文章、一整本书、一行代码）       | 严格统一的 **"提问 + 回答" 配对格式 **<br><br>（所有样本都遵循：`人类说：XXX → 助手说：XXX`） |
| **文本内容**   | 什么内容都有，好坏不分<br><br>（既有科学知识，也有谣言、脏话、广告、垃圾信息）       | 只保留**高质量、有用、无害**的内容<br><br>（所有回答都经过人工审核，过滤掉错误、有害和低质信息）         |
| **文本意图**   | 没有任何明确意图<br><br>（只是客观存在的文字，不指向任何 "任务"）            | 每一条都有明确的**人类意图**<br><br>（提问、请求、命令、咨询等，都是人类希望模型完成的任务）           |
| **文本规模**   | 极其庞大<br><br>（万亿级 token，相当于几百万本书）                  | 极其微小<br><br>（千级到百万级样本，相当于几本书）                                  |
| **教会模型什么** | 教会模型 "**语言是什么样的**"<br><br>（学会语法、词汇、知识和世界规律）       | 教会模型 "**人类希望你怎么说话**"<br><br>（学会听懂指令、输出正确格式、符合人类偏好）             |

### 📌 核心一句话总结
> 预训练教会模型**说话**，SFT教会模型**听话**。

---

## 四、SFT 的数据准备

### 4.1 数据长什么样？

标准的 SFT 训练数据格式通常包含三个字段：instruction（指令）、input（可选输入）、output（期望输出）。

```json
{
  "instruction": "用三句话解释什么是量子力学",
  "input": "",
  "output": "量子力学是研究微观粒子运动规律的物理学分支。在微观世界里，粒子不再像日常生活中那样有确定的位置和速度，而是以概率的方式存在。这就好比抛硬币，落地前你不知道是正面还是反面，只能说出正反面的概率各是多少，量子力学就是用来计算这些概率的规则。"
}
```

带输入字段的例子：
```json
{
  "instruction": "将以下英文翻译成中文",
  "input": "The quick brown fox jumps over the lazy dog.",
  "output": "那只敏捷的棕色狐狸跳过了那只懒惰的狗。"
}
```

### 4.2 数据从哪来？

| 数据来源  | 说明                          | 优缺点            |
| ----- | --------------------------- | -------------- |
| 人工标注  | 真人根据指令编写高质量回答               | 质量最高，但成本极高，速度慢 |
| 大模型蒸馏 | 用 GPT-4 等强模型自动生成数据          | 成本低、速度快，质量次之   |
| 公开数据集 | Alpaca、Belle、ShareGPT 等开源数据 | 免费获取，但同质化严重    |

### 4.3 数据质量 > 数量

> "1000 条高质量数据 > 10 万条垃圾数据"

数据筛选的关键标准：
- **准确性**：回答内容必须正确无误
- **多样性**：覆盖不同类型的任务（问答、翻译、代码、写作等）
- **格式规范**：遵循统一的模板，便于模型学习
- **无害性**：过滤掉有害、偏见、违法内容

---

## 五、SFT 的技术方法

### 5.1 全量微调 vs 高效微调

| 方法    | 训练哪些参数      | 显存需求      | 适合场景        |
| ----- | ----------- | --------- | ----------- |
| 全量微调  | 更新模型所有参数    | 极高（80GB+） | 资源充足 / 深度定制 |
| LoRA  | 只训练新增的小适配器  | 低（16GB）   | 日常首选        |
| QLoRA | 量化模型 + LoRA | 极低（8GB）   | 单卡训练大模型     |

### 5.2 各方法详解

#### 全量微调
最传统的方式，把模型所有参数都拿出来训练。
- **优点**：效果最好，模型完全适配任务
- **缺点**：显存需求巨大，训练时间长，容易遗忘预训练知识（灾难性遗忘）
- **适用**：公司有充足 GPU 资源，需要深度定制模型

#### LoRA（Low-Rank Adaptation）
在模型旁边加一个"小插件"，只训练插件部分，原始模型参数完全冻结。
- **优点**：显存需求大幅降低，训练速度快，原始模型不受影响
- **缺点**：效果略低于全量微调
- **适用**：大多数场景的首选方案

#### QLoRA
在 LoRA 基础上，先把模型量化（压缩精度），再训练适配器。
- **优点**：显存需求极低，消费级显卡就能跑
- **缺点**：量化会带来一定精度损失
- **适用**：个人开发者、研究者在单卡上训练大模型

### 5.3 LoRA 核心思想（小白友好版）

> 就像给模型戴了一副"特制眼镜"，不需要改变模型本身，只加一个小插件就能让它学会新技能。

类比：假设你是一个精通各种知识的百科全书（预训练模型），现在要学习如何当客服。
- **全量微调** = 重新学习，把百科全书的内容全部改写
- **LoRA** = 在百科全书旁边放一本"客服手册"，回答客服问题时参考手册即可

---

## 六、SFT 的训练流程

### 6.1 完整流程（代码实现版）

```python
# ============================================================
# SFT 监督微调完整流程
# ============================================================

# ---------------------- 步骤1：准备数据 ----------------------
from datasets import load_dataset

# 加载训练数据（JSON格式，包含instruction、input、output字段）
dataset = load_dataset('json', data_files='sft_data.json')

# ---------------------- 步骤2：选择基座模型 ----------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载预训练好的基座模型和对应的分词器
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ---------------------- 步骤3：选择微调方法 ----------------------
from peft import LoraConfig, get_peft_model

# 配置LoRA参数（高效微调方法）
lora_config = LoraConfig(
    r=16,              # LoRA秩，越大效果越好但显存消耗越大
    lora_alpha=32,     # 缩放系数，通常设为r的2倍
    target_modules=["q_proj", "v_proj"],  # 要应用LoRA的层
    lora_dropout=0.05, # Dropout比例，防止过拟合
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)  # 将LoRA应用到模型上

# ---------------------- 步骤4：设置训练参数 ----------------------
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./sft_output",      # 模型保存路径
    num_train_epochs=3,             # 训练轮数：数据被学习几遍
    per_device_train_batch_size=8,  # 批次大小：一次看多少条数据
    gradient_accumulation_steps=4,  # 梯度累积：缓解显存不足
    learning_rate=2e-4,             # 学习率：参数更新步长
    lr_scheduler_type="cosine",     # 学习率调度：余弦衰减
    fp16=True,                      # 混合精度训练：节省显存
    logging_steps=10,               # 每10步打印一次日志
    save_strategy="epoch",          # 每个epoch保存一次模型
)

# ---------------------- 步骤5：开始训练 ----------------------
from trl import SFTTrainer

# 创建SFT训练器
trainer = SFTTrainer(
    model=model,                    # 要训练的模型
    args=training_args,             # 训练参数
    train_dataset=dataset["train"], # 训练数据集
    tokenizer=tokenizer,            # 分词器
    dataset_text_field="text",      # 数据集中文本字段名
    max_seq_length=2048,            # 最大序列长度
)

trainer.train()                     # 启动训练，观察Loss下降
trainer.save_model("./sft_model")   # 保存训练好的模型

# ---------------------- 步骤6：评估效果 ----------------------
# 加载测试数据，人工验证模型输出质量
# （可使用BLEU、ROUGE等自动指标 + 人工主观评估）

# ---------------------- 步骤7：部署使用 ----------------------
# 将训练好的模型部署为API服务或集成到应用中
```

### 6.2 关键训练参数解释

| 参数名                | 含义                 | 通俗解释              | 常见取值        |
| ------------------ | ------------------ | ----------------- | ----------- |
| Learning Rate（学习率） | 参数更新的步长            | 学得太快容易学歪，学得太慢浪费时间 | 1e-4 ~ 5e-5 |
| Epoch（训练轮数）        | 数据被学习的次数           | 一本书看几遍            | 1 ~ 5       |
| Batch Size（批次大小）   | 一次看多少条数据           | 一口吃多少饭            | 8 ~ 32      |
| Max Length（最大长度）   | 输入 + 输出的最大 token 数 | 一次最多能读多长的内容       | 512 ~ 4096  |
| Loss（损失值）          | 模型预测与正确答案的差距       | 学得好不好的指标          | 越低越好        |

---

## 七、SFT 的核心机制：Loss Mask

#### 核心代码实现：Loss Mask 的完整流程

```python
# ============================================================
# Loss Mask 的具体实现：为什么只对 assistant 计算 Loss
# ============================================================

import torch
from transformers import AutoTokenizer

# ---------- 1. 原始对话数据 ----------
# 训练样本：用户提问 + 助手回答
conversation = [
    {"role": "user", "content": "什么是人工智能？"},
    {"role": "assistant", "content": "人工智能是让机器学会像人一样思考和做事的技术。"}
]

# ---------- 2. 将对话拼接成模型输入格式 ----------
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# 使用 tokenizer 的 chat_template 将对话转为文本
text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,  # 先不进行 tokenization，只看文本格式
    add_generation_prompt=False
)

print("【拼接后的完整文本】")
print(text)
# 输出示例：
# <|im_start|>user
# 什么是人工智能？<|im_end|>
# <|im_start|>assistant
# 人工智能是让机器学会像人一样思考和做事的技术。<|im_end|>

# ---------- 3. Tokenize 得到 input_ids ----------
# 将文本转为模型能理解的 token ID 序列
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
input_ids = inputs["input_ids"][0]

print(f"\n【Input IDs 长度】: {len(input_ids)}")
# 示例：[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, ...]

# ---------- 4. 构造 Labels（核心：Loss Mask 的实现）----------
# 关键步骤：将 user 部分的 token 设置为 -100（IGNORE_INDEX）
# 这样训练时，这些位置就不会参与 Loss 计算

IGNORE_INDEX = -100

# 方法1：手动构造 labels（理解原理）
labels = input_ids.clone()  # 先复制一份 input_ids

# 找到 assistant 回复的起始位置
assistant_start = (input_ids == tokenizer.encode("<|im_start|>assistant")[0]).nonzero()[0][0]

# 将 assistant 之前的所有 token 都设为 IGNORE_INDEX
labels[:assistant_start] = IGNORE_INDEX

print("\n【Labels 构造示意】")
print(f"User 部分（不计算 Loss）: labels[0:{assistant_start}] = {IGNORE_INDEX}")
print(f"Assistant 部分（计算 Loss）: labels[{assistant_start}:] = 原始 token IDs")

# ---------- 5. 验证：哪些位置参与 Loss 计算 ----------
# mask = 1 表示计算 Loss，mask = 0 表示不计算 Loss
loss_mask = (labels != IGNORE_INDEX).int()

print("\n【Loss Mask 分布】")
print(f"User 部分 mask 值: {loss_mask[:assistant_start].tolist()}")
print(f"Assistant 部分 mask 值: {loss_mask[assistant_start:].tolist()[:5]}...")
# 输出：
# User 部分 mask 值: [0, 0, 0, 0, ...] ← 全部为 0，不参与计算
# Assistant 部分 mask 值: [1, 1, 1, 1, 1...] ← 全部为 1，参与计算

# ---------- 6. 计算 Loss（模型内部的实际操作）----------
# 将 input_ids 输入模型，得到每个位置的预测概率
outputs = model(input_ids=input_ids.unsqueeze(0), labels=labels.unsqueeze(0))

# 模型内部实际做的事情：
# 1. 将 input_ids 向右 shift 一位，得到预测目标
# 2. 计算每个位置的 cross-entropy loss
# 3. 将 labels == -100 的位置的 loss 设为 0
# 4. 对剩余位置的 loss 求平均

loss = outputs.loss
print(f"\n【Loss 值】: {loss.item()}")
# Loss 越小，说明模型回答越接近标准答案
```

#### 理论解释：为什么要这样设计？

**1. SFT 的本质：延续预训练的"预测下一个 token"逻辑**

- **预训练时**：整段文本都是训练数据，每个 token 都要预测下一个，模型学会"续写"
- **SFT 时**：只想教会模型"怎么回答"，user 部分是人类说的话，不需要模型学习

**2. 数学题类比**

| 对话部分                | 类比  | 是否需要学习      |
| ------------------- | --- | ----------- |
| user = "请计算 2+2=?"  | 题目  | 不需要，题目是外界给的 |
| assistant = "2+2=4" | 解答  | 需要，检查答得对不对  |

**3. 如果不做 Mask 会怎样？**

模型会被"教傻"：
- **学奇怪的东西**：模型会尝试学习 user 的提问方式，甚至把 prompt 格式（如 `<|user|>`）也当作学习内容
- **稀释训练效果**：user 部分占大量 token，稀释 assistant 部分的梯度信号
- **偏离训练目标**：想让模型学会回答，结果一半精力在学怎么提问

**4. 多轮对话的 Mask 规则**

```
<|user|> 你好                  ← Mask = 0，不计算 Loss
<|assistant|> 你好！有什么能帮你？ ← Mask = 1，计算 Loss ✅
<|user|> 帮我写首诗            ← Mask = 0，不计算 Loss
<|assistant|> 好的，这是一首...  ← Mask = 1，计算 Loss ✅
```

**规则：所有 assistant 的回复部分计算 Loss，所有 user 的提问部分不计算 Loss。**

---

## 八、SFT 的实际应用
### 8.1 TRL 库实现（工业界常用）

TRL（Transformer Reinforcement Learning）是 HuggingFace 官方推出的 SFT 训练库，**自动处理**数据格式化、Loss Mask 等所有细节。

```python
# ============================================================
# 使用 TRL 的 SFTTrainer 实现 SFT 训练
# 代码量大幅减少，所有底层细节自动处理
# ============================================================

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# 加载数据（支持 JSON、CSV 等格式）
dataset = load_dataset('json', data_files='sft_data.json')

# 配置训练参数
sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    max_seq_length=2048,
    dataset_text_field="text",  # 告诉 TRL 数据在哪个字段
    dataset_num_proc=4,         # 数据处理使用 4 个进程
)

# 创建训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=dataset["train"],
)

# 开始训练
# 底层自动做的事情：
# 1. 将数据转为对话格式
# 2. 自动构造 labels，实现 Loss Mask
# 3. 自动处理 padding、batching
# 4. 执行训练循环
trainer.train()
```

**TRL 自动处理的底层细节：**
- `apply_chat_template`：自动拼接对话格式
- `labels` 构造：自动将 user 部分设为 -100
- `padding`：自动补齐不同长度的序列
- `DataLoader`：自动 batching 和 shuffle

---

## 九、常见问题 FAQ

### Q1：SFT 需要多少数据？
> 几千到几十万条不等，关键看质量而非数量。一般场景 1 万 ~ 5 万条高质量数据即可。

### Q2：SFT 会让模型变笨吗？
> 可能会出现"灾难性遗忘"，即模型忘记预训练时学到的知识。使用 LoRA 可以大幅缓解这个问题，因为原始模型参数没有被修改。

### Q3：SFT 和 RLHF 有什么区别？
> SFT 教会模型"正确回答问题"，RLHF 教会模型"回答得更让人类喜欢"。SFT 在前，RLHF 在后，是上下级关系。

### Q4：我只有一个显卡，能做 SFT 吗？
> 可以！使用 QLoRA 方法，一张 8GB 显存的显卡就能微调 7B 规模的模型。推荐使用 LLaMA-Factory 等一键训练工具。

### Q5：为什么 SFT 数据量远小于预训练，效果却这么好？
> 因为预训练已经打下了坚实的知识基础，SFT 只是在已有基础上"调方向"，相当于让一个已经毕业的大学生学习新的工作流程，而不是从头学习。

---
