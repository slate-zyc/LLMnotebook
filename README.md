# LLMnotebook — 大模型学习笔记

这是 [slate-zyc](https://github.com/slate-zyc) 持续整理的大模型学习笔记，覆盖预训练、模型架构、微调对齐、推理优化和应用开发。仓库用于记录学习过程、收集参考资料，并逐步补充自己的理解与实践。

完整目录见 [index.md](index.md)，更新记录见 [log.md](log.md)。

## 内容导览

| 目录 | 内容 | 阅读入口 |
|------|------|----------|
| [预训练](预训练/) | 数据工程、分词器、规模定律 | [Tokenizer 与 BPE](预训练/Tokenizer与BPE算法.md)、[Scaling Laws](预训练/Scaling%20Laws.md) |
| [架构](架构/) | Transformer、注意力机制、位置编码与缓存 | [Transformer](架构/Transformer.md)、[注意力机制](架构/注意力机制.md)、[RoPE](架构/RoPE旋转位置编码.md)、[KV Cache](架构/KV%20Cache.md) |
| [微调](微调/) | SFT、LoRA/QLoRA、偏好优化与强化学习 | [LoRA](微调/LoRA.md)、[QLoRA](微调/QLoRA.md)、[LoRA/QLoRA 外部原文](微调/lora_qlora.md)、[DPO](微调/DPO.md)、[GRPO](微调/GRPO.md) |
| [推理](推理/) | 量化、长上下文、混合精度与分布式训练 | [推理优化](推理/推理优化.md)、[量化技术](推理/量化技术.md)、[长上下文技术](推理/长上下文技术.md) |
| [应用](应用/) | RAG、Agent、工作流与应用平台 | [RAG](应用/RAG.md)、[Agent 与工具调用](应用/Agent与工具调用.md)、[LangGraph](应用/LangGraph.md) |
| [综合](综合/) | 评估基准与面试知识整理 | [大模型评估基准](综合/大模型评估基准.md)、[大模型面试八股](综合/大模型面试八股.md) |

## 学习路线

预训练数据与分词 → Transformer 与注意力机制 → SFT 与 LoRA/QLoRA → RLHF、DPO 与 GRPO → 量化与推理优化 → RAG 与 Agent。

## 阅读方式

可以直接通过上面的链接在 GitHub 阅读，也可以克隆到本地：

```bash
git clone https://github.com/slate-zyc/LLMnotebook.git
```

使用 Obsidian 时，选择“打开文件夹作为仓库”，打开本仓库目录。笔记之间使用 `[[笔记名]]` 双链，在 Obsidian 中可以跳转；在 GitHub 上阅读时可通过本页的普通链接进入各主题。

## 维护方式

- 由 slate-zyc 整理，Codex 协助维护。
- 笔记维护规范见 [AGENT.md](AGENT.md)。
- 新增或更新内容时同步维护索引，并按照 [log.md](log.md) 中的格式记录变动。

## 来源与鸣谢

- 部分基础笔记参考公开学习资料，原始内容归相应作者所有。
- [LoRA/QLoRA 外部原文](微调/lora_qlora.md) 来源于 [youngyangyang04/llm-master](https://github.com/youngyangyang04/llm-master/blob/main/docs/llm/app/lora_qlora.md)，保留正文，仅将两处相对链接改为完整地址。
- 后续引入的外部资料会在对应笔记、索引或更新日志中注明来源。

## 说明

笔记用于个人学习与交流，可能存在疏漏；代码示例仅供学习，实际使用前请自行验证。如涉及原作者权益，请联系仓库维护者处理。
