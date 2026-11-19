# 大模型学习笔记 - 索引

> 最后更新：2026-11-19

---

## 架构/

| 笔记 | 标签 | 简介 |
|------|------|------|
| [[Transformer]] | 架构, 核心 | Transformer 架构详解 |
| [Transformer 专题目录](Transformer/README.md) | 架构, 外部资料 | 12 篇笔记及配图（[原仓库](https://github.com/jiayun-xiong/deep-learning-notes)） |
| [[注意力机制]] | 架构, 注意力机制, 高效Transformer | 注意力机制变体详解 |
| [[RoPE旋转位置编码]] | 架构, 位置编码 | 旋转位置编码 |
| [[SwiGLU激活函数]] | 架构, FFN | 现代大模型标配激活函数 |
| [[自回归生成]] | 架构, 推理 | 自回归生成原理 |
| [[KV Cache]] | 架构, 推理优化 | KV Cache 原理和优化 |

---

## 微调/

| 笔记 | 标签 | 简介 |
|------|------|------|
| [[SFT：监督微调]] | 微调, SFT | 监督微调流程 |
| [[LoRA]] | 微调, 参数高效 | LoRA 微调 |
| [[QLoRA]] | 微调, 量化 | 量化 LoRA |
| [[raw/lora_qlora/lora_qlora\|LoRA/QLoRA 低秩微调详解]] | 微调, LoRA, QLoRA, 外部资料 | 低秩微调、参数设置与显存优化（[原文来源](https://github.com/youngyangyang04/llm-master/blob/main/docs/llm/app/lora_qlora.md)） |
| [[DPO]] | 微调, 对齐 | 直接偏好优化 |
| [[PPO]] | 微调, 对齐, RL | 近端策略优化 |
| [[GRPO]] | 微调, 对齐, RL | 组相对策略优化 |
| [[RLHF]] | 微调, 对齐 | 基于人类反馈的强化学习 |
| [[奖励模型]] | 微调, 对齐 | 奖励模型训练 |
| [[偏好数据构建]] | 微调, 对齐, 数据 | 偏好对构造、标注准则、去重划分与 JSONL 示例 |

---

## 预训练/

| 笔记 | 标签 | 简介 |
|------|------|------|
| [[Scaling Laws]] | 预训练, 规模定律 | 模型规模与性能 |
| [[预训练数据工程]] | 预训练, 数据 | 预训练数据处理 |
| [[Tokenizer与BPE算法]] | 预训练, Tokenizer | 分词器和 BPE |

---

## 推理/

| 笔记 | 标签 | 简介 |
|------|------|------|
| [[推理优化]] | 推理, 优化 | 推理优化综合指南 |
| [[量化技术]] | 推理, 量化 | 模型量化 |
| [[长上下文技术]] | 推理, 长上下文 | 长序列处理 |
| [[分布式训练]] | 训练, 分布式 | 分布式训练 |
| [[混合精度训练]] | 训练, 精度 | 混合精度训练 |

---

## 应用/

| 笔记 | 标签 | 简介 |
|------|------|------|
| [[RAG]] | 应用, 检索增强 | 检索增强生成总览 |
| [[Agentic RAG]] | 应用, 检索增强, Agent | 智能体增强的检索增强生成 |
| [[Embedding模型]] | 应用, 检索增强, Embedding | 文本向量化技术 |
| [[向量数据库]] | 应用, 检索增强, 向量数据库 | 向量存储与检索 |
| [[文档分块策略]] | 应用, 检索增强, 分块 | Chunking 技术 |
| [[混合检索]] | 应用, 检索增强, 检索 | 向量检索 + 关键词检索 |
| [[重排序Reranking]] | 应用, 检索增强, 排序 | 检索结果精排 |
| [[查询改写QueryRewriting]] | 应用, 检索增强, 查询 | 优化检索查询 |
| [[知识蒸馏]] | 应用, 蒸馏 | 知识蒸馏 |
| [[Agent与工具调用]] | 应用, Agent | Agent 架构 |
| [[LangGraph]] | 应用, Agent, 框架 | LLM 应用状态图编排框架 |
| [[Prompt Engineering与CoT]] | 应用, 提示工程 | 提示工程 |
| [[Coze平台使用教程]] | 应用, 平台, Bot开发 | Coze 零代码 AI Bot 开发（概念篇） |
| [[Coze实战案例]] | 应用, 平台, 实战, Coze | Coze 六种开发模式实战（案例篇） |

---

## 综合/

| 笔记 | 标签 | 简介 |
|------|------|------|
| [[大模型面试八股]] | 面试, 综合 | 面试高频问题 |
| [[大模型评估基准]] | 评估 | 模型评估 |

---

## 标签云

- **架构**：Transformer, 注意力机制, RoPE, SwiGLU, 自回归生成, KV Cache
- **微调**：LoRA, QLoRA, SFT, DPO, PPO, GRPO, RLHF
- **预训练**：Scaling Laws, 预训练数据, Tokenizer
- **推理**：推理优化, 量化, 长上下文, 分布式训练
- **应用**：RAG, Embedding, 向量数据库, 分块, 混合检索, Reranking, 查询改写, 知识蒸馏, Agent, LangGraph, Prompt Engineering, Coze

## 重要性

- ⭐⭐⭐ **核心**：Transformer, 注意力机制, LoRA, RLHF, 推理优化, RAG
- ⭐⭐ **重要**：SwiGLU, KV Cache, 量化, DPO, PPO, Embedding, 混合检索, Reranking, LangGraph
- ⭐ **了解**：Scaling Laws, 知识蒸馏, Agent, 向量数据库, 分块策略, 查询改写, Coze
