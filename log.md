# 大模型学习笔记 - 更新日志

> 记录每次对笔记系统的更新，便于追踪和回溯。

---

## [2026-11-19] ingest | 补写偏好数据构建笔记

**操作**：ingest（补写空白笔记）

**内容**：
- 介绍偏好对、标注准则和难负例，并解释同题分组、去重、截断和裁判偏差。
- 补充可运行的 Python 示例，将原始胜负标注转换为 JSONL 偏好记录。
- 更新微调索引，补齐 DPO、RLHF 与偏好数据构建的交叉引用。

**更新的文件**：
- 补写：微调/偏好数据构建.md
- 更新：微调/DPO.md、微调/RLHF.md、index.md、log.md

---

## [2026-11-19] update | 精简 Git 忽略规则

**操作**：update（更新忽略配置）

**内容**：
- 保留现有 Obsidian 本地配置的忽略规则。
- 移除当前仓库没有对应文件或目录的 Claude Code、.DS_Store 和 Coze 原始资料规则。

**更新的文件**：
- 更新：.gitignore、log.md

---

## [2026-11-19] ingest | 导入 Transformer 专题笔记

**操作**：ingest（导入一组外部笔记）

**内容**：
- 从 [jiayun-xiong/deep-learning-notes](https://github.com/jiayun-xiong/deep-learning-notes) 导入 12 篇 Markdown 笔记及其 PDF、配图、Notebook 等附件。来源提交为 cec5f24。
- 保留主题目录及文件的原有布局。专题 README 仅保留「笔记目录」，不带入来源仓库的 .gitignore。
- 来源中的两条「系列规划」链接指向同一个未收录的文件，保留原文链接。

**更新的文件**：
- 新建：Transformer/（专题目录与笔记文件）
- 更新：README.md、index.md、log.md

---

## [2026-11-19] refactor | 整理 LoRA/QLoRA 笔记图片

**操作**：refactor（整理笔记与配图）

**内容**：
- 将原文和配图归入 `raw/lora_qlora/`，Markdown 文件与 `figs/` 同级。
- 将 6 张图片保存到 `figs/`，正文图片改用相对路径。
- 更新 README 和索引中的笔记入口，将 README 的链接名称改为 lora_qlora，并移除「来源与鸣谢」中的两条说明。

**更新的文件**：
- 移动：`raw/lora_qlora.md` → `raw/lora_qlora/lora_qlora.md`
- 新建：`raw/lora_qlora/figs/` 中的 6 张图片
- 更新：`README.md`、`index.md`、`log.md`

---

## [2026-11-19] refactor | 归档 LoRA/QLoRA 外部原文

**操作**：refactor（整理外部资料）

**内容**：
- 将 LoRA/QLoRA 外部原文移到 `raw/` 目录，加入 Git 跟踪。
- 修正 README 中“内容导览”和“来源与鸣谢”的两处原文链接。

**更新的文件**：
- 移动：`微调/lora_qlora.md` → `raw/lora_qlora.md`
- 更新：`README.md`（修正原文路径）
- 更新：`log.md`（记录本次整理）

---

## [2026-11-19] refactor | 清理旧维护者引用

**操作**：refactor（清理文字引用与历史关联）

**内容**：
- 清理 README 和既有日志中旧维护者的姓名、账号及仓库链接。
- 移除旧来源远端关联，按个人维护版本重建可见提交历史。
- 保留通用资料归属说明、其他来源标注，以及当前未提交的学习笔记修改。

**更新的文件**：
- 更新：`README.md`（清理旧维护者的明确引用）
- 更新：`log.md`（整理相关历史记录并记录本次清理）

---

## [2026-11-18] refactor | 更新仓库说明并准备个人维护版本

**操作**：refactor（整理仓库说明与维护历史）

**内容**：
- 按当前内容更新 README 的主题导览、学习路线、阅读方式与维护说明。
- 保留 基础笔记和 llm-master 外部文章的来源说明。
- 备份既有提交历史，按当前文件准备由 slate-zyc 提交的独立初始版本。

**更新的文件**：
- 更新：`README.md`（当前内容导览、维护方式与来源说明）
- 更新：`log.md`（记录仓库整理与独立维护版本的准备）

---

## [2026-11-18] ingest | 加入 LoRA/QLoRA 外部原文

**操作**：ingest（添加外部 Markdown）

**内容**：
- 直接导入 [youngyangyang04/llm-master 的 lora_qlora.md](https://github.com/youngyangyang04/llm-master/blob/main/docs/llm/app/lora_qlora.md)，保留原文正文与 Frontmatter。
- 将原文中的相对图片地址和前篇文章链接改为来源仓库的完整地址，保证移入本仓库后可访问。
- 将新笔记加入微调索引，注明来源并更新索引日期。

**更新的文件**：
- 新建：`微调/lora_qlora.md`（导入原文，仅适配两处相对链接）
- 更新：`index.md`（添加微调索引条目与来源）
- 更新：`log.md`（记录本次导入）

---

## [2026-11-18] update | 创建公开远端仓库

**操作**：update（配置远端仓库）

**内容**：
- 在 slate-zyc 账号下创建公开仓库 [LLMnotebook](https://github.com/slate-zyc/LLMnotebook)。
- 将个人仓库设置为默认远端。
- 更新 README 的克隆地址，并注明原始笔记来源。

**更新的文件**：
- 更新：`README.md`（本仓库克隆地址与来源说明）
- 更新：`log.md`（记录公开远端仓库的创建与配置）

---

## [2026-11-18] ingest | 创建本地仓库

**操作**：ingest（创建本地仓库）

**内容**：
- 在 `D:\LLMnotebook` 创建本地笔记仓库。
- 由 Codex 接手笔记维护，将 `CLAUDE.md` 更名为 `AGENT.md`，并同步更新维护者说明和文档引用。
- 清空原有日志记录，保留“日志格式说明”和“常用操作类型”，按原格式记录本次创建。

**更新的文件**：
- 重命名并更新：`CLAUDE.md` → `AGENT.md`（维护规范与日志要求）
- 更新：`README.md`（维护者和规范链接）
- 更新：`log.md`（清理旧记录，添加本次日志）

---

## 日志格式说明

每条日志格式：
```
## [日期] 操作类型 | 标题

**操作**：ingest（添加）/ update（更新）/ lint（检查）/ query（查询）

**内容**：
- 做了什么

**更新的文件**：
- 新建/更新了哪些文件
```

---

## 常用操作类型

| 操作 | 说明 |
|------|------|
| `ingest` | 添加新笔记或新内容 |
| `update` | 更新现有笔记 |
| `lint` | 检查笔记一致性 |
| `query` | 查询并形成新洞察 |
| `refactor` | 重构笔记结构 |
