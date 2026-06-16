# ⚖️ 劳动法 TruthFinder 系统 — 启动指南

## 环境要求

- **Python 3.10+**
- **Ollama**（本地 LLM 推理，需提前安装并拉取模型）
- 依赖包已在 `requirements-bert.txt` 和项目本地 deps 中

---

## 1. 启动 Ollama

在终端运行（或双击 Ollama 应用图标）：

```powershell
ollama serve
```

确认已拉取以下模型（至少 2 个）：

```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull mistral:7b-instruct-v0.3-q5_0
ollama pull gemma2:9b-instruct-q4_K_M
ollama pull koesn/mistral-7b-instruct:Q4_0
```

验证 Ollama 是否就绪：

```powershell
curl http://127.0.0.1:11434/api/tags
```

---

## 2. 启动 API 后端（非必需，但推荐）

在项目根目录运行：

```powershell
cd D:\Final\llm-streamlit-app
python -m labor_law_app.api --serve --host 127.0.0.1 --port 8008
```

验证 API：

```powershell
curl http://127.0.0.1:8008/schema
```

或用 Demo 模式测试：

```powershell
python -m labor_law_app.api --demo
```

---

## 3. 启动 Streamlit 前端

在项目根目录运行：

```powershell
cd D:\Final\llm-streamlit-app
streamlit run labor_law_app/app_labor_law.py --server.port 8501 --server.address 127.0.0.1
```

浏览器打开：

> **http://127.0.0.1:8501**

---

## 4. 使用流程

| 步骤 | 操作 | 说明 |
|------|------|------|
| Step 1 | 输入案件描述 + 选择模型 + 点击「开始四模型分析」 | 口语化输入即可，如："我2023年3月入职，一直没签合同，上个月被口头辞退了" |
| Step 3 | 查看每个模型的自然语言分析 + 展开结构化 JSON | 模型同时输出 `user_explanation` 和 `structured_analysis` |
| Step 4 | 查看 BERT 案件语义画像 | 5 轴分数：劳动关系/证据/违法程度/违规可能/诉求强度 |
| Step 5 | 查看七维度结构化对比表 | 7 维度 × 各模型的原始判断 |
| Step 6 | 点击「运行归一化」 | 将各模型输出统一映射到标准标签 |
| Step 7 | 点击「运行 TruthFinder」 | 多模型可信度交叉验证，输出维度假排名 |
| Step 8 | 查看律师综合报告 | 模板生成：法律关系→争议→法条→证据缺口→下一步 |

---

## 5. 快速验证（纯规则基线，不调 LLM）

如果想快速看结果，勾选「**纯规则基线模式**」→ 点击「开始四模型分析」→ 秒级输出法条映射和证据缺口。

---

## 6. 常见问题

| 问题 | 解决 |
|------|------|
| Ollama 连接超时 | 确认 `ollama serve` 已运行，`http://127.0.0.1:11434` 可访问 |
| 模型调用失败 | 确认模型已 `ollama pull`，内存充足（每个 7B 模型约需 4-6 GB） |
| BERT 首次加载慢 | 首次需下载 `BAAI/bge-small-zh-v1.5`（约 100MB），后续缓存即秒开 |
| Streamlit 端口冲突 | 修改 `--server.port` 参数 |
| 中文乱码 | 终端编码问题，不影响浏览器端显示 |

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `app_labor_law.py` | Streamlit 前端 |
| `api.py` | HTTP API 后端 |
| `normalize_labor.py` | 归一化引擎（7 维度 + 30 法条 + 规则抽取） |
| `labor_truthfinder.py` | TruthFinder EM 聚合算法 |
| `bert_prompts.py` | LLM 结构化 Prompt 模板 |
| `bert_processor.py` | BERT 嵌入 + 语义匹配 |
| `bert_input_processor.py` | 案件语义画像 |
| `bert_output_processor.py` | 模型输出语义对齐 |
| `bert_report_generator.py` | 分歧度/置信度报告 |
| `README.md` | API 输入输出规范 |
