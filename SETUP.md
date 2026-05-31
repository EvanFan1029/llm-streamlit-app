# 环境部署与运行指南

以下所有命令在 **PowerShell** 中逐条复制粘贴执行。

---

## 一、安装 Ollama 并拉取模型

```powershell
# 1. 安装 Ollama（如果已安装可跳过）
winget install Ollama.Ollama

# 2. 启动 Ollama 服务（另开一个 PowerShell 窗口，保持运行不要关）
ollama serve
```

等 `ollama serve` 显示 `Listening on 127.0.0.1:11434` 后，回到原窗口继续：

```powershell
# 3. 拉取四个模型（约 20GB，需 15-30 分钟）
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull mistral:7b-instruct-v0.3-q5_0
ollama pull gemma2:9b-instruct-q4_K_M
ollama pull deepseek-r1:7b
```

> **Intel Arc 独立显卡用户**：关闭之前那个 `ollama serve` 窗口，改用下面命令启动以启用 GPU 推理：
>
> ```powershell
> $env:OLLAMA_VULKAN=1; ollama serve
> ```

---

## 二、安装 Python 依赖

```powershell
pip install streamlit sentence-transformers requests pandas numpy pytest
```

---

## 三、克隆代码并启动

```powershell
git clone https://github.com/EvanFan1029/llm-streamlit-app.git
cd llm-streamlit-app
streamlit run labor_law_app/app_labor_law.py
```

浏览器自动打开 `http://localhost:8501` 即可使用。

---

## 四、其他场景启动

```powershell
streamlit run translation_app/app.py      # 翻译场景
streamlit run medical_app/app_medical.py  # 医疗场景
```

---

## 五、运行测试

```powershell
python -m pytest labor_law_app/ -v
```

---

## 系统架构

```
原始案件文本
    │
    ▼
┌─ BERT Input Processor ────────────────────────────────────────┐
│  BGE-small-zh-v1.5 (24MB, 零样本)                              │
│  提取语义画像 → 构建统一 prompt                                  │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
4 个本地 LLM 收到相同 prompt，各自独立推理
  qwen2.5:7b / mistral:7b / gemma2:9b / deepseek-r1:7b
  模型差异自然产生分歧
    │
    ▼
┌─ BERT Output Processor ───────────────────────────────────────┐
│  语义匹配：LLM 输出 → 67 个闭集选项                              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ normalize_labor.py ──────────────────────────────────────────┐
│  规则 + BERT 加权合并 → 7 个标准劳动法 objects                   │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ labor_truthfinder.py ────────────────────────────────────────┐
│  TruthFinder EM 迭代：模型可信度 + 事实置信度                     │
│  family_dependency 自动处理同家族折扣                           │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ BERT Report Generator ───────────────────────────────────────┐
│  分歧度（模型间 top1 一致性）+ 置信度（信任加权）                 │
│  自然语言律师综合报告                                           │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Streamlit 前端 7-Step UI 展示
```

## 7 个劳动法标准 Objects

| Object ID | 标签 | 模式 |
|-----------|------|------|
| `relationship_type` | 法律关系初筛 | 单选 |
| `dispute_focus` | 核心争议类型 | 多选 |
| `key_fact` | 已识别关键事实 | 多选 |
| `issue_keyword` | 重点解析关键词 | 多选 |
| `article_reference` | 重点法条方向 | 多选 |
| `adjudication_tendency` | 裁判/处理倾向初筛 | 单选 |
| `background` | 重要背景信息 | 多选 |

## ZK 兼容性

BERT 全部处理均在 Groth16 零知识证明电路边界之外。ZK 电路 (`truthfinder.circom`) 只证明数值化的 TruthFinder Q16 定点计算，不受 BERT 引入的影响。
