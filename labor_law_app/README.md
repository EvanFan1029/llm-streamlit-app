# 劳务/劳动合同纠纷 API 场景

该场景仿照原项目的“场景适配层 + TruthFinder 聚合层”结构，把在职场上的劳务合同/劳动合同纠纷整理成可调用 JSON API。

## 用户需要输入什么

最小输入只需要一段 `case_text`：

```json
{
  "case_id": "case_001",
  "case_text": "劳动者2023年3月入职，月薪8000元，一直未签书面劳动合同，有工资流水和打卡记录。2024年4月公司口头辞退，没有支付经济补偿。"
}
```

推荐输入可附带多个模型的结构化输出：

```json
{
  "case_text": "案情描述...",
  "model_outputs": {
    "legal_checker": {
      "case_summary": "主要争议为未签合同和违法解除。",
      "structured_analysis": {
        "relationship_type": "劳动关系倾向",
        "dispute_focus": ["未签书面劳动合同", "违法解除/辞退"],
        "key_fact": ["入职/用工时间明确", "未签书面劳动合同", "解除通知或离职原因明确"],
        "issue_keyword": ["双倍工资", "违法解除", "赔偿金"],
        "article_reference": ["劳动合同法第10条", "劳动合同法第82条", "劳动合同法第87条"],
        "adjudication_tendency": "支持劳动者主要请求倾向"
      }
    }
  }
}
```

## 标准 object / fact

API 会把自然语言和模型输出统一归一化到六个 object：

| object | 含义 | 类型 |
|---|---|---|
| `relationship_type` | 法律关系初筛：劳动关系、劳务/承揽、事实不清 | single |
| `dispute_focus` | 核心争议类型：未签合同、加班费、违法解除等 | multi |
| `key_fact` | 已识别关键事实：入职时间、工资记录、考勤、解除通知等 | multi |
| `issue_keyword` | 重点解析关键词：双倍工资、违法解除、仲裁时效等 | multi |
| `article_reference` | 重点法条方向：劳动合同法、劳动法、民法典等具体条文 | multi |
| `adjudication_tendency` | 裁判/处理倾向初筛 | single |

## API 调用

启动服务：

```bash
python -m labor_law_app.api --serve --host 127.0.0.1 --port 8008
```

查看 schema：

```bash
curl http://127.0.0.1:8008/schema
```

分析案件：

```bash
curl -X POST http://127.0.0.1:8008/analyze \
  -H "Content-Type: application/json" \
  -d "{\"case_text\":\"劳动者未签书面劳动合同，后被口头辞退，主张双倍工资和赔偿金。\"}"
```

## 主要输出

- `case_facts`：从用户原文抽取的标准 facts。
- `normalized_by_source`：各模型/来源归一化结果，保留 `from_model_fields` 和 `from_user_text`。
- `truthfinder`：模型可信度排序、各 object 候选事实置信度和 debug 摘要。
- `issue_keywords`：本案重点解析关键词。
- `article_candidates`：候选法条、置信度、触发来源、法条摘要。
- `issue_article_matrix`：哪些问题对应哪些法条，给律师做检索和论证提纲。
- `evidence_gaps`：建议补充核验的证据点。
- `lawyer_brief`：核心争议、关键事实、重点法条和下一步工作摘要。

边界：该接口用于事实整理、争议点归纳与法条筛选，不构成正式法律意见、裁判预测或案件结果保证。

