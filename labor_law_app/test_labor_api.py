from __future__ import annotations

from labor_law_app.api import analyze_labor_case, get_api_schema


def test_schema_exposes_analyze_endpoint() -> None:
    schema = get_api_schema()

    assert schema["endpoint"]["path"] == "/analyze"
    assert any(item["object_id"] == "article_reference" for item in schema["objects"])


def test_analyze_labor_case_rule_baseline_outputs_articles() -> None:
    result = analyze_labor_case(
        {
            "case_id": "case_unwritten_dismissal",
            "case_text": "劳动者2023年3月入职，月薪8000元，一直未签书面劳动合同，有工资流水和打卡记录。2024年4月公司口头辞退，不让继续上班，没有支付经济补偿。",
        }
    )

    assert result["ok"] is True
    assert result["used_rule_baseline"] is True
    labels = {item["label"] for item in result["article_candidates"]}
    keywords = {item["keyword"] for item in result["issue_keywords"]}

    assert "劳动合同法第82条" in labels
    assert "劳动合同法第87条" in labels
    assert {"双倍工资", "违法解除", "经济补偿"} & keywords
    assert result["lawyer_brief"]["core_disputes"]


def test_analyze_labor_case_accepts_multi_model_outputs() -> None:
    payload = {
        "case_id": "case_overtime",
        "case_text": "员工实行标准工时，有考勤和排班记录，长期休息日加班但公司未支付加班费。",
        "model_outputs": {
            "legal_checker": {
                "case_summary": "主要争议为加班费。",
                "structured_analysis": {
                    "relationship_type": "劳动关系倾向",
                    "dispute_focus": ["加班费争议"],
                    "key_fact": ["考勤/工时记录存在", "加班事实或审批记录存在"],
                    "issue_keyword": ["加班费", "举证责任"],
                    "article_reference": ["劳动法第44条", "劳动争议调解仲裁法第6条"],
                    "adjudication_tendency": "支持劳动者主要请求倾向",
                },
            },
            "evidence_checker": {
                "case_summary": "需要核验加班审批和工资发放。",
                "structured_analysis": {
                    "relationship_type": "劳动关系倾向",
                    "dispute_focus": ["加班费争议"],
                    "key_fact": ["考勤/工时记录存在"],
                    "issue_keyword": ["加班费"],
                    "article_reference": ["劳动合同法第31条", "劳动法第44条"],
                    "adjudication_tendency": "部分支持/需分项判断",
                },
            },
        },
    }

    result = analyze_labor_case(payload)

    assert result["ok"] is True
    assert result["used_rule_baseline"] is False
    labels = {item["label"] for item in result["article_candidates"]}
    assert "劳动法第44条" in labels
    assert result["truthfinder"]["model_trust_rank"]

