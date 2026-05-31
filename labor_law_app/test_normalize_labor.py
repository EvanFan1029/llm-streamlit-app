from __future__ import annotations

from labor_law_app.normalize_labor import (
    build_labor_fact_table,
    normalize_labor_fact,
    normalize_model_labor_output,
)


def test_unwritten_contract_and_dismissal_mapping() -> None:
    text = "2023年3月入职，月薪8000元，一直没有签书面劳动合同，有打卡记录，2024年4月被公司口头辞退，没有支付经济补偿。"
    result = normalize_model_labor_output({"structured_analysis": {}}, user_text=text)
    normalized = result["normalized"]

    assert normalized["relationship_type"] == ["劳动关系倾向"]
    assert "未签书面劳动合同" in normalized["dispute_focus"]
    assert "违法解除/辞退" in normalized["dispute_focus"]
    assert "经济补偿/赔偿金" in normalized["dispute_focus"]
    assert "双倍工资" in normalized["issue_keyword"]
    assert "违法解除" in normalized["issue_keyword"]
    assert "劳动合同法第82条" in normalized["article_reference"]
    assert "劳动合同法第87条" in normalized["article_reference"]


def test_noncompete_without_compensation_maps_articles() -> None:
    text = "员工签了两年竞业限制协议，离职后公司要求履行竞业义务，但一直未支付竞业补偿。"
    facts = normalize_labor_fact("article_reference", "", user_text=text)

    assert "劳动合同法第23条" in facts
    assert "劳动合同法第24条" in facts


def test_training_service_period_penalty_maps_article_22() -> None:
    text = "公司主张专项培训服务期违约金5万元，但只有内部培训记录，没有培训费发票。"
    result = normalize_model_labor_output({"structured_analysis": {}}, user_text=text)
    normalized = result["normalized"]

    assert "服务期/培训违约金" in normalized["dispute_focus"]
    assert "服务期违约金" in normalized["issue_keyword"]
    assert "劳动合同法第22条" in normalized["article_reference"]


def test_fact_table_excludes_missing_fallbacks() -> None:
    normalized_all = {
        "model_a": normalize_model_labor_output(
            {"structured_analysis": {"dispute_focus": ["加班费争议"]}},
            user_text="",
        )
    }

    keep = build_labor_fact_table(normalized_all, source="from_model_fields", exclude_fallbacks=False)
    filtered = build_labor_fact_table(normalized_all, source="from_model_fields", exclude_fallbacks=True)

    keep_relationship = next(row for row in keep if row["object_id"] == "relationship_type")
    filtered_relationship = next(row for row in filtered if row["object_id"] == "relationship_type")

    assert keep_relationship["facts"] == ["事实不清"]
    assert filtered_relationship["facts"] == []

