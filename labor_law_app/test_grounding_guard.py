from __future__ import annotations

import pytest
from labor_law_app.grounding_guard import filter_unsupported_facts


class TestGroundingGuard:
    def test_female_should_not_trigger_pregnancy(self):
        case = "我是女性，在公司工作两年，被口头辞退，没有签劳动合同。"
        norm = {"background": ["劳动者为孕期/产期/哺乳期女职工", "无特殊背景信息"]}
        filtered, warnings = filter_unsupported_facts(norm, case)
        assert "劳动者为孕期/产期/哺乳期女职工" not in filtered["background"]
        assert "无特殊背景信息" in filtered["background"]
        assert len(warnings) >= 1
        assert any("孕期" in w["reason"] for w in warnings)

    def test_pregnant_should_keep_pregnancy(self):
        case = "我怀孕后公司把我调岗降薪，后来又辞退我。"
        norm = {"background": ["劳动者为孕期/产期/哺乳期女职工"]}
        filtered, warnings = filter_unsupported_facts(norm, case)
        assert "劳动者为孕期/产期/哺乳期女职工" in filtered["background"]
        assert len(warnings) == 0

    def test_child_pickup_should_not_trigger_lactation(self):
        case = "我因为下班接孩子经常迟到，公司以此为由辞退我。"
        norm = {"background": ["劳动者为孕期/产期/哺乳期女职工"]}
        filtered, warnings = filter_unsupported_facts(norm, case)
        assert "劳动者为孕期/产期/哺乳期女职工" not in filtered["background"]
        assert len(warnings) >= 1

    def test_married_mother_should_not_trigger(self):
        case = "我是一位母亲，已婚，有两个小孩。被公司辞退，理由是绩效差。"
        norm = {"background": ["劳动者为孕期/产期/哺乳期女职工"]}
        filtered, warnings = filter_unsupported_facts(norm, case)
        assert "劳动者为孕期/产期/哺乳期女职工" not in filtered["background"]

    def test_baby_care_mom_should_not_trigger(self):
        case = "我是宝妈，需要经常请假带孩子看病，公司以旷工为由辞退我。"
        norm = {"background": ["劳动者为孕期/产期/哺乳期女职工"]}
        filtered, warnings = filter_unsupported_facts(norm, case)
        assert "劳动者为孕期/产期/哺乳期女职工" not in filtered["background"]

    def test_normal_facts_pass_through(self):
        case = "我入职两年没有签劳动合同，被口头辞退，欠工资三个月。"
        norm = {"dispute_focus": ["未签书面劳动合同", "违法解除/辞退", "工资拖欠或克扣"]}
        filtered, warnings = filter_unsupported_facts(norm, case)
        assert filtered["dispute_focus"] == norm["dispute_focus"]
        assert len(warnings) == 0

    def test_injury_triggers_work_injury(self):
        case = "我在工作中受伤，公司不给报工伤，现在想申请工伤认定。"
        norm = {"background": ["劳动者为工伤/职业病职工"]}
        filtered, warnings = filter_unsupported_facts(norm, case)
        assert "劳动者为工伤/职业病职工" in filtered["background"]
        assert len(warnings) == 0

    def test_no_injury_removes_work_injury(self):
        case = "我工作两年，最近一直腰疼，应该是坐久了，公司不管。"
        norm = {"background": ["劳动者为工伤/职业病职工"]}
        filtered, warnings = filter_unsupported_facts(norm, case)
        assert "劳动者为工伤/职业病职工" not in filtered["background"]

    def test_background_empty_falls_back_to_default(self):
        case = "我是女性，被公司无理由辞退。"
        norm = {"background": ["劳动者为孕期/产期/哺乳期女职工"]}
        filtered, warnings = filter_unsupported_facts(norm, case)
        assert filtered["background"] == ["无特殊背景信息"]

    def test_empty_input_returns_empty(self):
        filtered, warnings = filter_unsupported_facts({}, "")
        assert filtered == {}
        assert warnings == []
