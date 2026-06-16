from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LaborObjectSchema:
    object_id: str
    label: str
    mode: str
    options: tuple[str, ...]
    fallback: tuple[str, ...]


@dataclass(frozen=True)
class LegalArticle:
    article_id: str
    label: str
    law_name: str
    short_law: str
    article_no: str
    article_no_cn: str
    topic: str
    summary: str
    issue_keywords: tuple[str, ...]


LEGAL_ARTICLES: tuple[LegalArticle, ...] = (
    LegalArticle(
        "lcl_7",
        "劳动合同法第7条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第7条",
        "第七条",
        "劳动关系建立",
        "用工开始通常是判断劳动关系成立的核心时间点。",
        ("劳动关系",),
    ),
    LegalArticle(
        "lcl_10",
        "劳动合同法第10条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第10条",
        "第十条",
        "书面劳动合同订立",
        "建立劳动关系应订立书面劳动合同，已用工未同时订立的，应在法定期限内补签。",
        ("书面劳动合同", "双倍工资"),
    ),
    LegalArticle(
        "lcl_14",
        "劳动合同法第14条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第14条",
        "第十四条",
        "无固定期限劳动合同",
        "连续用工、连续订立固定期限合同等情形下，需关注无固定期限劳动合同规则。",
        ("书面劳动合同", "劳动关系"),
    ),
    LegalArticle(
        "lcl_19",
        "劳动合同法第19条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第19条",
        "第十九条",
        "试用期期限",
        "试用期长短与劳动合同期限联动，且同一用人单位与同一劳动者通常只能约定一次试用期。",
        ("试用期",),
    ),
    LegalArticle(
        "lcl_20",
        "劳动合同法第20条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第20条",
        "第二十条",
        "试用期工资",
        "试用期工资不得低于法定比例标准，并不得低于当地最低工资标准。",
        ("试用期", "工资支付"),
    ),
    LegalArticle(
        "lcl_22",
        "劳动合同法第22条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第22条",
        "第二十二条",
        "服务期与培训违约金",
        "只有在专项培训费用与服务期约定等条件下，服务期违约金才有重点审查价值。",
        ("服务期违约金",),
    ),
    LegalArticle(
        "lcl_23",
        "劳动合同法第23条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第23条",
        "第二十三条",
        "保密与竞业限制",
        "竞业限制、保密义务及经济补偿约定是审查竞业争议的基础。",
        ("竞业限制",),
    ),
    LegalArticle(
        "lcl_24",
        "劳动合同法第24条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第24条",
        "第二十四条",
        "竞业限制范围与期限",
        "竞业限制对象、范围、地域、期限均需合理，期限通常不得超过二年。",
        ("竞业限制",),
    ),
    LegalArticle(
        "lcl_30",
        "劳动合同法第30条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第30条",
        "第三十条",
        "劳动报酬支付",
        "用人单位应按劳动合同约定和国家规定及时足额支付劳动报酬。",
        ("工资支付",),
    ),
    LegalArticle(
        "lcl_31",
        "劳动合同法第31条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第31条",
        "第三十一条",
        "加班与劳动定额",
        "安排加班应依法支付加班费，劳动定额不得迫使劳动者变相超时劳动。",
        ("加班费",),
    ),
    LegalArticle(
        "lcl_39",
        "劳动合同法第39条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第39条",
        "第三十九条",
        "过失性解除",
        "用人单位以严重违纪、不符合录用条件等理由解除时，需要重点审查事实和制度依据。",
        ("违法解除", "举证责任"),
    ),
    LegalArticle(
        "lcl_40",
        "劳动合同法第40条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第40条",
        "第四十条",
        "无过失性解除",
        "非过失性解除通常涉及提前通知、代通知金及事实基础审查。",
        ("违法解除", "经济补偿"),
    ),
    LegalArticle(
        "lcl_46",
        "劳动合同法第46条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第46条",
        "第四十六条",
        "经济补偿情形",
        "解除或终止劳动合同是否应支付经济补偿，应先匹配法定情形。",
        ("经济补偿",),
    ),
    LegalArticle(
        "lcl_47",
        "劳动合同法第47条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第47条",
        "第四十七条",
        "经济补偿计算",
        "经济补偿通常按工作年限和劳动者月工资基数计算。",
        ("经济补偿", "赔偿金"),
    ),
    LegalArticle(
        "lcl_48",
        "劳动合同法第48条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第48条",
        "第四十八条",
        "违法解除后果",
        "违法解除后，劳动者可主张继续履行或依法请求赔偿。",
        ("违法解除", "赔偿金"),
    ),
    LegalArticle(
        "lcl_82",
        "劳动合同法第82条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第82条",
        "第八十二条",
        "未签书面劳动合同双倍工资",
        "未依法订立书面劳动合同或无固定期限劳动合同的，可能触发双倍工资责任。",
        ("书面劳动合同", "双倍工资"),
    ),
    LegalArticle(
        "lcl_87",
        "劳动合同法第87条",
        "中华人民共和国劳动合同法",
        "劳动合同法",
        "第87条",
        "第八十七条",
        "违法解除赔偿金",
        "用人单位违法解除或终止劳动合同，赔偿金通常以经济补偿标准的二倍为基础。",
        ("违法解除", "赔偿金"),
    ),
    LegalArticle(
        "labor_law_36",
        "劳动法第36条",
        "中华人民共和国劳动法",
        "劳动法",
        "第36条",
        "第三十六条",
        "标准工时",
        "标准工时是判断延长工作时间和加班争议的重要基础。",
        ("加班费",),
    ),
    LegalArticle(
        "labor_law_41",
        "劳动法第41条",
        "中华人民共和国劳动法",
        "劳动法",
        "第41条",
        "第四十一条",
        "延长工作时间限制",
        "安排延长工作时间需满足协商和时长限制等条件。",
        ("加班费",),
    ),
    LegalArticle(
        "labor_law_44",
        "劳动法第44条",
        "中华人民共和国劳动法",
        "劳动法",
        "第44条",
        "第四十四条",
        "加班工资标准",
        "工作日、休息日、法定休假日加班通常对应不同支付标准。",
        ("加班费",),
    ),
    LegalArticle(
        "labor_law_50",
        "劳动法第50条",
        "中华人民共和国劳动法",
        "劳动法",
        "第50条",
        "第五十条",
        "工资支付方式",
        "工资应以货币形式按月支付给劳动者本人，不得无故克扣或拖欠。",
        ("工资支付",),
    ),
    LegalArticle(
        "labor_law_72",
        "劳动法第72条",
        "中华人民共和国劳动法",
        "劳动法",
        "第72条",
        "第七十二条",
        "社会保险",
        "用人单位和劳动者依法参加社会保险并缴纳社会保险费。",
        ("社会保险",),
    ),
    LegalArticle(
        "ldar_6",
        "劳动争议调解仲裁法第6条",
        "中华人民共和国劳动争议调解仲裁法",
        "劳动争议调解仲裁法",
        "第6条",
        "第六条",
        "劳动争议举证责任",
        "当事人对主张负有举证责任；用人单位掌握管理的证据不提供时可能承担不利后果。",
        ("举证责任", "劳动关系", "工资支付", "加班费", "违法解除"),
    ),
    LegalArticle(
        "ldar_27",
        "劳动争议调解仲裁法第27条",
        "中华人民共和国劳动争议调解仲裁法",
        "劳动争议调解仲裁法",
        "第27条",
        "第二十七条",
        "仲裁时效",
        "劳动争议申请仲裁通常适用一年时效，劳动报酬争议有特别规则。",
        ("仲裁时效", "工资支付"),
    ),
    LegalArticle(
        "civil_code_143",
        "民法典第143条",
        "中华人民共和国民法典",
        "民法典",
        "第143条",
        "第一百四十三条",
        "民事法律行为有效条件",
        "在劳动关系不成立而转入民事合同评价时，可用于审查合同行为效力基础。",
        ("劳动关系", "服务期违约金", "竞业限制"),
    ),
    LegalArticle(
        "civil_code_465",
        "民法典第465条",
        "中华人民共和国民法典",
        "民法典",
        "第465条",
        "第四百六十五条",
        "依法成立合同受保护",
        "用于劳务、承揽等民事合同关系的合同效力与履行基础判断。",
        ("劳动关系",),
    ),
    LegalArticle(
        "civil_code_509",
        "民法典第509条",
        "中华人民共和国民法典",
        "民法典",
        "第509条",
        "第五百零九条",
        "合同履行原则",
        "在劳务合同、承揽合同或补充协议评价中，可辅助判断诚信履行义务。",
        ("劳动关系", "工资支付"),
    ),
    LegalArticle(
        "civil_code_577",
        "民法典第577条",
        "中华人民共和国民法典",
        "民法典",
        "第577条",
        "第五百七十七条",
        "违约责任",
        "劳动关系不成立而按民事合同处理时，可辅助评价违约责任。",
        ("服务期违约金", "竞业限制"),
    ),
    LegalArticle(
        "civil_code_1192",
        "民法典第1192条",
        "中华人民共和国民法典",
        "民法典",
        "第1192条",
        "第一千一百九十二条",
        "个人劳务关系责任",
        "用于个人之间劳务关系造成损害时的责任评价，不替代劳动法规则。",
        ("劳动关系",),
    ),
)

ARTICLE_BY_LABEL = {article.label: article for article in LEGAL_ARTICLES}
ARTICLE_BY_ID = {article.article_id: article for article in LEGAL_ARTICLES}

RELATIONSHIP_OPTIONS = ("劳动关系倾向", "劳务/承揽关系倾向", "事实不清")
DISPUTE_OPTIONS = (
    "事实劳动关系/身份争议",
    "未签书面劳动合同",
    "试用期争议",
    "工资拖欠或克扣",
    "加班费争议",
    "违法解除/辞退",
    "经济补偿/赔偿金",
    "社会保险缴纳争议",
    "竞业限制争议",
    "服务期/培训违约金",
    "仲裁时效/程序",
)
KEY_FACT_OPTIONS = (
    "入职/用工时间明确",
    "未签书面劳动合同",
    "工资标准/支付记录明确",
    "考勤/工时记录存在",
    "加班事实或审批记录存在",
    "解除通知或离职原因明确",
    "用人单位规章制度依据",
    "培训费用和服务期约定",
    "竞业限制协议和补偿约定",
    "社会保险缴纳记录",
    "仲裁申请时间",
    "证据不足/事实待补充",
)
ISSUE_KEYWORD_OPTIONS = (
    "劳动关系",
    "书面劳动合同",
    "双倍工资",
    "试用期",
    "工资支付",
    "加班费",
    "违法解除",
    "经济补偿",
    "赔偿金",
    "社会保险",
    "竞业限制",
    "服务期违约金",
    "仲裁时效",
    "举证责任",
)
ADJUDICATION_OPTIONS = (
    "支持劳动者主要请求倾向",
    "支持用人单位抗辩倾向",
    "部分支持/需分项判断",
    "证据不足/需补充事实",
)
BACKGROUND_OPTIONS = (
    "劳动者为孕期/产期/哺乳期女职工",
    "劳动者为工伤/职业病职工",
    "劳动者为高级管理人员",
    "劳动者为实习生/试用期员工",
    "劳动者为劳务派遣用工",
    "已向劳动监察部门投诉",
    "已申请劳动仲裁",
    "涉及多人/集体争议",
    "涉及跨省/跨境用工",
    "无特殊背景信息",
)

LABOR_OBJECTS: tuple[LaborObjectSchema, ...] = (
    LaborObjectSchema(
        "relationship_type",
        "法律关系初筛",
        "single",
        RELATIONSHIP_OPTIONS,
        ("事实不清",),
    ),
    LaborObjectSchema(
        "dispute_focus",
        "核心争议类型",
        "multi",
        DISPUTE_OPTIONS,
        (),
    ),
    LaborObjectSchema(
        "key_fact",
        "已识别关键事实",
        "multi",
        KEY_FACT_OPTIONS,
        ("证据不足/事实待补充",),
    ),
    LaborObjectSchema(
        "issue_keyword",
        "重点解析关键词",
        "multi",
        ISSUE_KEYWORD_OPTIONS,
        (),
    ),
    LaborObjectSchema(
        "article_reference",
        "重点法条方向",
        "multi",
        tuple(article.label for article in LEGAL_ARTICLES),
        (),
    ),
    LaborObjectSchema(
        "adjudication_tendency",
        "裁判/处理倾向初筛",
        "single",
        ADJUDICATION_OPTIONS,
        ("证据不足/需补充事实",),
    ),
    LaborObjectSchema(
        "background",
        "重要背景信息",
        "multi",
        BACKGROUND_OPTIONS,
        ("无特殊背景信息",),
    ),
)

OBJECT_INDEX = {schema.object_id: schema for schema in LABOR_OBJECTS}
OBJECT_OPTIONS = {schema.object_id: tuple(schema.options) for schema in LABOR_OBJECTS}
OBJECT_FACT_SETS = {schema.object_id: set(schema.options) for schema in LABOR_OBJECTS}

ISSUE_ARTICLE_MAP: dict[str, tuple[str, ...]] = {
    "劳动关系": ("劳动合同法第7条", "劳动合同法第10条", "劳动争议调解仲裁法第6条", "民法典第465条", "民法典第1192条"),
    "书面劳动合同": ("劳动合同法第10条", "劳动合同法第14条", "劳动合同法第82条"),
    "双倍工资": ("劳动合同法第10条", "劳动合同法第82条", "劳动争议调解仲裁法第27条"),
    "试用期": ("劳动合同法第19条", "劳动合同法第20条", "劳动合同法第39条"),
    "工资支付": ("劳动合同法第30条", "劳动法第50条", "劳动争议调解仲裁法第27条", "劳动争议调解仲裁法第6条"),
    "加班费": ("劳动合同法第31条", "劳动法第36条", "劳动法第41条", "劳动法第44条", "劳动争议调解仲裁法第6条"),
    "违法解除": ("劳动合同法第39条", "劳动合同法第40条", "劳动合同法第48条", "劳动合同法第87条", "劳动争议调解仲裁法第6条"),
    "经济补偿": ("劳动合同法第40条", "劳动合同法第46条", "劳动合同法第47条"),
    "赔偿金": ("劳动合同法第48条", "劳动合同法第87条", "劳动合同法第47条"),
    "社会保险": ("劳动法第72条",),
    "竞业限制": ("劳动合同法第23条", "劳动合同法第24条", "民法典第143条", "民法典第577条"),
    "服务期违约金": ("劳动合同法第22条", "民法典第143条", "民法典第577条"),
    "仲裁时效": ("劳动争议调解仲裁法第27条",),
    "举证责任": ("劳动争议调解仲裁法第6条", "劳动合同法第39条"),
}

DISPUTE_KEYWORD_MAP: dict[str, tuple[str, ...]] = {
    "事实劳动关系/身份争议": ("劳动关系", "举证责任"),
    "未签书面劳动合同": ("书面劳动合同", "双倍工资"),
    "试用期争议": ("试用期", "工资支付"),
    "工资拖欠或克扣": ("工资支付", "举证责任"),
    "加班费争议": ("加班费", "举证责任"),
    "违法解除/辞退": ("违法解除", "赔偿金", "举证责任"),
    "经济补偿/赔偿金": ("经济补偿", "赔偿金"),
    "社会保险缴纳争议": ("社会保险",),
    "竞业限制争议": ("竞业限制",),
    "服务期/培训违约金": ("服务期违约金",),
    "仲裁时效/程序": ("仲裁时效",),
}

SINGLE_PRIORITY = {
    "relationship_type": {
        "劳动关系倾向": 3,
        "劳务/承揽关系倾向": 2,
        "事实不清": 1,
    },
    "adjudication_tendency": {
        "支持劳动者主要请求倾向": 4,
        "支持用人单位抗辩倾向": 4,
        "部分支持/需分项判断": 3,
        "证据不足/需补充事实": 1,
    },
}


def get_labor_objects() -> list[dict[str, Any]]:
    return [
        {
            "object_id": schema.object_id,
            "label": schema.label,
            "mode": schema.mode,
            "options": list(schema.options),
            "fallback": list(schema.fallback),
        }
        for schema in LABOR_OBJECTS
    ]


def get_legal_articles() -> list[dict[str, Any]]:
    return [article_to_dict(article) for article in LEGAL_ARTICLES]


def article_to_dict(article: LegalArticle | str) -> dict[str, Any]:
    if isinstance(article, str):
        article = ARTICLE_BY_LABEL[article]
    return {
        "article_id": article.article_id,
        "label": article.label,
        "law_name": article.law_name,
        "short_law": article.short_law,
        "article_no": article.article_no,
        "article_no_cn": article.article_no_cn,
        "topic": article.topic,
        "summary": article.summary,
        "issue_keywords": list(article.issue_keywords),
    }


def flatten_raw_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        out: list[str] = []
        for item in value.values():
            out.extend(flatten_raw_value(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(flatten_raw_value(item))
        return out
    return [str(value)]


def normalize_text_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("（", "(").replace("）", ")")
    return normalized


def _dedupe_keep_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _order_by_schema(object_id: str, values: Sequence[str]) -> list[str]:
    order = {fact: index for index, fact in enumerate(OBJECT_OPTIONS.get(object_id, ()))}
    return sorted(_dedupe_keep_order(values), key=lambda fact: order.get(fact, 10_000))


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    normalized = normalize_text_for_match(text)
    return any(normalize_text_for_match(phrase) in normalized for phrase in phrases if phrase)


def _has_regex(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text or "", flags=re.IGNORECASE))


def _extract_exact_options(object_id: str, values: Sequence[str]) -> list[str]:
    options = OBJECT_OPTIONS.get(object_id, ())
    out: list[str] = []
    for raw in values:
        raw_text = str(raw).strip()
        raw_norm = normalize_text_for_match(raw_text)
        for option in options:
            option_norm = normalize_text_for_match(option)
            if raw_norm == option_norm or option_norm in raw_norm:
                out.append(option)
    return _order_by_schema(object_id, out)


def _infer_relationship_from_text(text: str) -> list[str]:
    labor_signals = (
        "入职",
        "用工",
        "上班",
        "岗位",
        "月薪",
        "工资",
        "考勤",
        "打卡",
        "排班",
        "工牌",
        "社保",
        "主管",
        "绩效",
        "公司安排",
        "劳动合同",
        "劳动关系",
        "员工",
    )
    service_signals = (
        "劳务合同",
        "承揽",
        "项目交付",
        "外包",
        "按项目结算",
        "自带工具",
        "个人雇佣",
        "平台接单",
        "自由接单",
    )
    labor_score = sum(1 for signal in labor_signals if _contains_any(text, (signal,)))
    service_score = sum(1 for signal in service_signals if _contains_any(text, (signal,)))
    if _contains_any(text, ("劳动关系", "事实劳动关系", "劳动合同")) or labor_score >= 2:
        return ["劳动关系倾向"]
    if service_score >= 2 and labor_score < 2:
        return ["劳务/承揽关系倾向"]
    return []


def _infer_dispute_focus_from_text(text: str) -> list[str]:
    out: list[str] = []
    normalized = normalize_text_for_match(text)
    if _has_regex(text, r"(未|没|没有|从未|一直没).{0,4}(签|订立|签订).{0,8}(劳动合同|书面合同|合同)") or "双倍工资" in normalized:
        out.append("未签书面劳动合同")
    if _contains_any(text, ("劳动关系", "事实劳动关系", "劳务关系", "承揽", "外包", "用工主体", "兼职关系")):
        out.append("事实劳动关系/身份争议")
    if _contains_any(text, ("试用期", "转正", "录用条件")):
        out.append("试用期争议")
    if _has_regex(text, r"(拖欠|克扣|欠薪|未足额|少发|不发|拒发).{0,8}(工资|提成|奖金|报酬)") or _has_regex(text, r"(工资|提成|奖金).{0,8}(拖欠|克扣|未发|少发|拒发)"):
        out.append("工资拖欠或克扣")
    if _contains_any(text, ("加班费", "延时工资", "休息日加班", "法定节假日加班")) or (
        "加班" in normalized and not _has_regex(text, r"(没有|无|未).{0,3}加班(?!费)")
    ):
        out.append("加班费争议")
    if _contains_any(text, ("违法解除", "辞退", "开除", "解雇", "解除劳动合同", "口头解除", "劝退", "裁员", "不让上班")):
        out.append("违法解除/辞退")
    if _contains_any(text, ("经济补偿", "补偿金", "赔偿金", "n+1", "2n", "双倍赔偿", "代通知金")):
        out.append("经济补偿/赔偿金")
    if _contains_any(text, ("社保", "五险", "养老保险", "医疗保险", "工伤保险", "失业保险", "生育保险")):
        out.append("社会保险缴纳争议")
    if _contains_any(text, ("竞业限制", "竞业协议", "竞业补偿", "竞业违约金")):
        out.append("竞业限制争议")
    if _contains_any(text, ("服务期", "专项培训", "培训费", "培训协议")) or (
        "违约金" in normalized and _contains_any(text, ("培训", "服务期"))
    ):
        out.append("服务期/培训违约金")
    if _contains_any(text, ("劳动仲裁", "仲裁时效", "超过一年", "一年时效", "仲裁申请", "管辖")):
        out.append("仲裁时效/程序")
    return _order_by_schema("dispute_focus", out)


def _infer_key_facts_from_text(text: str) -> list[str]:
    out: list[str] = []
    if _contains_any(text, ("入职", "用工", "上班时间", "工作时间")) or _has_regex(text, r"\d{4}年.{0,6}(入职|开始工作|上班)"):
        out.append("入职/用工时间明确")
    if "未签书面劳动合同" in _infer_dispute_focus_from_text(text):
        out.append("未签书面劳动合同")
    if _contains_any(text, ("工资", "月薪", "薪资", "工资条", "银行流水", "转账记录", "支付记录", "提成", "奖金")):
        out.append("工资标准/支付记录明确")
    if _contains_any(text, ("考勤", "打卡", "排班", "工时", "工牌", "工作群", "值班表")):
        out.append("考勤/工时记录存在")
    if _contains_any(text, ("加班", "加班审批", "值班", "调休", "休息日", "法定节假日")):
        out.append("加班事实或审批记录存在")
    if _contains_any(text, ("辞退", "解除通知", "离职证明", "口头辞退", "不让上班", "劝退", "裁员", "开除")):
        out.append("解除通知或离职原因明确")
    if _contains_any(text, ("员工手册", "规章制度", "严重违纪", "旷工", "录用条件", "民主程序", "公示")):
        out.append("用人单位规章制度依据")
    if _contains_any(text, ("专项培训", "培训费", "服务期", "培训协议", "培训发票")):
        out.append("培训费用和服务期约定")
    if _contains_any(text, ("竞业限制", "竞业协议", "竞业补偿", "保密协议")):
        out.append("竞业限制协议和补偿约定")
    if _contains_any(text, ("社保", "五险", "缴费记录", "社保记录")):
        out.append("社会保险缴纳记录")
    if _contains_any(text, ("劳动仲裁", "仲裁申请", "仲裁时效", "超过一年", "申请时间")):
        out.append("仲裁申请时间")
    if _contains_any(text, ("无证据", "没有证据", "证据不足", "事实不清", "无法证明")):
        out.append("证据不足/事实待补充")
    return _order_by_schema("key_fact", out)


def _infer_issue_keywords_from_parts(
    text: str,
    dispute_focus: Sequence[str] | None = None,
    key_facts: Sequence[str] | None = None,
) -> list[str]:
    out: list[str] = []
    for dispute in dispute_focus or []:
        out.extend(DISPUTE_KEYWORD_MAP.get(dispute, ()))
    facts = set(key_facts or [])
    if "入职/用工时间明确" in facts:
        out.append("劳动关系")
    if "未签书面劳动合同" in facts:
        out.extend(["书面劳动合同", "双倍工资"])
    if "工资标准/支付记录明确" in facts:
        out.append("工资支付")
    if "加班事实或审批记录存在" in facts:
        out.append("加班费")
    if "解除通知或离职原因明确" in facts:
        out.extend(["违法解除", "经济补偿", "赔偿金"])
    if "用人单位规章制度依据" in facts:
        out.extend(["违法解除", "举证责任"])
    if "培训费用和服务期约定" in facts:
        out.append("服务期违约金")
    if "竞业限制协议和补偿约定" in facts:
        out.append("竞业限制")
    if "社会保险缴纳记录" in facts:
        out.append("社会保险")
    if "仲裁申请时间" in facts:
        out.append("仲裁时效")
    exact = _extract_exact_options("issue_keyword", flatten_raw_value(text))
    out.extend(exact)
    return _order_by_schema("issue_keyword", out)


def _infer_articles_from_keywords(keywords: Sequence[str]) -> list[str]:
    out: list[str] = []
    for keyword in keywords:
        out.extend(ISSUE_ARTICLE_MAP.get(keyword, ()))
    return _order_by_schema("article_reference", out)


def _article_matches_text(article: LegalArticle, text: str) -> bool:
    normalized = normalize_text_for_match(text)
    label_norm = normalize_text_for_match(article.label)
    if label_norm and label_norm in normalized:
        return True
    law_terms = (
        article.short_law,
        article.law_name,
        article.law_name.replace("中华人民共和国", ""),
    )
    article_terms = (article.article_no, article.article_no_cn)
    has_law = any(normalize_text_for_match(term) in normalized for term in law_terms)
    has_article = any(normalize_text_for_match(term) in normalized for term in article_terms)
    return has_law and has_article


def _extract_article_labels_from_text(text: str) -> list[str]:
    out = [article.label for article in LEGAL_ARTICLES if _article_matches_text(article, text)]
    return _order_by_schema("article_reference", out)


def _infer_adjudication_from_text(
    text: str,
    relationship: Sequence[str],
    dispute_focus: Sequence[str],
    key_facts: Sequence[str],
) -> list[str]:
    if _contains_any(text, ("证据不足", "无证据", "事实不清", "无法证明")):
        return ["证据不足/需补充事实"]

    employee_score = 0
    employer_score = 0
    disputes = set(dispute_focus)
    facts = set(key_facts)

    if "劳动关系倾向" in relationship:
        employee_score += 1
    if "未签书面劳动合同" in disputes:
        employee_score += 2
    if "工资拖欠或克扣" in disputes:
        employee_score += 2
    if "违法解除/辞退" in disputes and "解除通知或离职原因明确" in facts:
        employee_score += 2
    if "竞业限制争议" in disputes and _contains_any(text, ("未支付竞业补偿", "没有竞业补偿", "不给竞业补偿")):
        employee_score += 2
    if "服务期/培训违约金" in disputes and _contains_any(text, ("违约金过高", "全额赔偿", "培训费不明确")):
        employee_score += 1

    if _contains_any(text, ("严重违纪", "旷工", "不符合录用条件", "员工主动离职", "个人原因离职", "协商一致", "规章制度")):
        employer_score += 2
    if "用人单位规章制度依据" in facts and "违法解除/辞退" in disputes:
        employer_score += 1
    if "劳务/承揽关系倾向" in relationship:
        employer_score += 1

    if employee_score >= 2 and employer_score >= 2:
        return ["部分支持/需分项判断"]
    if employee_score >= 2:
        return ["支持劳动者主要请求倾向"]
    if employer_score >= 2:
        return ["支持用人单位抗辩倾向"]
    if disputes or facts:
        return ["部分支持/需分项判断"]
    return []


def _empty_patch_map() -> dict[str, list[str]]:
    return {schema.object_id: [] for schema in LABOR_OBJECTS}


def _empty_field_status() -> dict[str, dict[str, bool]]:
    return {
        schema.object_id: {"missing": False, "used_fallback": False}
        for schema in LABOR_OBJECTS
    }


def _truncate_candidates(
    object_id: str,
    facts: Sequence[str],
    max_candidates_per_object: int | None,
) -> list[str]:
    ordered = _order_by_schema(object_id, facts)
    if max_candidates_per_object is None:
        return ordered
    return ordered[: max(0, int(max_candidates_per_object))]


def _update_single_select(object_id: str, facts: Sequence[str]) -> list[str]:
    valid = [fact for fact in _dedupe_keep_order(facts) if fact in OBJECT_FACT_SETS.get(object_id, set())]
    if not valid:
        return []
    priority = SINGLE_PRIORITY.get(object_id, {})
    return [max(valid, key=lambda fact: priority.get(fact, 0))]


def _merge_patch_map_into_normalized(
    normalized: dict[str, list[str]],
    patch_map: Mapping[str, Sequence[str]],
) -> None:
    for schema in LABOR_OBJECTS:
        current = list(normalized.get(schema.object_id, []))
        incoming = [fact for fact in patch_map.get(schema.object_id, []) if fact in OBJECT_FACT_SETS[schema.object_id]]
        if schema.mode == "single":
            normalized[schema.object_id] = _update_single_select(schema.object_id, current + incoming)
        else:
            normalized[schema.object_id] = _order_by_schema(schema.object_id, current + incoming)


def _infer_background_from_text(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    result: list[str] = []
    if re.search(r"(怀孕|孕期|产期|哺乳期|产妇|孕妇)", t):
        result.append("劳动者为孕期/产期/哺乳期女职工")
    if re.search(r"(工伤|职业病|伤残等级|劳动能力鉴定|工伤认定)", t):
        result.append("劳动者为工伤/职业病职工")
    if re.search(r"(高级管理人员|高管|总监|总经理|副总|部门经理)", t):
        result.append("劳动者为高级管理人员")
    if re.search(r"(实习|实习生|试用期员工|试用期内|试用期限)", t):
        result.append("劳动者为实习生/试用期员工")
    if re.search(r"(劳务派遣|派遣用工|被派遣)", t):
        result.append("劳动者为劳务派遣用工")
    if re.search(r"(劳动监察|已投诉|已举报|向.*投诉|投诉至)", t):
        result.append("已向劳动监察部门投诉")
    if re.search(r"(已申请仲裁|已提起仲裁|仲裁立案|仲裁申请|已仲裁)", t):
        result.append("已申请劳动仲裁")
    if re.search(r"(多人|集体争议|群体|多名劳动者|数名员工)", t):
        result.append("涉及多人/集体争议")
    if re.search(r"(跨省|跨境|境外|涉外|外国人|台港澳)", t):
        result.append("涉及跨省/跨境用工")
    if not result:
        result.append("无特殊背景信息")
    return result


def _derive_patch_from_text(text: str) -> dict[str, list[str]]:
    patch = _empty_patch_map()
    relationship = _infer_relationship_from_text(text)
    dispute_focus = _infer_dispute_focus_from_text(text)
    key_facts = _infer_key_facts_from_text(text)
    issue_keywords = _infer_issue_keywords_from_parts(text, dispute_focus, key_facts)
    article_refs = _order_by_schema(
        "article_reference",
        _extract_article_labels_from_text(text) + _infer_articles_from_keywords(issue_keywords),
    )
    adjudication = _infer_adjudication_from_text(text, relationship, dispute_focus, key_facts)
    background = _infer_background_from_text(text)

    patch["relationship_type"] = relationship
    patch["dispute_focus"] = dispute_focus
    patch["key_fact"] = key_facts
    patch["issue_keyword"] = issue_keywords
    patch["article_reference"] = article_refs
    patch["adjudication_tendency"] = adjudication
    patch["background"] = background
    return patch


def normalize_labor_fact(
    object_id: str,
    raw_value: Any,
    *,
    user_text: str = "",
) -> list[str]:
    if object_id not in OBJECT_INDEX:
        raise ValueError(f"Unknown labor object_id: {object_id}")
    schema = OBJECT_INDEX[object_id]
    texts = flatten_raw_value(raw_value)
    joined = "。".join(str(item) for item in texts if str(item).strip())
    if user_text:
        joined = (joined + "。" + user_text).strip("。")

    exact = _extract_exact_options(object_id, texts)
    inferred_patch = _derive_patch_from_text(joined)
    inferred = inferred_patch.get(object_id, [])
    facts = _dedupe_keep_order(exact + inferred)

    if schema.mode == "single":
        facts = _update_single_select(object_id, facts)
    else:
        facts = _order_by_schema(object_id, facts)
    return facts


def normalize_model_labor_output(
    model_output: Any,
    *,
    user_text: str = "",
    max_candidates_per_object: int | None = None,
    bert_processor: Any | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    case_summary = ""
    raw_structured_analysis: dict[str, Any] = {}

    if isinstance(model_output, str):
        try:
            parsed = json.loads(model_output)
        except Exception:
            parsed = None
        model_output = parsed if isinstance(parsed, Mapping) else {"case_summary": model_output}

    if isinstance(model_output, Mapping):
        summary_value = model_output.get("case_summary") or model_output.get("user_explanation")
        if summary_value is not None:
            case_summary = " ".join(flatten_raw_value(summary_value)).strip()

        structured_value = model_output.get("structured_analysis")
        if isinstance(structured_value, Mapping):
            raw_structured_analysis = dict(structured_value)
        else:
            top_level_fields = {
                schema.object_id: model_output[schema.object_id]
                for schema in LABOR_OBJECTS
                if schema.object_id in model_output
            }
            if top_level_fields:
                raw_structured_analysis = top_level_fields
                warnings.append("structured_analysis missing or not a dict; used top-level labor fields")
            else:
                warnings.append("structured_analysis missing or not a dict")
    else:
        case_summary = " ".join(flatten_raw_value(model_output)).strip()
        warnings.append("model_output is not a dict; structured_analysis unavailable")

    from_model_fields = _empty_patch_map()
    from_user_text = _derive_patch_from_text(user_text)
    field_status = _empty_field_status()
    normalized = _empty_patch_map()
    rows: list[dict[str, Any]] = []

    bert_matches: dict[str, Any] = {}
    if bert_processor is not None:
        try:
            from labor_law_app.bert_output_processor import BERTOutputProcessor
            output_proc = BERTOutputProcessor(bert_processor)
            bert_match_results = output_proc.match_all_objects(
                {"structured_analysis": raw_structured_analysis}
                if raw_structured_analysis
                else {},
                model_name="",
            )
            for oid, match_result in bert_match_results.items():
                bert_matches[oid] = {
                    "best_match": match_result.best_match.option if match_result.best_match else None,
                    "best_confidence": match_result.best_match.confidence if match_result.best_match else 0.0,
                    "selected": [
                        {"option": m.option, "similarity": round(m.similarity, 4), "confidence": round(m.confidence, 4)}
                        for m in match_result.selected_matches
                    ],
                    "all_matches": [
                        {"option": m.option, "similarity": round(m.similarity, 4), "confidence": round(m.confidence, 4)}
                        for m in match_result.matches[:5]
                    ],
                }
        except Exception:
            pass

    for schema in LABOR_OBJECTS:
        raw_value = raw_structured_analysis.get(schema.object_id)
        is_missing = schema.object_id not in raw_structured_analysis
        if is_missing:
            warnings.append(f"Missing field: {schema.object_id}")
        model_facts = normalize_labor_fact(schema.object_id, raw_value, user_text="")
        used_fallback = bool(is_missing and schema.fallback and not model_facts)
        if used_fallback:
            model_facts = list(schema.fallback)

        if bert_matches and schema.object_id in bert_matches:
            bert_match_data = bert_matches[schema.object_id]
            bert_facts = [m["option"] for m in bert_match_data.get("selected", [])]
            merged = list(model_facts)
            for bf in bert_facts:
                if bf not in merged:
                    merged.append(bf)
            if schema.mode == "single":
                merged = merged[:2]
            model_facts = merged

        field_status[schema.object_id]["missing"] = is_missing
        field_status[schema.object_id]["used_fallback"] = used_fallback
        from_model_fields[schema.object_id] = list(model_facts)
        normalized[schema.object_id] = list(model_facts)
        rows.append(
            {
                "object_id": schema.object_id,
                "object_label": schema.label,
                "raw_value": raw_value,
                "model_field_facts": list(model_facts),
                "user_text_patch_facts": list(from_user_text.get(schema.object_id, [])),
                "final_normalized_facts": list(model_facts),
                "normalized_facts": list(model_facts),
            }
        )

    _merge_patch_map_into_normalized(normalized, from_user_text)
    derived_keywords = _infer_issue_keywords_from_parts(
        "。".join([case_summary, user_text]),
        normalized.get("dispute_focus", []),
        normalized.get("key_fact", []),
    )
    normalized["issue_keyword"] = _order_by_schema(
        "issue_keyword",
        normalized.get("issue_keyword", []) + derived_keywords,
    )
    normalized["article_reference"] = _order_by_schema(
        "article_reference",
        normalized.get("article_reference", []) + _infer_articles_from_keywords(normalized["issue_keyword"]),
    )
    if not normalized["adjudication_tendency"]:
        normalized["adjudication_tendency"] = _infer_adjudication_from_text(
            "。".join([case_summary, user_text]),
            normalized.get("relationship_type", []),
            normalized.get("dispute_focus", []),
            normalized.get("key_fact", []),
        )

    for schema in LABOR_OBJECTS:
        facts = normalized.get(schema.object_id, [])
        if schema.mode == "single":
            facts = _update_single_select(schema.object_id, facts)
        else:
            facts = _truncate_candidates(schema.object_id, facts, max_candidates_per_object)
        if schema.mode == "single" and not facts:
            facts = list(schema.fallback)
        if schema.object_id == "key_fact" and not facts:
            facts = list(schema.fallback)
        normalized[schema.object_id] = facts

    for row in rows:
        final_facts = list(normalized[row["object_id"]])
        row["final_normalized_facts"] = final_facts
        row["normalized_facts"] = final_facts

    # Grounding Guard: filter unsupported facts
    grounding_warnings: list[dict[str, Any]] = []
    try:
        from labor_law_app.grounding_guard import filter_unsupported_facts
        filtered_normalized, grounding_warnings = filter_unsupported_facts(
            normalized,
            user_text or "",
            source_name="normalize",
        )
        normalized = filtered_normalized
        for row in rows:
            row["final_normalized_facts"] = list(normalized.get(row["object_id"], []))
            row["normalized_facts"] = list(normalized.get(row["object_id"], []))
    except ImportError:
        pass

    return {
        "case_summary": case_summary,
        "raw_structured_analysis": raw_structured_analysis,
        "normalized": normalized,
        "rows": rows,
        "field_status": field_status,
        "patches": {
            "from_model_fields": from_model_fields,
            "from_user_text": from_user_text,
            "derived_article_refs": {"article_reference": _infer_articles_from_keywords(normalized.get("issue_keyword", []))},
        },
        "warnings": _dedupe_keep_order(warnings),
        "grounding_warnings": grounding_warnings,
    }


def normalize_all_models_labor_outputs(
    model_outputs: Mapping[str, Any],
    *,
    user_text: str = "",
    bert_processor: Any | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        model_name: normalize_model_labor_output(
            model_output, user_text=user_text, bert_processor=bert_processor
        )
        for model_name, model_output in model_outputs.items()
    }


def build_labor_fact_table(
    normalized_all: Mapping[str, Mapping[str, Any]],
    source: str = "normalized",
    exclude_fallbacks: bool = False,
) -> list[dict[str, Any]]:
    if source not in {"normalized", "from_model_fields", "from_user_text"}:
        raise ValueError(f"Invalid labor fact table source: {source}")
    table: list[dict[str, Any]] = []
    for model_name, result in normalized_all.items():
        if source == "normalized":
            fact_map = result.get("normalized", {})
        else:
            fact_map = (result.get("patches", {}) or {}).get(source, {})
        field_status = result.get("field_status", {}) or {}
        for schema in LABOR_OBJECTS:
            facts = list((fact_map or {}).get(schema.object_id, []))
            if (
                source == "from_model_fields"
                and exclude_fallbacks
                and bool((field_status.get(schema.object_id, {}) or {}).get("used_fallback"))
            ):
                facts = []
            table.append(
                {
                    "model": model_name,
                    "object_id": schema.object_id,
                    "object_label": schema.label,
                    "facts": facts,
                }
            )
    return table


def build_issue_article_links(normalized: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    issue_keywords = list(normalized.get("issue_keyword", []) or [])
    selected_articles = set(normalized.get("article_reference", []) or [])
    links: list[dict[str, Any]] = []
    for keyword in issue_keywords:
        candidate_labels = ISSUE_ARTICLE_MAP.get(keyword, ())
        article_labels = [label for label in candidate_labels if not selected_articles or label in selected_articles]
        if not article_labels:
            continue
        links.append(
            {
                "issue_keyword": keyword,
                "problem_to_check": _problem_text_for_keyword(keyword),
                "article_refs": [article_to_dict(label) for label in article_labels],
                "why": _why_text_for_keyword(keyword),
            }
        )
    return links


def _problem_text_for_keyword(keyword: str) -> str:
    mapping = {
        "劳动关系": "先确认是否存在受用人单位管理、领取劳动报酬、持续提供劳动等劳动关系要素。",
        "书面劳动合同": "核对是否在法定期限内订立书面劳动合同，以及是否存在无固定期限劳动合同触发条件。",
        "双倍工资": "审查未签书面劳动合同的期间、责任主体、仲裁时效和工资基数。",
        "试用期": "核对试用期期限、工资标准、约定次数及解除理由。",
        "工资支付": "核对工资标准、支付周期、拖欠或扣减依据及支付记录。",
        "加班费": "核对工时制度、加班事实、审批或考勤证据以及加班费计算标准。",
        "违法解除": "核对解除事由、通知程序、规章制度依据和用人单位举证材料。",
        "经济补偿": "核对解除/终止原因是否落入经济补偿法定情形。",
        "赔偿金": "若解除违法，进一步核算赔偿金与经济补偿基数。",
        "社会保险": "核对社保缴纳记录、补缴路径和劳动争议/行政处理边界。",
        "竞业限制": "核对竞业对象、范围、期限、补偿支付和违约金合理性。",
        "服务期违约金": "核对是否有专项培训费用、服务期约定及违约金上限。",
        "仲裁时效": "核对权利被侵害之日、劳动关系存续状态和仲裁申请日期。",
        "举证责任": "确认关键证据由哪一方掌握，特别是考勤、工资和解除依据。",
    }
    return mapping.get(keyword, "围绕该争议关键词补充事实和证据。")


def _why_text_for_keyword(keyword: str) -> str:
    mapping = {
        "劳动关系": "劳动关系成立是适用劳动法体系的前提，也影响请求权基础。",
        "书面劳动合同": "未签书面劳动合同通常直接关联双倍工资和无固定期限劳动合同风险。",
        "双倍工资": "该请求对起止期间、工资基数和时效较敏感。",
        "试用期": "试用期违法常影响解除合法性和工资差额。",
        "工资支付": "拖欠、克扣工资通常需要工资标准和支付流水交叉验证。",
        "加班费": "加班费争议高度依赖工时制度、考勤和审批证据。",
        "违法解除": "解除类争议的裁判重点通常落在事实依据、制度依据和程序。",
        "经济补偿": "经济补偿需先识别解除或终止原因，再进入年限和工资基数计算。",
        "赔偿金": "赔偿金通常以违法解除为前提，与经济补偿标准联动。",
        "社会保险": "社保争议可能涉及劳动争议与行政处理的路径选择。",
        "竞业限制": "竞业限制的人员范围、期限、补偿和违约金均需逐项审查。",
        "服务期违约金": "服务期违约金通常受专项培训费用和未履行服务期比例限制。",
        "仲裁时效": "时效抗辩可能直接影响请求能否获得支持。",
        "举证责任": "证据掌握方和举证不能后果会影响事实认定。",
    }
    return mapping.get(keyword, "用于把事实问题和法条筛选结果对应起来。")


def extract_case_fact_patch(user_text: str) -> dict[str, list[str]]:
    return _derive_patch_from_text(user_text or "")


__all__ = [
    "ARTICLE_BY_ID",
    "ARTICLE_BY_LABEL",
    "ISSUE_ARTICLE_MAP",
    "LABOR_OBJECTS",
    "LEGAL_ARTICLES",
    "LaborObjectSchema",
    "LegalArticle",
    "article_to_dict",
    "build_issue_article_links",
    "build_labor_fact_table",
    "extract_case_fact_patch",
    "flatten_raw_value",
    "get_labor_objects",
    "get_legal_articles",
    "normalize_all_models_labor_outputs",
    "normalize_labor_fact",
    "normalize_model_labor_output",
    "normalize_text_for_match",
]

