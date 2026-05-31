from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from typing import Any, Mapping

from labor_law_app.labor_truthfinder import (
    explain_truth_per_labor_object,
    labor_truthfinder_run,
    rank_models_by_trust,
)
from labor_law_app.normalize_labor import (
    ARTICLE_BY_LABEL,
    build_issue_article_links,
    extract_case_fact_patch,
    get_labor_objects,
    get_legal_articles,
    normalize_all_models_labor_outputs,
)


DISCLAIMER = (
    "本接口输出为律师办案和法律检索场景下的AI辅助事实整理、争议点归纳和法条筛选结果，"
    "不构成正式法律意见、裁判预测或对案件结果的保证；引用法条请以官方公布的最新有效文本为准。"
)


def get_api_schema() -> dict[str, Any]:
    return {
        "endpoint": {
            "method": "POST",
            "path": "/analyze",
            "content_type": "application/json",
        },
        "input": {
            "case_id": "可选，案件编号；未提供时默认 case_0",
            "case_text": "必填，自然语言案情、合同条款、证据摘要或律师询问笔记",
            "model_outputs": {
                "optional": True,
                "description": "多模型结构化输出，推荐 3-4 个来源；未提供时使用规则抽取基线",
                "shape": {
                    "模型名": {
                        "case_summary": "模型对案情的简短总结",
                        "structured_analysis": {
                            "relationship_type": "法律关系初筛，single",
                            "dispute_focus": "核心争议类型，multi",
                            "key_fact": "已识别关键事实，multi",
                            "issue_keyword": "重点解析关键词，multi",
                            "article_reference": "重点法条方向，multi",
                            "adjudication_tendency": "裁判/处理倾向初筛，single",
                        },
                    }
                },
            },
            "truthfinder_source": "可选，默认 from_model_fields；也可传 normalized/from_user_text",
        },
        "objects": get_labor_objects(),
        "legal_articles": get_legal_articles(),
        "output": {
            "case_facts": "从用户原文抽取的标准 facts",
            "normalized_by_source": "各模型/来源归一化结果",
            "truthfinder": "模型可信度、各 object 聚合候选和 debug 摘要",
            "issue_keywords": "重点解析关键词及置信度",
            "article_candidates": "法条候选、置信度、触发来源和摘要",
            "issue_article_matrix": "哪些问题对应哪些法条",
            "evidence_gaps": "建议律师补充核验的证据点",
            "lawyer_brief": "给律师的案件要点摘要",
        },
        "disclaimer": DISCLAIMER,
    }


def build_labor_prompt(case_text: str) -> str:
    object_blocks = []
    for item in get_labor_objects():
        options = "\n".join(f"- {option}" for option in item["options"])
        plural_hint = "可以多个" if item["mode"] == "multi" else "只能选一个"
        object_blocks.append(
            f"{item['object_id']}（{item['label']}，{plural_hint}）:\n{options}"
        )
    object_text = "\n\n".join(object_blocks)
    return f"""
你是面向执业律师的劳动争议案情结构化助手。请只输出合法 JSON，不要输出 Markdown、代码块或解释性前缀。

任务：围绕在职场景下的劳务/劳动合同纠纷，整理事实、争议点、重点关键词、裁判处理倾向初筛，并筛选相关法条。

严格要求：
1. 不输出正式法律意见书，不保证案件结果。
2. 不能直接说“必胜”“必败”，只能输出初筛倾向和需核验事实。
3. structured_analysis 的字段值必须尽量从候选集合中选择。
4. article_reference 只放需要重点检索的法条标签。
5. 重点解释哪些事实可能有问题，以及对应哪些法条。

六个 object 与候选集合如下：

{object_text}

输出 JSON 格式：
{{
  "case_summary": "用2-4句话概括案情和关键争议",
  "structured_analysis": {{
    "relationship_type": "只能选一个",
    "dispute_focus": ["可以多个"],
    "key_fact": ["可以多个"],
    "issue_keyword": ["可以多个"],
    "article_reference": ["可以多个"],
    "adjudication_tendency": "只能选一个"
  }},
  "issue_analysis": [
    {{
      "issue_keyword": "例如：违法解除",
      "problem": "该问题在本案中具体体现在哪里",
      "facts_to_check": ["需要核验的事实或证据"],
      "article_reference": ["对应法条标签"],
      "reasoning_note": "简短说明"
    }}
  ],
  "evidence_gaps": ["仍需补充的证据"]
}}

案情/材料：
\"\"\"{case_text.strip()}\"\"\"
""".strip()


def try_parse_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```json\s*(\{.*?})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        try:
            return json.loads(text[left : right + 1])
        except Exception:
            return None
    return None


def parse_labor_model_output(raw_output: Any) -> dict[str, Any]:
    if isinstance(raw_output, Mapping):
        structured = raw_output.get("structured_analysis", {})
        if not isinstance(structured, Mapping):
            structured = {}
        return {
            "ok": True,
            "raw_output": raw_output,
            "case_summary": str(raw_output.get("case_summary", "") or raw_output.get("user_explanation", "") or "").strip(),
            "structured_analysis": dict(structured),
            "issue_analysis": raw_output.get("issue_analysis", []),
            "evidence_gaps": raw_output.get("evidence_gaps", []),
        }

    parsed = try_parse_json(str(raw_output or ""))
    if not isinstance(parsed, Mapping):
        return {
            "ok": False,
            "raw_output": raw_output,
            "case_summary": str(raw_output or "").strip(),
            "structured_analysis": {},
            "issue_analysis": [],
            "evidence_gaps": [],
            "error": "无法解析为合法 JSON，已按自然语言摘要处理",
        }
    return parse_labor_model_output(parsed)


def _coerce_model_outputs(payload: Mapping[str, Any], case_text: str) -> tuple[dict[str, Any], bool]:
    raw_outputs = payload.get("model_outputs")
    if isinstance(raw_outputs, Mapping) and raw_outputs:
        return {
            str(model_name): parse_labor_model_output(model_payload)
            for model_name, model_payload in raw_outputs.items()
        }, False

    if isinstance(raw_outputs, list) and raw_outputs:
        outputs: dict[str, Any] = {}
        for index, item in enumerate(raw_outputs, start=1):
            if isinstance(item, Mapping):
                model_name = str(item.get("model") or item.get("name") or f"model_{index}")
                outputs[model_name] = parse_labor_model_output(item.get("output", item))
            else:
                outputs[f"model_{index}"] = parse_labor_model_output(item)
        return outputs, False

    patch = extract_case_fact_patch(case_text)
    return {
        "rule_based_extractor": {
            "ok": True,
            "raw_output": {"case_summary": "规则抽取基线", "structured_analysis": patch},
            "case_summary": "未提供多模型输出，接口使用规则抽取基线生成可联调结果。",
            "structured_analysis": patch,
            "issue_analysis": [],
            "evidence_gaps": [],
        }
    }, True


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {_json_key(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _json_key(value: Any) -> str:
    if isinstance(value, tuple):
        return " :: ".join(_json_key(item) for item in value)
    return str(value)


def _row_by_object(truth_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("object_id")): row for row in truth_rows}


def _confidence_lookup(row: dict[str, Any]) -> dict[str, float]:
    return {
        str(candidate.get("fact")): float(candidate.get("confidence", 0.0))
        for candidate in row.get("candidates", []) or []
    }


def _selected_with_conf(row: dict[str, Any], fallback_facts: list[str]) -> list[dict[str, Any]]:
    conf = _confidence_lookup(row)
    selected = list(row.get("selected_facts", []) or [])
    if not selected:
        selected = fallback_facts
    out = []
    for fact in selected:
        if not fact or fact == "(空)":
            continue
        out.append({"fact": fact, "confidence": float(conf.get(fact, 0.50))})
    return out


def _build_article_candidates(
    truth_rows: list[dict[str, Any]],
    case_patch: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    row_map = _row_by_object(truth_rows)
    article_row = row_map.get("article_reference", {})
    selected_articles = _selected_with_conf(article_row, list(case_patch.get("article_reference", []) or []))
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for item in selected_articles:
        label = item["fact"]
        article = ARTICLE_BY_LABEL.get(label)
        if not article or label in seen:
            continue
        seen.add(label)
        payload = {
            "label": label,
            "confidence": round(float(item["confidence"]), 4),
            "source": "truthfinder" if article_row else "case_rule_map",
            "match_reason": "由多模型/规则结构化事实聚合后入选",
            **_article_payload(article),
        }
        candidates.append(payload)

    for label in case_patch.get("article_reference", []) or []:
        if label in seen:
            continue
        article = ARTICLE_BY_LABEL.get(label)
        if not article:
            continue
        seen.add(label)
        candidates.append(
            {
                "label": label,
                "confidence": 0.5,
                "source": "case_rule_map",
                "match_reason": "由用户原文关键词和争议类型规则触发，建议律师复核",
                **_article_payload(article),
            }
        )
    return candidates


def _article_payload(article) -> dict[str, Any]:
    return {
        "article_id": article.article_id,
        "law_name": article.law_name,
        "short_law": article.short_law,
        "article_no": article.article_no,
        "article_no_cn": article.article_no_cn,
        "topic": article.topic,
        "summary": article.summary,
        "issue_keywords": list(article.issue_keywords),
    }


def _build_issue_keywords(
    truth_rows: list[dict[str, Any]],
    case_patch: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    row = _row_by_object(truth_rows).get("issue_keyword", {})
    selected = _selected_with_conf(row, list(case_patch.get("issue_keyword", []) or []))
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in selected:
        keyword = item["fact"]
        if keyword in seen:
            continue
        seen.add(keyword)
        out.append({"keyword": keyword, "confidence": round(float(item["confidence"]), 4)})
    for keyword in case_patch.get("issue_keyword", []) or []:
        if keyword not in seen:
            seen.add(keyword)
            out.append({"keyword": keyword, "confidence": 0.5})
    return out


def _build_combined_normalized(
    truth_rows: list[dict[str, Any]],
    case_patch: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    combined = {key: list(value or []) for key, value in case_patch.items()}
    for row in truth_rows:
        object_id = str(row.get("object_id"))
        selected = [fact for fact in row.get("selected_facts", []) or [] if fact and fact != "(空)"]
        if selected:
            existing = combined.get(object_id, [])
            combined[object_id] = _dedupe(existing + selected)
    return combined


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _build_evidence_gaps(combined: Mapping[str, list[str]]) -> list[str]:
    issues = set(combined.get("issue_keyword", []) or [])
    facts = set(combined.get("key_fact", []) or [])
    gaps: list[str] = []

    if "劳动关系" in issues and not {"工资标准/支付记录明确", "考勤/工时记录存在"} & facts:
        gaps.append("补充劳动关系证据：工资流水、考勤打卡、工牌、工作群安排、社保记录、岗位/管理关系材料。")
    if {"书面劳动合同", "双倍工资"} & issues:
        if "入职/用工时间明确" not in facts:
            gaps.append("补充入职或实际用工起算时间，用于确定未签书面劳动合同责任期间。")
        if "工资标准/支付记录明确" not in facts:
            gaps.append("补充工资基数证据，用于核算双倍工资差额。")
    if "工资支付" in issues and "工资标准/支付记录明确" not in facts:
        gaps.append("补充工资约定、工资条、银行流水、欠薪期间和已付款明细。")
    if "加班费" in issues and not {"考勤/工时记录存在", "加班事实或审批记录存在"} & facts:
        gaps.append("补充考勤、排班、加班审批、工作记录或聊天记录，用于证明加班事实和时长。")
    if {"违法解除", "经济补偿", "赔偿金"} & issues:
        if "解除通知或离职原因明确" not in facts:
            gaps.append("补充解除通知、离职证明、沟通记录，明确解除主体、时间和理由。")
        if "用人单位规章制度依据" not in facts:
            gaps.append("核验用人单位主张解除依据时，要求提供员工手册、规章制度、民主程序和送达/公示证据。")
    if "竞业限制" in issues and "竞业限制协议和补偿约定" not in facts:
        gaps.append("补充竞业限制协议、补偿支付记录、限制期限/地域/业务范围和违约金约定。")
    if "服务期违约金" in issues and "培训费用和服务期约定" not in facts:
        gaps.append("补充专项培训协议、培训费用凭证、服务期起止和已履行期间。")
    if "仲裁时效" in issues and "仲裁申请时间" not in facts:
        gaps.append("补充仲裁申请日期、争议发生日期、劳动关系解除或终止日期，用于判断时效。")
    if not gaps:
        gaps.append("继续核对原始合同、工资流水、考勤、解除沟通记录和仲裁材料，确保事实链闭合。")
    return _dedupe(gaps)


def _build_lawyer_brief(
    combined: Mapping[str, list[str]],
    issue_keywords: list[dict[str, Any]],
    article_candidates: list[dict[str, Any]],
    evidence_gaps: list[str],
) -> dict[str, Any]:
    tendency = (combined.get("adjudication_tendency", []) or ["证据不足/需补充事实"])[0]
    return {
        "core_disputes": combined.get("dispute_focus", []),
        "key_facts": combined.get("key_fact", []),
        "priority_keywords": [item["keyword"] for item in issue_keywords],
        "initial_tendency": tendency,
        "priority_articles": [
            {
                "label": item["label"],
                "topic": item["topic"],
                "confidence": item["confidence"],
            }
            for item in article_candidates[:10]
        ],
        "next_steps": evidence_gaps[:6],
    }


def _get_bert_processor():
    try:
        from labor_law_app.bert_processor import BERTProcessor
        proc = BERTProcessor.get_instance()
        if not proc.is_loaded:
            proc._ensure_loaded()
        return proc
    except Exception:
        return None


def analyze_labor_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    case_text = str(
        payload.get("case_text")
        or payload.get("user_text")
        or payload.get("description")
        or ""
    ).strip()
    if not case_text:
        return {
            "ok": False,
            "error": "case_text is required",
            "schema": get_api_schema(),
        }

    case_id = str(payload.get("case_id") or "case_0")
    model_outputs, used_rule_baseline = _coerce_model_outputs(payload, case_text)
    models = list(model_outputs.keys())

    use_bert = bool(payload.get("use_bert", True))
    bert = _get_bert_processor() if use_bert else None
    bert_available = bert is not None and bert.is_loaded if bert else False

    normalized_all = normalize_all_models_labor_outputs(
        {
            model: {
                "case_summary": (model_outputs.get(model, {}) or {}).get("case_summary", ""),
                "structured_analysis": (model_outputs.get(model, {}) or {}).get("structured_analysis", {}),
            }
            for model in models
        },
        user_text=case_text,
        bert_processor=bert if bert_available else None,
    )

    source = str(payload.get("truthfinder_source") or "from_model_fields")
    exclude_fallbacks = bool(payload.get("exclude_fallbacks", True))
    t_score, s_score, cand_map, debug_info = labor_truthfinder_run(
        models=models,
        case_id=case_id,
        normalized_all=normalized_all,
        source=source,
        exclude_fallbacks=exclude_fallbacks,
        support_mode=str(payload.get("support_mode") or "multi"),
        return_debug=True,
    )
    truth_rows = explain_truth_per_labor_object(
        case_id=case_id,
        s_score=s_score,
        cand_map=cand_map,
        support=debug_info.get("support"),
    )

    case_patch = extract_case_fact_patch(case_text)
    combined = _build_combined_normalized(truth_rows, case_patch)
    issue_keywords = _build_issue_keywords(truth_rows, case_patch)
    article_candidates = _build_article_candidates(truth_rows, case_patch)
    issue_article_matrix = build_issue_article_links(combined)
    evidence_gaps = _build_evidence_gaps(combined)
    lawyer_brief = _build_lawyer_brief(
        combined,
        issue_keywords,
        article_candidates,
        evidence_gaps,
    )

    bert_analysis: dict[str, Any] = {"bert_available": bert_available}
    if bert_available and bert is not None:
        try:
            from labor_law_app.bert_input_processor import BERTInputProcessor
            input_proc = BERTInputProcessor(bert)
            profile = input_proc.analyze_case(case_text)
            bert_analysis["semantic_case_profile"] = profile.to_dict()
            bert_analysis["unified_prompt"] = input_proc.build_unified_prompt(case_text)
        except Exception:
            bert_analysis["semantic_case_profile"] = None

        try:
            from labor_law_app.bert_report_generator import BERTReportGenerator
            report_gen = BERTReportGenerator()
            comprehensive_report = report_gen.generate_comprehensive_report(
                case_id=case_id,
                case_text=case_text,
                truth_rows=truth_rows,
                t_score=t_score,
                effective_trust=debug_info.get("effective_trust"),
                model_coverage=debug_info.get("model_coverage"),
                change_history=debug_info.get("change_history"),
                article_candidates=article_candidates,
                evidence_gaps=evidence_gaps,
            )
            bert_analysis["comprehensive_report"] = comprehensive_report.to_dict()
        except Exception:
            bert_analysis["comprehensive_report"] = None

    return {
        "ok": True,
        "case_id": case_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "used_rule_baseline": used_rule_baseline,
        "input": {
            "case_text": case_text,
            "models": models,
            "truthfinder_source": source,
            "exclude_fallbacks": exclude_fallbacks,
        },
        "case_facts": case_patch,
        "normalized_by_source": {
            model: {
                "case_summary": result.get("case_summary", ""),
                "normalized": result.get("normalized", {}),
                "patches": result.get("patches", {}),
                "warnings": result.get("warnings", []),
            }
            for model, result in normalized_all.items()
        },
        "truthfinder": {
            "model_trust_rank": [
                {"model": model, "trust": round(float(score), 4)}
                for model, score in rank_models_by_trust(t_score)
            ],
            "object_results": truth_rows,
            "debug_summary": _json_safe(debug_info.get("jsonable", {})),
        },
        "issue_keywords": issue_keywords,
        "article_candidates": article_candidates,
        "issue_article_matrix": issue_article_matrix,
        "evidence_gaps": evidence_gaps,
        "lawyer_brief": lawyer_brief,
        "prompt_template": build_labor_prompt(case_text),
        "disclaimer": DISCLAIMER,
        "_bert_analysis": bert_analysis,
    }


class LaborAPIHandler(BaseHTTPRequestHandler):
    server_version = "LaborLawTruthAPI/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"", "/schema"}:
            self._send_json(200, get_api_schema())
            return
        self._send_json(404, {"ok": False, "error": "Not found", "path": self.path})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/analyze":
            self._send_json(404, {"ok": False, "error": "Not found", "path": self.path})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body) if body.strip() else {}
            if not isinstance(payload, Mapping):
                raise ValueError("JSON body must be an object")
            result = analyze_labor_case(payload)
            self._send_json(200 if result.get("ok") else 400, result)
        except Exception as ex:
            self._send_json(500, {"ok": False, "error": f"{type(ex).__name__}: {ex}"})


def serve(host: str = "127.0.0.1", port: int = 8008) -> None:
    httpd = ThreadingHTTPServer((host, port), LaborAPIHandler)
    try:
        print(f"Labor law analysis API running at http://{host}:{port}")
        print("POST /analyze with JSON body, or GET /schema")
    except (OSError, ValueError):
        pass
    httpd.serve_forever()


def _demo_payload() -> dict[str, Any]:
    return {
        "case_id": "demo_labor_001",
        "case_text": "劳动者2023年3月入职某科技公司，月薪8000元，一直未签书面劳动合同，有打卡和工资流水。2024年4月公司口头辞退，不让继续上班，也没有支付经济补偿。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Labor contract dispute analysis API")
    parser.add_argument("--serve", action="store_true", help="Start stdlib HTTP API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--demo", action="store_true", help="Print a demo API response")
    args = parser.parse_args()

    if args.serve:
        try:
            serve(args.host, args.port)
        except Exception as ex:
            log_path = Path(__file__).resolve().parent.parent / "labor_law_app_server.err.log"
            log_path.write_text(f"{type(ex).__name__}: {ex}\n", encoding="utf-8")
            raise
        return

    payload = _demo_payload()
    result = analyze_labor_case(payload)
    print(json.dumps(result if args.demo else get_api_schema(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
