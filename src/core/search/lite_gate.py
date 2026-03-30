from __future__ import annotations

from typing import Any


LOW_RELEVANCE_REASON_CODE = "LOW_RELEVANCE_GATE"


def compute_l1_confidence(l1_hits: list[Any], l1_trace: dict[str, Any] | None = None) -> float:
    if not l1_hits:
        return 0.0

    ordered_scores = sorted((max(0.0, _safe_score(hit)) for hit in l1_hits), reverse=True)
    top1 = ordered_scores[0] if ordered_scores else 0.0
    top3 = ordered_scores[:3]
    top3_mean = sum(top3) / max(len(top3), 1)

    metrics = _trace_metrics(l1_trace or {})
    evidence_count = int(metrics.get("evidence_count", len(l1_hits)) or 0)
    evidence_score = min(1.0, evidence_count / 3.0)
    coverage_score = _safe_float(metrics.get("coverage_score", 0.0))

    confidence = 0.5 * top1 + 0.25 * top3_mean + 0.15 * evidence_score + 0.1 * coverage_score
    return max(0.0, min(1.0, float(confidence)))


def should_trigger_full_rag(l1_confidence: float, threshold: float) -> bool:
    return float(l1_confidence) >= float(threshold)


def build_template_response(query: str, reason_code: str) -> str:
    normalized_query = str(query or "").strip()
    normalized_reason = str(reason_code or LOW_RELEVANCE_REASON_CODE).strip() or LOW_RELEVANCE_REASON_CODE
    return (
        "当前问题与知识库内容相关性不足，已触发轻量门控并返回模板答复。\n"
        f"问题：{normalized_query or '（空）'}\n"
        f"原因：{normalized_reason}\n"
        "建议：请补充更具体的业务背景、平台名称、目标市场或关键约束后再试。"
    )


def _safe_score(hit: Any) -> float:
    if isinstance(hit, dict):
        return _safe_float(hit.get("score", 0.0))
    return _safe_float(getattr(hit, "score", 0.0))


def _trace_metrics(l1_trace: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(l1_trace, dict):
        return {}
    metrics = l1_trace.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
