from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config  # noqa: E402
from src.core.bot_agent import ReActAgent  # noqa: E402


def _load_items(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "") or item.get("question", "")).strip()
        if not query:
            continue
        rows.append(item)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay dataset over RAG-lite gate and report routing metrics.")
    parser.add_argument("--dataset", required=True, help="JSON list with query/question; optional relevant bool")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--threshold", default="", help="Optional L1 trigger threshold override")
    parser.add_argument("--verbose", action="store_true", help="Include per-item routing records")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(json.dumps({"ok": False, "error": "dataset_not_found"}, ensure_ascii=False, indent=2))
        return 2

    items = _load_items(dataset_path)
    if not items:
        print(json.dumps({"ok": False, "error": "dataset_empty_or_invalid"}, ensure_ascii=False, indent=2))
        return 2

    config = Config(args.config)
    if str(args.threshold).strip():
        try:
            config.search.l1_trigger_threshold = max(0.0, min(1.0, float(args.threshold)))
        except Exception:
            pass

    agent = ReActAgent(config)
    orchestrator = agent.search_orchestrator
    if not all(hasattr(orchestrator, name) for name in ("run_l1_partial", "route_by_l1_confidence", "run_l2_full")):
        print(json.dumps({"ok": False, "error": "orchestrator_lite_methods_unavailable"}, ensure_ascii=False, indent=2))
        return 2

    template_count = 0
    full_rag_count = 0
    false_reject_count = 0
    l1_confidences: list[float] = []
    l2_confidences: list[float] = []
    per_item: list[dict[str, Any]] = []

    for item in items:
        query = str(item.get("query", "") or item.get("question", "")).strip()
        l1_result = orchestrator.run_l1_partial(query)
        decision = orchestrator.route_by_l1_confidence(l1_result)
        l1_conf = float(getattr(l1_result, "confidence", 0.0) or 0.0)
        l1_confidences.append(l1_conf)

        triggered = bool(getattr(decision, "trigger_full_rag", False))
        route = "full_rag" if triggered else "template"
        l2_conf = 0.0
        if triggered:
            full_rag_count += 1
            l2_result = orchestrator.run_l2_full(query, l1_result)
            l2_conf = float(getattr(l2_result, "retrieval_confidence", 0.0) or 0.0)
            l2_confidences.append(l2_conf)
        else:
            template_count += 1
            if bool(item.get("relevant", False)):
                false_reject_count += 1

        if args.verbose:
            per_item.append(
                {
                    "query": query,
                    "route": route,
                    "l1_confidence": round(l1_conf, 6),
                    "threshold": float(getattr(decision, "threshold", config.search.l1_trigger_threshold) or 0.0),
                    "reason_code": str(getattr(decision, "reason_code", "")),
                    "l2_retrieval_confidence": round(l2_conf, 6),
                    "relevant": bool(item.get("relevant", False)),
                }
            )

    total = len(items)
    payload: dict[str, Any] = {
        "ok": True,
        "total": total,
        "template_count": template_count,
        "full_rag_count": full_rag_count,
        "template_rate": round(template_count / max(total, 1), 4),
        "full_rag_rate": round(full_rag_count / max(total, 1), 4),
        "false_reject_count": false_reject_count,
        "false_reject_rate": round(false_reject_count / max(template_count, 1), 4) if template_count else 0.0,
        "l1_confidence_avg": round(statistics.fmean(l1_confidences), 6),
        "l2_confidence_avg": round(statistics.fmean(l2_confidences), 6) if l2_confidences else 0.0,
    }
    if args.verbose:
        payload["items"] = per_item
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
