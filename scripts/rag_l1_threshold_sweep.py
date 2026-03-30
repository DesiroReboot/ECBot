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


def _load_queries(dataset_path: Path) -> list[str]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    queries: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and item.strip():
                queries.append(item.strip())
                continue
            if isinstance(item, dict):
                query = str(item.get("query", "") or item.get("question", "")).strip()
                if query:
                    queries.append(query)
    return queries


def _parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for token in str(raw).split(","):
        text = token.strip()
        if not text:
            continue
        try:
            value = max(0.0, min(1.0, float(text)))
        except Exception:
            continue
        values.append(value)
    return sorted(set(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep L1 trigger threshold over a replay dataset.")
    parser.add_argument("--dataset", required=True, help="JSON dataset with query/question fields")
    parser.add_argument("--thresholds", default="0.40,0.50,0.58,0.65,0.72", help="Comma separated thresholds")
    parser.add_argument("--config", default=None, help="Config file path")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(json.dumps({"ok": False, "error": "dataset_not_found"}, ensure_ascii=False, indent=2))
        return 2

    queries = _load_queries(dataset_path)
    thresholds = _parse_thresholds(args.thresholds)
    if not queries or not thresholds:
        print(
            json.dumps(
                {"ok": False, "error": "invalid_dataset_or_thresholds", "query_count": len(queries)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    config = Config(args.config)
    agent = ReActAgent(config)
    orchestrator = agent.search_orchestrator
    if not all(hasattr(orchestrator, name) for name in ("run_l1_partial", "route_by_l1_confidence")):
        print(json.dumps({"ok": False, "error": "orchestrator_lite_methods_unavailable"}, ensure_ascii=False, indent=2))
        return 2

    l1_confidences: list[float] = []
    for query in queries:
        l1_result = orchestrator.run_l1_partial(query)
        l1_confidences.append(float(getattr(l1_result, "confidence", 0.0) or 0.0))

    report: list[dict[str, Any]] = []
    for threshold in thresholds:
        triggers = sum(1 for conf in l1_confidences if conf >= threshold)
        trigger_rate = triggers / max(len(l1_confidences), 1)
        report.append(
            {
                "threshold": threshold,
                "trigger_count": triggers,
                "trigger_rate": round(trigger_rate, 4),
                "template_block_rate": round(1.0 - trigger_rate, 4),
            }
        )

    payload = {
        "ok": True,
        "query_count": len(queries),
        "l1_confidence_avg": round(statistics.fmean(l1_confidences), 6),
        "l1_confidence_p50": round(statistics.median(l1_confidences), 6),
        "report": report,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
