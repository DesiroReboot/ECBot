"""Evaluation helpers for golden-set checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any


@dataclass
class GoldenItem:
    id: str
    question: str
    scenario: str = ""
    difficulty: str = ""
    source_of_question: str = ""


@dataclass
class EvalResult:
    id: str
    question: str
    scenario: str
    difficulty: str
    source_of_question: str
    answer: str
    must_source_recall: float
    source_precision: float
    must_keyword_coverage: float
    should_keyword_coverage: float
    keyword_score: float
    strict_citation_hit: bool
    relaxed_citation_hit: bool
    claim_supported_rate: float
    claim_citation_precision: float
    hallucination_rate: float
    failure_path: dict[str, Any]
    repeat_runs: int = 1
    pass_count: int = 0
    pass_rate_at_n: float = 0.0

    def __post_init__(self) -> None:
        if self.pass_count <= 0:
            self.pass_count = 1 if self._is_passed() else 0
        if self.repeat_runs <= 0:
            self.repeat_runs = 1
        if self.pass_rate_at_n <= 0.0 and self.repeat_runs > 0:
            self.pass_rate_at_n = round(self.pass_count / self.repeat_runs, 4)

    def _is_passed(self) -> bool:
        status = str(self.failure_path.get("status", "")).strip().lower()
        return status in {"passed", "pass", "ok", "success"}


@dataclass
class EvalSummary:
    total: int
    passed: int
    pass_rate_at_n_avg: float


class EvalChecker:
    def evaluate_single(self, item: GoldenItem) -> tuple[EvalResult, dict[str, Any]]:
        raise NotImplementedError("evaluate_single should be provided by runtime pipeline")

    def evaluate_all(
        self,
        items: list[GoldenItem],
        *,
        repeat: int = 1,
    ) -> tuple[EvalSummary, list[EvalResult], dict[str, Any]]:
        safe_repeat = max(1, int(repeat))
        aggregated_results: list[EvalResult] = []
        traces: dict[str, Any] = {}

        for item in items:
            run_results: list[EvalResult] = []
            run_traces: list[dict[str, Any]] = []
            for _ in range(safe_repeat):
                result, trace = self.evaluate_single(item)
                run_results.append(result)
                run_traces.append(trace)

            pass_count = sum(1 for row in run_results if self._is_passed(row))
            pass_rate = round(pass_count / safe_repeat, 4)
            base = run_results[-1]
            aggregated_results.append(
                EvalResult(
                    id=base.id,
                    question=base.question,
                    scenario=base.scenario,
                    difficulty=base.difficulty,
                    source_of_question=base.source_of_question,
                    answer=base.answer,
                    must_source_recall=base.must_source_recall,
                    source_precision=base.source_precision,
                    must_keyword_coverage=base.must_keyword_coverage,
                    should_keyword_coverage=base.should_keyword_coverage,
                    keyword_score=base.keyword_score,
                    strict_citation_hit=base.strict_citation_hit,
                    relaxed_citation_hit=base.relaxed_citation_hit,
                    claim_supported_rate=base.claim_supported_rate,
                    claim_citation_precision=base.claim_citation_precision,
                    hallucination_rate=base.hallucination_rate,
                    failure_path=base.failure_path,
                    repeat_runs=safe_repeat,
                    pass_count=pass_count,
                    pass_rate_at_n=pass_rate,
                )
            )
            traces[item.id] = {
                "multi_run": {
                    "repeat": safe_repeat,
                    "pass_count": pass_count,
                    "runs": run_traces,
                }
            }

        passed_items = sum(1 for row in aggregated_results if row.pass_count > 0)
        pass_rate_avg = (
            round(mean([row.pass_rate_at_n for row in aggregated_results]), 4)
            if aggregated_results
            else 0.0
        )
        summary = EvalSummary(
            total=len(aggregated_results),
            passed=passed_items,
            pass_rate_at_n_avg=pass_rate_avg,
        )
        return summary, aggregated_results, traces

    def calculate_claim_metrics(
        self,
        *,
        answer: str,
        citations: list[dict[str, Any]],
        rag_trace: dict[str, Any],
    ) -> tuple[float, float]:
        claims = [seg.strip() for seg in re.split(r"[。！？!?\n]+", str(answer)) if seg.strip()]
        if not claims:
            return 0.0, 0.0

        cited_sources = {
            str(item.get("source") or item.get("title") or "").strip().lower()
            for item in citations
            if isinstance(item, dict)
        }
        docs = rag_trace.get("search", {}).get("final_results", []) if isinstance(rag_trace, dict) else []
        corpus = "\n".join(str(doc.get("content", "")) for doc in docs if isinstance(doc, dict)).lower()

        supported_count = 0
        for claim in claims:
            tokens = self._claim_tokens(claim.lower())
            if not tokens:
                continue
            overlap = sum(1 for tok in tokens if tok in corpus)
            if overlap / max(len(tokens), 1) >= 0.3:
                supported_count += 1

        source_hits = 0
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            source = str(doc.get("source", "")).strip().lower()
            if source and source in cited_sources:
                source_hits += 1

        supported_rate = supported_count / max(len(claims), 1)
        citation_precision = source_hits / max(len(cited_sources), 1)
        return round(supported_rate, 4), round(citation_precision, 4)

    def _claim_tokens(self, text: str) -> list[str]:
        ascii_tokens = [tok for tok in re.findall(r"[a-z0-9_]{2,}", text)]
        cjk_spans = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        cjk_tokens: list[str] = []
        for span in cjk_spans:
            if len(span) <= 4:
                cjk_tokens.append(span)
            for idx in range(0, len(span) - 1):
                cjk_tokens.append(span[idx : idx + 2])
        deduped: list[str] = []
        seen: set[str] = set()
        for token in ascii_tokens + cjk_tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped

    def check_citation_validity(
        self,
        citations: list[dict[str, Any]],
        expected_sources: dict[str, Any],
    ) -> tuple[bool, bool]:
        actual: set[str] = set()
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            for key in ("source", "title"):
                value = str(citation.get(key, "")).strip()
                if value:
                    actual.add(value)
            for alias in citation.get("aliases", []) if isinstance(citation.get("aliases"), list) else []:
                alias_value = str(alias).strip()
                if alias_value:
                    actual.add(alias_value)
            versions = citation.get("versions", [])
            if isinstance(versions, list):
                for version in versions:
                    if isinstance(version, dict):
                        value = str(version.get("source", "")).strip()
                        if value:
                            actual.add(value)

        must = {str(x).strip() for x in expected_sources.get("must", [])}
        should = {str(x).strip() for x in expected_sources.get("should", [])}

        strict_hit = must.issubset(actual)
        relaxed_hit = strict_hit
        if not relaxed_hit and should:
            relaxed_hit = any(src in actual for src in must | should)
        return strict_hit, relaxed_hit

    def write_report_artifacts(
        self,
        *,
        dataset_name: str,
        run_tag: str,
        output_path: Path,
        report_root: Path,
    ) -> dict[str, Any]:
        report_root.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_file = str(output_path.resolve())
        run_manifest = {
            "dataset": dataset_name,
            "run_tag": run_tag,
            "report_file": report_file,
        }

        run_index_path = output_path.parent / "index.json"
        run_index_path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        global_index_path = report_root / "index.json"
        if global_index_path.exists():
            global_index = json.loads(global_index_path.read_text(encoding="utf-8"))
        else:
            global_index = {}

        dataset_index = global_index.get(dataset_name, {"latest_run": "", "runs": {}})
        dataset_index["latest_run"] = run_tag
        dataset_index.setdefault("runs", {})[run_tag] = {
            "report_file": report_file,
            "index_file": str(run_index_path.resolve()),
        }
        global_index[dataset_name] = dataset_index

        global_index_path.write_text(
            json.dumps(global_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return run_manifest

    @staticmethod
    def _is_passed(result: EvalResult) -> bool:
        status = str(result.failure_path.get("status", "")).strip().lower()
        return status in {"passed", "pass", "ok", "success"}
