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
    answer_completeness: float = 0.0
    instruction_following_rate: float = 0.0
    actionability_score: float = 0.0
    generation_quality_score: float = 0.0
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
    answer_completeness_avg: float = 0.0
    instruction_following_rate_avg: float = 0.0
    actionability_score_avg: float = 0.0
    generation_quality_score_avg: float = 0.0
    overall_score_v2: float = 0.0


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
                    answer_completeness=base.answer_completeness,
                    instruction_following_rate=base.instruction_following_rate,
                    actionability_score=base.actionability_score,
                    generation_quality_score=base.generation_quality_score,
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

        summary = self._calculate_summary(aggregated_results, repeat_runs=safe_repeat)
        return summary, aggregated_results, traces

    def calculate_answer_completeness(
        self,
        *,
        answer: str,
        must_keyword_coverage: float,
        rubric: dict[str, Any] | None = None,
    ) -> float:
        safe_rubric = rubric or {}
        sections = [str(item).strip() for item in safe_rubric.get("must_have_sections", []) if str(item).strip()]
        if not sections:
            section_coverage = 1.0
        else:
            hits = sum(1 for section in sections if section in answer)
            section_coverage = hits / len(sections)
        score = 0.6 * max(0.0, min(1.0, float(must_keyword_coverage))) + 0.4 * section_coverage
        return round(max(0.0, min(1.0, score)), 4)

    def calculate_instruction_following_rate(
        self,
        *,
        answer: str,
        citations: list[dict[str, Any]],
        rubric: dict[str, Any] | None = None,
        forbidden_claims: list[str] | None = None,
    ) -> float:
        safe_rubric = rubric or {}
        score = 1.0
        if bool(safe_rubric.get("citation_required")) and not citations:
            score -= 0.4
        sections = [str(item).strip() for item in safe_rubric.get("must_have_sections", []) if str(item).strip()]
        if sections:
            hits = sum(1 for section in sections if section in answer)
            section_coverage = hits / len(sections)
            score -= 0.4 * (1.0 - section_coverage)
        for claim in forbidden_claims or []:
            token = str(claim).strip()
            if token and token in answer:
                score -= 0.2
        return round(max(0.0, min(1.0, score)), 4)

    def calculate_actionability_score(self, answer: str) -> float:
        lines = [line.strip() for line in str(answer).splitlines() if line.strip()]
        if not lines:
            return 0.0
        step_hits = len(re.findall(r"(?m)(^\d+\.\s+|步骤|step)", answer.lower()))
        action_tokens = re.findall(r"(先|再|最后|执行|检查|优化|验证|follow|review|test)", answer.lower())
        structure = min(1.0, step_hits / 3.0)
        action_density = min(1.0, len(action_tokens) / 4.0)
        score = 0.55 * structure + 0.45 * action_density
        return round(max(0.0, min(1.0, score)), 4)

    def calculate_generation_quality_score(
        self,
        *,
        answer_completeness: float,
        instruction_following_rate: float,
        actionability_score: float,
    ) -> float:
        score = (
            0.4 * max(0.0, min(1.0, float(answer_completeness)))
            + 0.35 * max(0.0, min(1.0, float(instruction_following_rate)))
            + 0.25 * max(0.0, min(1.0, float(actionability_score)))
        )
        return round(max(0.0, min(1.0, score)), 4)

    def _calculate_summary(self, results: list[EvalResult], *, repeat_runs: int) -> EvalSummary:
        if not results:
            return EvalSummary(total=0, passed=0, pass_rate_at_n_avg=0.0)

        passed_items = sum(1 for row in results if row.pass_count > 0)
        pass_rate_avg = round(mean([row.pass_rate_at_n for row in results]), 4)
        answer_completeness_avg = round(mean([float(row.answer_completeness) for row in results]), 4)
        instruction_following_rate_avg = round(mean([float(row.instruction_following_rate) for row in results]), 4)
        actionability_score_avg = round(mean([float(row.actionability_score) for row in results]), 4)
        generation_quality_score_avg = round(mean([float(row.generation_quality_score) for row in results]), 4)

        repeat_factor = min(1.0, max(1, int(repeat_runs)) / 3.0)
        overall_score = 0.45 * generation_quality_score_avg + 0.45 * pass_rate_avg + 0.1 * repeat_factor
        overall_score_v2 = round(max(0.0, min(1.0, overall_score)), 4)

        return EvalSummary(
            total=len(results),
            passed=passed_items,
            pass_rate_at_n_avg=pass_rate_avg,
            answer_completeness_avg=answer_completeness_avg,
            instruction_following_rate_avg=instruction_following_rate_avg,
            actionability_score_avg=actionability_score_avg,
            generation_quality_score_avg=generation_quality_score_avg,
            overall_score_v2=overall_score_v2,
        )

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
