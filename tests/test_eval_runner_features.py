from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.eval_check import EvalChecker, EvalResult, GoldenItem


def _build_result(*, passed: bool) -> EvalResult:
    if passed:
        return EvalResult(
            id="001",
            question="q",
            scenario="s",
            difficulty="medium",
            source_of_question="real",
            answer="ok",
            must_source_recall=1.0,
            source_precision=1.0,
            must_keyword_coverage=1.0,
            should_keyword_coverage=1.0,
            keyword_score=1.0,
            strict_citation_hit=True,
            relaxed_citation_hit=True,
            claim_supported_rate=0.9,
            claim_citation_precision=1.0,
            hallucination_rate=0.1,
            failure_path={"status": "passed", "quality_gate_fail_reasons": []},
        )
    return EvalResult(
        id="001",
        question="q",
        scenario="s",
        difficulty="medium",
        source_of_question="real",
        answer="bad",
        must_source_recall=0.0,
        source_precision=0.0,
        must_keyword_coverage=0.0,
        should_keyword_coverage=0.0,
        keyword_score=0.0,
        strict_citation_hit=False,
        relaxed_citation_hit=False,
        claim_supported_rate=0.1,
        claim_citation_precision=0.0,
        hallucination_rate=0.9,
        failure_path={"status": "degraded", "quality_gate_fail_reasons": ["strict_citation_not_hit"]},
    )


def test_evaluate_all_computes_pass_rate_at_n_and_multi_run_trace() -> None:
    checker = EvalChecker.__new__(EvalChecker)
    runs = [_build_result(passed=True), _build_result(passed=False), _build_result(passed=True)]
    idx = {"value": 0}

    def fake_evaluate_single(_: GoldenItem):
        result = runs[idx["value"]]
        idx["value"] += 1
        return result, {"id": "001", "rag_trace": {}, "failure_path": result.failure_path, "metrics": {}}

    checker.evaluate_single = fake_evaluate_single  # type: ignore[method-assign]

    summary, results, traces = checker.evaluate_all(
        [GoldenItem(id="001", question="q")],
        repeat=3,
    )

    assert len(results) == 1
    assert results[0].repeat_runs == 3
    assert results[0].pass_count == 2
    assert abs(results[0].pass_rate_at_n - 0.6667) < 1e-6
    assert abs(summary.pass_rate_at_n_avg - 0.6667) < 1e-6
    assert traces["001"]["multi_run"]["repeat"] == 3
    assert traces["001"]["multi_run"]["pass_count"] == 2
    assert len(traces["001"]["multi_run"]["runs"]) == 3


def test_write_report_artifacts_writes_versioned_manifest() -> None:
    checker = EvalChecker.__new__(EvalChecker)
    test_root = Path("tmp_test_eval_report_manifest")
    report_root = test_root / "report"
    output_file = report_root / "golden_set" / "20260319-101010" / "eval_report.json"

    if test_root.exists():
        shutil.rmtree(test_root, ignore_errors=True)

    manifest = checker.write_report_artifacts(
        dataset_name="golden_set",
        run_tag="20260319-101010",
        output_path=output_file,
        report_root=report_root,
    )

    try:
        assert manifest["run_tag"] == "20260319-101010"
        assert manifest["report_file"] == str(output_file.resolve())

        index_manifest = json.loads((report_root / "index.json").read_text(encoding="utf-8"))
        assert index_manifest["golden_set"]["latest_run"] == "20260319-101010"
        assert index_manifest["golden_set"]["runs"]["20260319-101010"]["report_file"] == str(output_file.resolve())
        assert (report_root / "golden_set" / "20260319-101010" / "index.json").exists()
    finally:
        if test_root.exists():
            shutil.rmtree(test_root, ignore_errors=True)
