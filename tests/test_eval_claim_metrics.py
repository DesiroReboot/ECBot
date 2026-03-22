from __future__ import annotations

from scripts.eval_check import EvalChecker


def test_claim_metrics_supported_and_cited() -> None:
    checker = EvalChecker.__new__(EvalChecker)
    answer = "先分析市场需求，再评估利润空间。最后小批量测试。"
    citations = [{"source": "market.pdf", "title": "market.pdf"}]
    rag_trace = {
        "search": {
            "final_results": [
                {
                    "source": "market.pdf",
                    "content": "市场需求需要先调研，评估利润空间并进行小批量测试。",
                }
            ]
        }
    }

    supported, precision = checker.calculate_claim_metrics(
        answer=answer,
        citations=citations,
        rag_trace=rag_trace,
    )

    assert supported > 0.0
    assert precision > 0.0


def test_citation_validity_accepts_alias_versions() -> None:
    checker = EvalChecker.__new__(EvalChecker)
    citations = [
        {
            "source": "01-市场调研-f3211fd0f5.pdf",
            "title": "01-市场调研-f3211fd0f5.pdf",
            "aliases": ["01-市场调研与选品.txt"],
            "versions": [
                {"source": "01-市场调研-f3211fd0f5.pdf", "path": "/kb/a.pdf"},
                {"source": "01-市场调研与选品.txt", "path": "/kb/b.txt"},
            ],
        }
    ]
    expected_sources = {
        "must": ["01-市场调研与选品.txt"],
        "should": [],
        "equivalent_sources": [],
        "source_aliases": {},
    }

    strict_hit, relaxed_hit = checker.check_citation_validity(citations, expected_sources)

    assert strict_hit is True
    assert relaxed_hit is True


def test_generation_quality_metrics_with_constraints() -> None:
    checker = EvalChecker.__new__(EvalChecker)
    answer = "步骤一：先做市场调研。步骤二：再做小批量测试。注意风险与成本。"
    citations = [{"source": "market.pdf", "title": "market.pdf"}]
    rubric = {"must_have_sections": ["步骤", "风险"], "citation_required": True}

    answer_completeness = checker.calculate_answer_completeness(
        answer=answer,
        must_keyword_coverage=0.5,
        rubric=rubric,
    )
    instruction_following_rate = checker.calculate_instruction_following_rate(
        answer=answer,
        citations=citations,
        rubric=rubric,
        forbidden_claims=["保证盈利"],
    )
    actionability_score = checker.calculate_actionability_score(answer)
    generation_quality_score = checker.calculate_generation_quality_score(
        answer_completeness=answer_completeness,
        instruction_following_rate=instruction_following_rate,
        actionability_score=actionability_score,
    )

    assert answer_completeness > 0.5
    assert instruction_following_rate == 1.0
    assert actionability_score > 0.0
    assert generation_quality_score > 0.0
