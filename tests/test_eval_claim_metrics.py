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
