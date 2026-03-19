from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.RAG.config.kbase_config import KBaseConfig
from src.RAG.preprocessing.parser import DocumentParser
from src.RAG.reader.chunker import Chunker
from src.core.search.context_selector import ContextSelector
from src.core.search.grader import ResultGrader
from src.core.search.query_preprocessor import QueryPreprocessor
from src.core.search.source_utils import build_grouped_citations


def test_grader_adds_doc_priors_and_pdf_quality_signal() -> None:
    grader = ResultGrader()
    fused_results = [
        {
            "file_uuid": "a",
            "chunk_id": 0,
            "source": "alpha.pdf",
            "source_path": "/kb/alpha.pdf",
            "doc_type": "pdf",
            "doc_chunk_count": 12,
            "content": "跨境电商选品需要先看市场需求，再分析竞争和利润空间。",
            "rrf_score": 0.08,
            "fts_rank": 1,
            "vec_similarity": 0.88,
        },
        {
            "file_uuid": "b",
            "chunk_id": 0,
            "source": "beta.txt",
            "source_path": "/kb/beta.txt",
            "doc_type": "text",
            "doc_chunk_count": 3,
            "content": "stream xref endobj /Filter /Length",
            "rrf_score": 0.07,
            "fts_rank": 2,
            "vec_similarity": 0.78,
        },
    ]
    candidates, source_scores = grader.grade(
        query_tokens=["跨境", "选品", "利润"],
        fused_results=fused_results,
    )

    assert len(candidates) == 2
    assert len(source_scores) == 2
    for row in candidates:
        grading = row["grading"]
        assert "doc_evidence_mass_norm" in grading
        assert "doc_size_quality_norm" in grading
        assert "final_score" in grading
        assert "readability_score" in grading
        assert abs(row["score"] - grading["final_score"]) < 1e-6

    pdf_row = next(row for row in candidates if row["source"] == "alpha.pdf")
    txt_row = next(row for row in candidates if row["source"] == "beta.txt")
    assert pdf_row["grading"]["readability_score"] > txt_row["grading"]["readability_score"]
    assert txt_row["grading"]["noise_penalty"] > pdf_row["grading"]["noise_penalty"]


def test_context_selector_soft_quota_prefers_high_value_source() -> None:
    selector = ContextSelector()
    candidates = [
        {
            "file_uuid": "a",
            "chunk_id": idx,
            "source": "high.pdf",
            "source_path": "/kb/high.pdf",
            "content": f"high-{idx}",
            "score": 1.0 - idx * 0.05,
        }
        for idx in range(5)
    ] + [
        {
            "file_uuid": "b",
            "chunk_id": idx,
            "source": "low.txt",
            "source_path": "/kb/low.txt",
            "content": f"low-{idx}",
            "score": 0.4 - idx * 0.05,
        }
        for idx in range(2)
    ]
    source_scores = [
        {
            "source": "high.pdf",
            "score": 0.9,
            "doc_evidence_mass_norm": 1.0,
        },
        {
            "source": "low.txt",
            "score": 0.3,
            "doc_evidence_mass_norm": 0.1,
        },
    ]
    selected, citations = selector.select(
        candidates=candidates,
        source_scores=source_scores,
        top_k=4,
    )

    assert len(selected) == 4
    high_count = sum(1 for row in selected if row["source"] == "high.pdf")
    assert high_count >= 3
    assert selector.last_source_quotas["high.pdf"] >= 3
    assert citations


def test_canonical_source_groups_pdf_and_txt_versions() -> None:
    citations = build_grouped_citations(
        [
            {
                "source": "01-市场调研-f3211fd0f5.pdf",
                "source_path": "/kb/01-市场调研-f3211fd0f5.pdf",
                "score": 0.8,
            },
            {
                "source": "01-市场调研与选品.txt",
                "source_path": "/kb/01-市场调研与选品.txt",
                "score": 0.7,
            },
        ]
    )
    assert len(citations) == 1
    assert len(citations[0]["versions"]) == 2
    assert citations[0]["aliases"]


def test_pdf_parser_sanitizes_noise_tokens() -> None:
    parser = DocumentParser(KBaseConfig())
    temp_dir = Path("tmp_test_pdf_parser") / uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_text(
        "stream\nxref\n正常的中文句子用于选品分析。\nendobj\n",
        encoding="utf-8",
    )
    try:
        content, metadata = parser.parse(pdf_path)
        assert metadata["type"] == "pdf"
        assert "parse_method" in metadata
        assert metadata["readability_score"] >= 0.0
        assert "正常的中文句子" in content
    finally:
        if pdf_path.exists():
            pdf_path.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()


def test_query_preprocessor_extracts_theme_hints() -> None:
    preprocessor = QueryPreprocessor()
    processed = preprocessor.process("报关流程和物流时效怎么优化？")
    assert "报关物流" in processed["theme_hints"]
    assert processed["tokens"]


def test_chunker_uses_semantic_units_before_windowing() -> None:
    chunker = Chunker(chunk_size=36, chunk_overlap=8)
    chunks = chunker.split(
        "第一句介绍选品原则。第二句介绍利润核算。第三句介绍供应链稳定性。\n\n"
        "第四句讲报关物流准备。第五句讲合同条款核对。"
    )
    assert len(chunks) >= 2
    assert any("报关物流" in chunk for chunk in chunks)


def test_grader_source_theme_boost_prefers_matching_source() -> None:
    grader = ResultGrader()
    fused_results = [
        {
            "file_uuid": "logistics",
            "chunk_id": 1,
            "source": "04-报关物流-fbcc42e5c3.pdf",
            "source_path": "/kb/04-报关物流-fbcc42e5c3.pdf",
            "doc_type": "pdf",
            "doc_chunk_count": 10,
            "content": "报关需要核对HS编码、申报要素与清关资料。",
            "rrf_score": 0.07,
            "fts_rank": 2,
            "vec_similarity": 0.8,
        },
        {
            "file_uuid": "other",
            "chunk_id": 1,
            "source": "01-客户开发-051dbc4035.pdf",
            "source_path": "/kb/01-客户开发-051dbc4035.pdf",
            "doc_type": "pdf",
            "doc_chunk_count": 10,
            "content": "开发客户需要构建邮件触达节奏和跟进SOP。",
            "rrf_score": 0.07,
            "fts_rank": 2,
            "vec_similarity": 0.8,
        },
    ]
    candidates, _ = grader.grade(
        query_tokens=["报关", "物流", "清关"],
        query_theme_hints=["报关物流"],
        fused_results=fused_results,
    )
    assert candidates[0]["source"] == "04-报关物流-fbcc42e5c3.pdf"
    assert candidates[0]["grading"]["source_theme_boost"] > 0
