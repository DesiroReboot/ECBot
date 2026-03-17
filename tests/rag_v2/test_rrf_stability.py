from __future__ import annotations

from src.core.search.fusion import ReciprocalRankFusion


def test_rrf_fusion_order_is_stable_across_repeated_runs() -> None:
    fusion = ReciprocalRankFusion(rrf_k=60)

    fts_results = [
        {"file_uuid": "A", "chunk_id": 1, "content": "a", "fts_rank": 1},
        {"file_uuid": "B", "chunk_id": 1, "content": "b", "fts_rank": 2},
    ]
    vec_results = [
        {"file_uuid": "C", "chunk_id": 1, "content": "c", "vec_rank": 1},
        {"file_uuid": "A", "chunk_id": 1, "content": "a", "vec_rank": 3},
    ]

    expected_order = [("A", 1), ("C", 1), ("B", 1)]
    for _ in range(30):
        fused = fusion.fuse(fts_results=fts_results, vec_results=vec_results)
        order = [(row["file_uuid"], row["chunk_id"]) for row in fused]
        assert order == expected_order
