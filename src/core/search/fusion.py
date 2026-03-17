from __future__ import annotations

from typing import Any


class ReciprocalRankFusion:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(
        self,
        fts_results: list[dict[str, Any]],
        vec_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, int], dict[str, Any]] = {}
        for result in fts_results:
            key = (str(result["file_uuid"]), int(result["chunk_id"]))
            merged.setdefault(key, {}).update(result)
        for result in vec_results:
            key = (str(result["file_uuid"]), int(result["chunk_id"]))
            merged.setdefault(key, {}).update(result)

        fused: list[dict[str, Any]] = []
        for payload in merged.values():
            rrf_score = 0.0
            if payload.get("fts_rank"):
                rrf_score += 1.0 / (self.rrf_k + int(payload["fts_rank"]))
            if payload.get("vec_rank"):
                rrf_score += 1.0 / (self.rrf_k + int(payload["vec_rank"]))
            payload["rrf_score"] = rrf_score
            fused.append(payload)
        fused.sort(key=lambda item: item["rrf_score"], reverse=True)
        return fused
