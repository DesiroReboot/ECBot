from __future__ import annotations

from collections import defaultdict
from typing import Any


class ResultGrader:
    def grade(
        self,
        *,
        query_tokens: list[str],
        fused_results: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not fused_results:
            return [], []

        max_rrf = max((float(item.get("rrf_score", 0.0)) for item in fused_results), default=1.0)
        max_similarity = max(
            (max(0.0, float(item.get("vec_similarity", 0.0))) for item in fused_results),
            default=1.0,
        )
        max_lexical_rank = max((int(item.get("fts_rank", 0)) for item in fused_results), default=1)
        content_hash_to_top_score: dict[str, float] = {}

        candidate_results: list[dict[str, Any]] = []
        for item in fused_results:
            content = str(item.get("content", ""))
            content_lower = content.lower()
            overlap = 0.0
            if query_tokens:
                overlap = sum(1 for token in query_tokens if token in content_lower) / len(query_tokens)

            lexical_norm = 0.0
            if item.get("fts_rank"):
                lexical_norm = 1.0 - ((int(item["fts_rank"]) - 1) / max(max_lexical_rank, 1))
            semantic_norm = max(0.0, float(item.get("vec_similarity", 0.0))) / max(
                max_similarity,
                1e-9,
            )
            rrf_norm = float(item.get("rrf_score", 0.0)) / max(max_rrf, 1e-9)

            source = str(item.get("source", "")).lower()
            section_title = str(item.get("section_title", "")).lower()
            metadata_boost = min(
                1.0,
                sum(0.5 for token in query_tokens if token in source or token in section_title),
            )

            noise_penalty = 0.15 if len(content.strip()) < 40 else 0.0
            content_hash = str(item.get("content_hash") or hash(content))
            prior_score = content_hash_to_top_score.get(content_hash, 0.0)
            redundancy_penalty = 0.15 if prior_score > 0.75 else 0.0

            candidate_score = (
                0.30 * rrf_norm
                + 0.20 * lexical_norm
                + 0.25 * semantic_norm
                + 0.15 * overlap
                + 0.10 * metadata_boost
                - redundancy_penalty
                - noise_penalty
            )

            graded = {
                **item,
                "grading": {
                    "rrf_score": round(float(item.get("rrf_score", 0.0)), 6),
                    "lexical_score": round(lexical_norm, 6),
                    "semantic_score": round(semantic_norm, 6),
                    "overlap_score": round(overlap, 6),
                    "metadata_boost": round(metadata_boost, 6),
                    "redundancy_penalty": round(redundancy_penalty, 6),
                    "noise_penalty": round(noise_penalty, 6),
                    "candidate_score": round(candidate_score, 6),
                },
                "score": candidate_score,
            }
            content_hash_to_top_score[content_hash] = max(prior_score, candidate_score)
            candidate_results.append(graded)

        candidate_results.sort(key=lambda item: item["score"], reverse=True)
        source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in candidate_results:
            source_groups[str(item.get("source", ""))].append(item)

        source_results: list[dict[str, Any]] = []
        for source, items in source_groups.items():
            top_score = items[0]["score"]
            second_score = items[1]["score"] if len(items) > 1 else 0.0
            coverage = min(1.0, len(items) / 3.0)
            citation_readiness = 1.0 if source and items[0].get("source_path") else 0.7
            source_score = (
                0.50 * top_score
                + 0.20 * second_score
                + 0.15 * coverage
                + 0.15 * citation_readiness
            )
            source_results.append(
                {
                    "source": source,
                    "source_path": items[0].get("source_path", ""),
                    "score": round(source_score, 6),
                    "coverage": round(coverage, 6),
                    "citation_readiness": round(citation_readiness, 6),
                    "top_chunk_id": items[0]["chunk_id"],
                    "chunk_count": len(items),
                }
            )

        source_results.sort(key=lambda item: item["score"], reverse=True)
        source_scores = {item["source"]: item["score"] for item in source_results}
        for item in candidate_results:
            item["grading"]["source_score"] = round(source_scores.get(item.get("source", ""), 0.0), 6)
        return candidate_results, source_results
