from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any

from src.core.search.source_utils import canonical_source_id


class ResultGrader:
    _NOISE_MARKERS = (
        "flatedecode",
        "xref",
        "endobj",
        "stream",
        "/filter",
        "/length",
        "obj",
    )

    def grade(
        self,
        *,
        query_tokens: list[str],
        query_theme_hints: list[str] | None = None,
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

        pre_scored: list[dict[str, Any]] = []
        theme_hints = [hint for hint in (query_theme_hints or []) if hint]
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

            readability_score = self._readability_score(content)
            short_penalty = 0.15 if len(content.strip()) < 40 else 0.0
            doc_type = str(item.get("doc_type", "text")).lower()
            pdf_noise_penalty = 0.25 * (1.0 - readability_score) if doc_type == "pdf" else 0.0
            general_noise_penalty = 0.08 * (1.0 - readability_score)
            noise_penalty = short_penalty + pdf_noise_penalty + general_noise_penalty
            source_key = str(item.get("source", ""))
            source_path = str(item.get("source_path", ""))
            source_theme_boost = self._source_theme_boost(
                source=source_key,
                source_path=source_path,
                section_title=str(item.get("section_title", "")),
                doc_type=doc_type,
                query_theme_hints=theme_hints,
            )
            content_hash = str(item.get("content_hash") or hash(content))
            prior_score = content_hash_to_top_score.get(content_hash, 0.0)
            redundancy_penalty = 0.15 if prior_score > 0.75 else 0.0

            candidate_score = (
                0.30 * rrf_norm
                + 0.20 * lexical_norm
                + 0.25 * semantic_norm
                + 0.15 * overlap
                + 0.10 * metadata_boost
                + source_theme_boost
                - redundancy_penalty
                - noise_penalty
            )
            pre_scored.append(
                {
                    **item,
                    "source": source_key,
                    "source_path": source_path,
                    "canonical_source_id": canonical_source_id(source_key, source_path),
                    "_candidate_score": candidate_score,
                    "_readability_score": readability_score,
                    "_noise_penalty": noise_penalty,
                    "_redundancy_penalty": redundancy_penalty,
                    "_overlap": overlap,
                    "_lexical_norm": lexical_norm,
                    "_semantic_norm": semantic_norm,
                    "_rrf_norm": rrf_norm,
                    "_metadata_boost": metadata_boost,
                    "_source_theme_boost": source_theme_boost,
                }
            )
            content_hash_to_top_score[content_hash] = max(prior_score, candidate_score)

        source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in pre_scored:
            source_groups[str(item.get("source", ""))].append(item)

        doc_evidence_mass_by_source: dict[str, float] = {}
        doc_size_prior_by_source: dict[str, float] = {}
        doc_quality_prior_by_source: dict[str, float] = {}
        for source, items in source_groups.items():
            sorted_items = sorted(items, key=lambda row: float(row["_candidate_score"]), reverse=True)
            doc_evidence_mass = sum(
                max(0.0, float(row["_candidate_score"])) for row in sorted_items[:3]
            )
            doc_chunk_count = max(
                int(row.get("doc_chunk_count", 0) or 0) for row in sorted_items
            )
            if doc_chunk_count <= 0:
                doc_chunk_count = len(sorted_items)
            doc_size_prior = math.log1p(max(1, doc_chunk_count))
            doc_quality_prior = sum(
                float(row.get("_readability_score", 0.0)) for row in sorted_items[:5]
            ) / max(min(len(sorted_items), 5), 1)
            doc_evidence_mass_by_source[source] = doc_evidence_mass
            doc_size_prior_by_source[source] = doc_size_prior
            doc_quality_prior_by_source[source] = max(0.05, doc_quality_prior)

        evidence_mass_norm = self._minmax_normalize(doc_evidence_mass_by_source)
        size_quality = {
            source: doc_size_prior_by_source[source] * doc_quality_prior_by_source[source]
            for source in source_groups
        }
        size_quality_norm = self._minmax_normalize(size_quality)

        candidate_results: list[dict[str, Any]] = []
        for item in pre_scored:
            source = str(item.get("source", ""))
            candidate_score = float(item["_candidate_score"])
            final_score = (
                0.65 * candidate_score
                + 0.25 * evidence_mass_norm.get(source, 0.0)
                + 0.10 * size_quality_norm.get(source, 0.0)
            )
            graded = {
                **item,
                "grading": {
                    "rrf_score": round(float(item.get("rrf_score", 0.0)), 6),
                    "lexical_score": round(float(item["_lexical_norm"]), 6),
                    "semantic_score": round(float(item["_semantic_norm"]), 6),
                    "overlap_score": round(float(item["_overlap"]), 6),
                    "metadata_boost": round(float(item["_metadata_boost"]), 6),
                    "source_theme_boost": round(float(item["_source_theme_boost"]), 6),
                    "readability_score": round(float(item["_readability_score"]), 6),
                    "redundancy_penalty": round(float(item["_redundancy_penalty"]), 6),
                    "noise_penalty": round(float(item["_noise_penalty"]), 6),
                    "candidate_score": round(candidate_score, 6),
                    "doc_evidence_mass_norm": round(evidence_mass_norm.get(source, 0.0), 6),
                    "doc_size_quality_norm": round(size_quality_norm.get(source, 0.0), 6),
                    "final_score": round(final_score, 6),
                },
                "score": final_score,
            }
            for key in (
                "_candidate_score",
                "_readability_score",
                "_noise_penalty",
                "_redundancy_penalty",
                "_overlap",
                "_lexical_norm",
                "_semantic_norm",
                "_rrf_norm",
                "_metadata_boost",
                "_source_theme_boost",
            ):
                graded.pop(key, None)
            candidate_results.append(graded)

        candidate_results.sort(key=lambda item: float(item["score"]), reverse=True)
        source_results: list[dict[str, Any]] = []
        for source, items in source_groups.items():
            ranked_items = sorted(
                (row for row in candidate_results if str(row.get("source", "")) == source),
                key=lambda row: float(row["score"]),
                reverse=True,
            )
            if not ranked_items:
                continue
            top_score = float(ranked_items[0]["score"])
            second_score = float(ranked_items[1]["score"]) if len(ranked_items) > 1 else 0.0
            coverage = min(1.0, len(items) / 3.0)
            citation_readiness = 1.0 if source and ranked_items[0].get("source_path") else 0.7
            source_score = (
                0.40 * top_score
                + 0.20 * second_score
                + 0.15 * coverage
                + 0.10 * citation_readiness
                + 0.15 * evidence_mass_norm.get(source, 0.0)
            )
            source_results.append(
                {
                    "source": source,
                    "source_path": ranked_items[0].get("source_path", ""),
                    "canonical_source_id": ranked_items[0].get("canonical_source_id", ""),
                    "score": round(source_score, 6),
                    "coverage": round(coverage, 6),
                    "citation_readiness": round(citation_readiness, 6),
                    "top_chunk_id": ranked_items[0]["chunk_id"],
                    "chunk_count": len(items),
                    "doc_evidence_mass": round(doc_evidence_mass_by_source.get(source, 0.0), 6),
                    "doc_evidence_mass_norm": round(evidence_mass_norm.get(source, 0.0), 6),
                    "doc_size_prior": round(doc_size_prior_by_source.get(source, 0.0), 6),
                    "doc_quality_prior": round(doc_quality_prior_by_source.get(source, 0.0), 6),
                    "doc_size_quality_prior_norm": round(size_quality_norm.get(source, 0.0), 6),
                }
            )

        source_results.sort(key=lambda item: float(item["score"]), reverse=True)
        source_scores = {item["source"]: item["score"] for item in source_results}
        for item in candidate_results:
            item["grading"]["source_score"] = round(source_scores.get(item.get("source", ""), 0.0), 6)
        return candidate_results, source_results

    def _minmax_normalize(self, values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        minimum = min(values.values())
        maximum = max(values.values())
        if maximum - minimum <= 1e-9:
            return {key: 1.0 for key in values}
        return {
            key: (value - minimum) / (maximum - minimum)
            for key, value in values.items()
        }

    def _readability_score(self, content: str) -> float:
        text = str(content or "")
        if not text.strip():
            return 0.0
        lowered = text.lower()
        noise_hits = sum(lowered.count(marker) for marker in self._NOISE_MARKERS)
        readable_chars = re.findall(
            r"[A-Za-z0-9\u4e00-\u9fff，。！？；：、“”‘’（）()、,.!?;:\-_/ ]",
            text,
        )
        readability_ratio = len(readable_chars) / max(len(text), 1)
        noise_ratio = min(1.0, noise_hits / max(len(text) / 80.0, 1.0))
        score = 0.75 * readability_ratio + 0.25 * (1.0 - noise_ratio)
        return max(0.0, min(1.0, score))

    def _source_theme_boost(
        self,
        *,
        source: str,
        source_path: str,
        section_title: str,
        doc_type: str,
        query_theme_hints: list[str],
    ) -> float:
        if not query_theme_hints:
            return 0.0
        source_text = " ".join(
            [
                str(source).lower(),
                str(source_path).lower(),
                str(section_title).lower(),
                canonical_source_id(source, source_path),
            ]
        )
        if not source_text.strip():
            return 0.0

        theme_aliases: dict[str, tuple[str, ...]] = {
            "报关物流": ("报关", "清关", "物流", "运输", "fba", "海外仓"),
            "报价合同": ("报价", "合同", "条款", "询盘", "成交"),
            "生产备货": ("生产", "备货", "补货", "库存", "断货"),
            "收汇退税": ("收汇", "结汇", "退税", "税"),
            "客户开发": ("客户", "开发", "广告", "acos", "转化", "关键词", "标题"),
        }
        boost = 0.0
        for hint in query_theme_hints:
            aliases = theme_aliases.get(hint, (hint.lower(),))
            if any(alias and alias in source_text for alias in aliases):
                boost += 0.08
        if boost > 0 and doc_type == "pdf":
            boost += 0.04
        return min(0.28, boost)
