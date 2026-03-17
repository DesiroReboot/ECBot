from __future__ import annotations

from typing import Any


class ContextSelector:
    def select(
        self,
        *,
        candidates: list[dict[str, Any]],
        source_scores: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        allowed_sources = [item["source"] for item in source_scores]
        per_source_count: dict[str, int] = {}
        seen_content: set[str] = set()
        selected: list[dict[str, Any]] = []

        for source in allowed_sources:
            for candidate in candidates:
                if candidate.get("source") != source:
                    continue
                if per_source_count.get(source, 0) >= 2:
                    continue
                content_key = str(candidate.get("content", "")).strip()
                if content_key in seen_content:
                    continue
                selected.append(candidate)
                per_source_count[source] = per_source_count.get(source, 0) + 1
                seen_content.add(content_key)
                if len(selected) >= top_k:
                    break
            if len(selected) >= top_k:
                break

        citations: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for item in selected:
            source = str(item.get("source", ""))
            if source in seen_sources:
                continue
            seen_sources.add(source)
            citations.append(
                {
                    "source": source,
                    "title": source,
                    "path": item.get("source_path", ""),
                }
            )
        return selected, citations
