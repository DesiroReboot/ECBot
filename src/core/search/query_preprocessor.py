from __future__ import annotations

import re


class QueryPreprocessor:
    def process(self, query: str) -> dict[str, object]:
        normalized = re.sub(r"\s+", " ", query).strip()
        lowered = normalized.lower()
        tokens = self._tokenize(lowered)
        return {
            "original": query,
            "normalized": normalized,
            "lowered": lowered,
            "tokens": tokens,
        }

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
        seen: set[str] = set()
        ordered: list[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                ordered.append(token)
        return ordered
