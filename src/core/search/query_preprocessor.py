from __future__ import annotations

import re
from typing import TypedDict


class QueryPreprocessResult(TypedDict):
    original: str
    normalized: str
    lowered: str
    tokens: list[str]
    theme_hints: list[str]


class QueryPreprocessor:
    _THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
        "报关物流": ("报关", "清关", "物流", "头程", "运输", "海外仓", "fba", "发货", "配送"),
        "报价合同": ("报价", "合同", "条款", "付款", "询盘", "pi", "成交"),
        "生产备货": ("生产", "备货", "补货", "库存", "断货", "上架", "新品"),
        "收汇退税": ("收汇", "结汇", "退税", "税务", "财税", "利润", "成本"),
        "客户开发": ("客户", "开发", "广告", "acos", "转化", "评价", "关键词", "标题"),
    }

    def process(self, query: str) -> QueryPreprocessResult:
        normalized = re.sub(r"\s+", " ", query).strip()
        lowered = normalized.lower()
        tokens = self._tokenize(lowered)
        theme_hints = self._detect_theme_hints(lowered)
        return {
            "original": query,
            "normalized": normalized,
            "lowered": lowered,
            "tokens": tokens,
            "theme_hints": theme_hints,
        }

    def extract_keywords(self, query: str, top_k: int = 4) -> list[str]:
        processed = self.process(query)
        ordered = list(processed["tokens"])
        selected: list[str] = []
        for token in ordered:
            text = str(token).strip()
            if len(text) < 2:
                continue
            if re.fullmatch(r"\d+", text):
                continue
            if text not in selected:
                selected.append(text)
            if len(selected) >= max(1, int(top_k)):
                break
        return selected

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        tokens.extend(re.findall(r"[a-z0-9_]+", text))
        cjk_spans = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        for span in cjk_spans:
            tokens.append(span)
            for idx in range(0, len(span) - 1):
                tokens.append(span[idx : idx + 2])
        if not cjk_spans:
            tokens.extend(re.findall(r"[\u4e00-\u9fff]", text))
        seen: set[str] = set()
        ordered: list[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                ordered.append(token)
        return ordered

    def _detect_theme_hints(self, lowered: str) -> list[str]:
        scored: list[tuple[int, str]] = []
        for theme, keywords in self._THEME_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in lowered)
            if hits > 0:
                scored.append((hits, theme))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [theme for _, theme in scored[:3]]
