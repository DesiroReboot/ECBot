from __future__ import annotations

from collections import Counter
import re

from src.RAG.config.kbase_config import KBaseConfig


class Classifier:
    FOREIGN_TRADE_KEYWORDS = {
        "import",
        "export",
        "customs",
        "shipping",
        "incoterms",
        "bill of lading",
        "freight",
        "tariff",
        "clearance",
        "trade",
        "外贸",
        "报关",
        "清关",
        "关税",
    }
    ECOMMERCE_KEYWORDS = {
        "amazon",
        "fba",
        "listing",
        "shopify",
        "dropshipping",
        "ppc",
        "conversion",
        "marketplace",
        "电商",
        "跨境",
        "亚马逊",
        "选品",
    }

    def __init__(self, config: KBaseConfig):
        self.config = config

    def classify(self, content: str) -> tuple[str, float]:
        text = content.lower().strip()
        if not text:
            return "uncategorized", 0.0

        foreign_score = self._score(text, self.FOREIGN_TRADE_KEYWORDS)
        ecommerce_score = self._score(text, self.ECOMMERCE_KEYWORDS)
        total = max(foreign_score + ecommerce_score, 1e-9)

        if foreign_score <= 0 and ecommerce_score <= 0:
            return "uncategorized", 0.0
        if foreign_score >= ecommerce_score:
            return "foreign_trade", round(foreign_score / total, 4)
        return "cross_border_ecommerce", round(ecommerce_score / total, 4)

    def classify_batch(self, contents: list[str]) -> list[tuple[str, float]]:
        return [self.classify(content) for content in contents]

    def extract_keywords(self, content: str, top_n: int = 10) -> list[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]+|[\u4e00-\u9fff]+", content.lower())
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "about",
            "guide",
            "is",
            "are",
            "to",
            "of",
            "in",
        }
        filtered = [token for token in tokens if token not in stop_words and len(token) > 1]
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(max(1, top_n))]

    def _score(self, text: str, keywords: set[str]) -> float:
        score = 0.0
        for keyword in keywords:
            if keyword in text:
                score += 1.0
        return score
