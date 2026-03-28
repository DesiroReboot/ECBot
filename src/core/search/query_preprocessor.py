from __future__ import annotations

import re
from typing import TypedDict


class QueryIntent(TypedDict):
    core_entities: list[str]
    intent_terms: list[str]
    constraint_terms: list[str]
    temporal_terms: list[str]
    need_web_search: bool
    route_mode: str


class QueryPreprocessResult(TypedDict):
    original: str
    normalized: str
    lowered: str
    tokens: list[str]
    theme_hints: list[str]
    query_intent: QueryIntent


class QueryPreprocessor:
    _TEMPORAL_TERMS: tuple[str, ...] = (
        "最近",
        "最新",
        "近期",
        "刚刚",
        "今年",
        "本月",
        "本周",
        "today",
        "latest",
        "recent",
        "this year",
        "this month",
    )
    _INTENT_TERMS: tuple[str, ...] = (
        "打造",
        "做",
        "做出",
        "搭建",
        "优化",
        "提升",
        "增长",
        "推广",
        "选品",
        "上架",
        "爆款",
        "爆品",
        "策略",
        "方案",
        "步骤",
    )
    _CONSTRAINT_TERMS: tuple[str, ...] = (
        "平台政策",
        "政策",
        "合规",
        "监管",
        "规则",
        "法规",
        "风控",
        "限制",
        "关税",
        "物流",
        "清关",
    )
    _THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
        "报关物流": ("报关", "清关", "物流", "头程", "运输", "海外仓", "fba", "发货", "配送"),
        "报价合同": ("报价", "合同", "条款", "付款", "询盘", "pi", "成交"),
        "生产备货": ("生产", "备货", "补货", "库存", "断货", "上架", "新品"),
        "收汇退税": ("收汇", "结汇", "退税", "税务", "财税", "利润", "成本"),
        "客户开发": ("客户", "开发", "广告", "acos", "转化", "评价", "关键词", "标题"),
    }
    _ENTITY_STOPWORDS: set[str] = {
        "如何",
        "怎么",
        "哪些",
        "什么",
        "可以",
        "需要",
        "请问",
        "这个",
        "那个",
        "平台",
        "政策",
        "合规",
        "最新",
        "最近",
        "近期",
        "本月",
        "本周",
        "今年",
        "打造",
        "爆款",
    }
    _TOKEN_BLACKLIST: set[str] = {"在最", "最新", "新的", "在新"}
    _MAX_ENTITY_COUNT = 6

    def process(self, query: str) -> QueryPreprocessResult:
        normalized = re.sub(r"\s+", " ", str(query or "")).strip()
        lowered = normalized.lower()
        query_intent = self._build_query_intent(
            normalized=normalized,
            lowered=lowered,
        )
        tokens = self._build_tokens(lowered, query_intent)
        theme_hints = self._detect_theme_hints(lowered)
        return {
            "original": query,
            "normalized": normalized,
            "lowered": lowered,
            "tokens": tokens,
            "theme_hints": theme_hints,
            "query_intent": query_intent,
        }

    def extract_keywords(self, query: str, top_k: int = 4) -> list[str]:
        processed = self.process(query)
        intent = processed.get("query_intent", {})
        ordered = (
            list(intent.get("core_entities", []))
            + list(intent.get("intent_terms", []))
            + list(intent.get("constraint_terms", []))
            + list(processed["tokens"])
        )
        selected: list[str] = []
        for token in ordered:
            text = str(token).strip()
            if len(text) < 2:
                continue
            if re.fullmatch(r"\d+", text):
                continue
            if text in self._TOKEN_BLACKLIST:
                continue
            if text not in selected:
                selected.append(text)
            if len(selected) >= max(1, int(top_k)):
                break
        return selected

    def extract_progress_keywords(self, query: str, top_k: int = 4) -> list[str]:
        processed = self.process(query)
        intent = processed.get("query_intent", {})
        prioritized = list(intent.get("core_entities", [])) + list(intent.get("intent_terms", []))
        selected: list[str] = []
        for token in prioritized:
            text = str(token).strip()
            if len(text) < 2 or text in self._TOKEN_BLACKLIST:
                continue
            if text not in selected:
                selected.append(text)
            if len(selected) >= max(1, int(top_k)):
                return selected
        return self.extract_keywords(query, top_k=top_k)

    def _build_tokens(self, text: str, query_intent: QueryIntent) -> list[str]:
        tokens: list[str] = []
        tokens.extend(re.findall(r"[a-z0-9_]+", text))
        tokens.extend(query_intent.get("core_entities", []))
        tokens.extend(query_intent.get("intent_terms", []))
        tokens.extend(query_intent.get("constraint_terms", []))
        tokens.extend(query_intent.get("temporal_terms", []))

        cjk_spans = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        for span in cjk_spans:
            if len(span) <= 16:
                tokens.append(span)
            # ngram fallback only for unmatched spans to avoid entity fragmentation as primary terms.
            for ngram_len in (3, 2):
                if len(span) < ngram_len:
                    continue
                for idx in range(0, len(span) - ngram_len + 1):
                    token = span[idx : idx + ngram_len]
                    if token in self._TOKEN_BLACKLIST:
                        continue
                    if token in query_intent.get("core_entities", []):
                        continue
                    tokens.append(token)

        seen: set[str] = set()
        ordered: list[str] = []
        for token in tokens:
            normalized = str(token).strip()
            if not normalized:
                continue
            if len(normalized) < 2:
                continue
            if normalized in self._TOKEN_BLACKLIST:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def _detect_theme_hints(self, lowered: str) -> list[str]:
        scored: list[tuple[int, str]] = []
        for theme, keywords in self._THEME_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in lowered)
            if hits > 0:
                scored.append((hits, theme))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [theme for _, theme in scored[:3]]

    def _build_query_intent(self, *, normalized: str, lowered: str) -> QueryIntent:
        temporal_terms = [term for term in self._TEMPORAL_TERMS if term in lowered]
        constraint_terms = [term for term in self._CONSTRAINT_TERMS if term in lowered]
        intent_terms = [term for term in self._INTENT_TERMS if term in lowered]
        core_entities = self._extract_core_entities(
            normalized=normalized,
            lowered=lowered,
            intent_terms=intent_terms,
            constraint_terms=constraint_terms,
            temporal_terms=temporal_terms,
        )

        need_web_search, route_mode = self._route_hint(
            core_entities=core_entities,
            temporal_terms=temporal_terms,
            constraint_terms=constraint_terms,
        )
        return {
            "core_entities": core_entities,
            "intent_terms": self._dedupe(intent_terms),
            "constraint_terms": self._dedupe(constraint_terms),
            "temporal_terms": self._dedupe(temporal_terms),
            "need_web_search": need_web_search,
            "route_mode": route_mode,
        }

    def _extract_core_entities(
        self,
        *,
        normalized: str,
        lowered: str,
        intent_terms: list[str],
        constraint_terms: list[str],
        temporal_terms: list[str],
    ) -> list[str]:
        candidates: list[str] = []

        explicit_patterns = (
            r"(?:类似|像|如同|比如)\s*([\u4e00-\u9fffA-Za-z0-9_\-\s]{2,40})",
            r"([\u4e00-\u9fffA-Za-z0-9_\-]{2,20})\s*(?:这类|这种|同款)",
        )
        for pattern in explicit_patterns:
            for span in re.findall(pattern, normalized):
                for token in re.split(r"[\s,，、/]+", span):
                    normalized_token = str(token).strip()
                    if self._is_entity_candidate(normalized_token):
                        candidates.append(normalized_token)

        cjk_words = re.findall(r"[\u4e00-\u9fff]{2,10}", normalized)
        blocked = set(intent_terms) | set(constraint_terms) | set(temporal_terms) | self._ENTITY_STOPWORDS
        for token in cjk_words:
            if token in blocked:
                continue
            if any(term in token for term in intent_terms if len(term) >= 2):
                continue
            if any(term in token for term in constraint_terms if len(term) >= 2):
                continue
            if self._is_entity_candidate(token):
                candidates.append(token)

        latin_words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,20}", lowered)
        for token in latin_words:
            normalized_token = token.lower().strip()
            if normalized_token in blocked:
                continue
            candidates.append(normalized_token)
        return self._dedupe(candidates)[: self._MAX_ENTITY_COUNT]

    def _route_hint(
        self,
        *,
        core_entities: list[str],
        temporal_terms: list[str],
        constraint_terms: list[str],
    ) -> tuple[bool, str]:
        has_policy_constraint = any(
            term in {"平台政策", "政策", "合规", "监管", "规则", "法规"} for term in constraint_terms
        )
        has_temporal = bool(temporal_terms)
        has_new_entity = any(len(entity) >= 3 for entity in core_entities)

        if has_temporal and (has_policy_constraint or has_new_entity):
            return True, "web_dominant"
        if has_temporal or has_policy_constraint:
            return True, "hybrid"
        return False, "kb_only"

    def _is_entity_candidate(self, token: str) -> bool:
        text = str(token or "").strip()
        if len(text) < 2:
            return False
        if text in self._ENTITY_STOPWORDS:
            return False
        if re.fullmatch(r"\d+", text):
            return False
        if len(text) == 2 and text in {"平台", "政策", "规则", "步骤"}:
            return False
        return True

    def _dedupe(self, items: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(text)
        return output
