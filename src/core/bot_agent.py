from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Config
from src.core.search.rag_search import RAGSearcher
from src.RAG.config.kbase_config import KBaseConfig
from src.RAG.storage.manifest_store import ManifestStore


@dataclass
class AgentResponse:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieval_confidence: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)


class ReActAgent:
    def __init__(self, config: Config):
        self.config = config
        self.kbase_config = KBaseConfig(
            db_path=config.database.db_path,
            source_dir=config.knowledge_base.source_dir,
            supported_extensions=config.knowledge_base.supported_extensions,
            auto_sync_on_startup=config.knowledge_base.auto_sync_on_startup,
            vector_dimension=config.embedding.dimension,
            rag_top_k=config.search.rag_top_k,
            fts_top_k=config.search.fts_top_k,
            vec_top_k=config.search.vec_top_k,
            fusion_rrf_k=config.search.fusion_rrf_k,
            context_top_k=config.search.context_top_k,
            chunk_size=config.knowledge_base.chunk_size,
            chunk_overlap=config.knowledge_base.chunk_overlap,
            embedding_provider=config.embedding.provider,
            embedding_base_url=config.embedding.base_url,
            embedding_api_key=config.embedding.api_key,
            embedding_model=config.embedding.model,
            embedding_batch_size=config.embedding.batch_size,
            embedding_timeout=config.embedding.timeout,
            embedding_max_retries=config.embedding.max_retries,
            build_version=config.knowledge_base.build_version,
        )
        self.manifest_store = ManifestStore(config.database.db_path, ensure_schema=False)
        self.searcher = RAGSearcher(
            db_path=config.database.db_path,
            top_k=config.search.rag_top_k,
            fts_top_k=config.search.fts_top_k,
            vec_top_k=config.search.vec_top_k,
            fusion_rrf_k=config.search.fusion_rrf_k,
            vector_dimension=config.embedding.dimension,
            context_top_k=config.search.context_top_k,
            embedding_provider=config.embedding.provider,
            embedding_base_url=config.embedding.base_url,
            embedding_api_key=config.embedding.api_key,
            embedding_model=config.embedding.model,
            embedding_batch_size=config.embedding.batch_size,
            embedding_timeout=config.embedding.timeout,
            embedding_max_retries=config.embedding.max_retries,
        )

    def run_sync(self, query: str, include_trace: bool = False) -> AgentResponse:
        results, search_trace = self.searcher.search_with_trace(query)
        manifest = self.manifest_store.get_manifest()
        trace: dict[str, Any] = {
            "query": query,
            "search": search_trace,
            "strategy_execution": [],
            "manifest": manifest or {},
        }

        if not results:
            reason = "index_not_ready" if not manifest else "no_retrieval_results"
            answer = (
                "当前索引未就绪，请先执行离线知识库同步。"
                if reason == "index_not_ready"
                else "未从知识库检索到足够相关的内容，请补充更具体的问题后重试。"
            )
            trace["strategy_execution"].append(
                {
                    "stage": "fallback_answer",
                    "reason": reason,
                }
            )
            trace["final_answer_preview"] = answer
            trace["final_citations"] = []
            trace["retrieval_confidence"] = 0.0
            return AgentResponse(
                answer=answer,
                citations=[],
                retrieval_confidence=0.0,
                trace=trace if include_trace else {},
            )

        selected = results[: self.config.search.rag_top_k]
        citations = self._build_citations(selected)
        answer = self._build_human_answer(query=query, selected=selected)
        confidence = min(
            1.0,
            sum(max(0.0, item.score) for item in selected) / max(len(selected), 1),
        )

        trace["strategy_execution"].append(
            {
                "stage": "context_selection",
                "selected_results": [asdict(item) for item in selected],
            }
        )
        trace["final_answer_preview"] = answer
        trace["final_citations"] = citations
        trace["retrieval_confidence"] = confidence
        trace["retrieved_results"] = [asdict(result) for result in results]

        return AgentResponse(
            answer=answer,
            citations=citations,
            retrieval_confidence=confidence,
            trace=trace if include_trace else {},
        )

    def _build_citations(self, selected: list[Any]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for item in selected:
            source = str(getattr(item, "source", "")).strip()
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)
            citations.append(
                {
                    "source": source,
                    "title": source,
                    "path": str(getattr(item, "source_path", "")),
                }
            )
        return citations

    def _build_human_answer(self, *, query: str, selected: list[Any]) -> str:
        query_terms = self._query_terms(query)
        theme = self._detect_theme(query, selected)
        evidence = self._extract_evidence(query=query, selected=selected, limit=6)
        steps = self._build_thematic_steps(theme=theme, query_terms=query_terms)

        lines = [f"问题：{query}", "建议执行步骤："]
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step}")

        if evidence:
            lines.append("关键信息：")
            for sentence in evidence[:3]:
                lines.append(f"- {sentence}")

        lines.append("参考来源：")
        for citation in self._build_citations(selected):
            lines.append(f"- {citation['source']}")
        return "\n".join(lines)

    def _detect_theme(self, query: str, selected: list[Any]) -> str:
        query_text = query.lower()
        source_text = " ".join(str(getattr(item, "source", "")) for item in selected).lower()
        theme_keywords = {
            "product_selection": ["选品", "类目", "需求", "利润", "竞争"],
            "listing": ["listing", "标题", "主图", "关键词", "上架", "五点"],
            "advertising": ["acos", "广告", "ppc", "出价", "投放"],
            "logistics": ["物流", "发货", "fba", "货运", "报关", "装船"],
            "customer_service": ["客服", "消息", "差评", "售后", "回复"],
            "inventory": ["库存", "断货", "补货", "周转", "安全库存"],
            "promotion": ["促销", "折扣", "活动", "秒杀", "优惠"],
            "conversion": ["转化", "点击", "加购", "成交", "评价"],
            "account_security": ["关联", "封号", "账号", "安全", "风控"],
        }
        best_theme = "general"
        best_score = 0
        for theme, keywords in theme_keywords.items():
            query_score = sum(1 for keyword in keywords if keyword in query_text)
            source_score = sum(1 for keyword in keywords if keyword in source_text)
            score = 3 * query_score + source_score
            if score > best_score:
                best_score = score
                best_theme = theme
        return best_theme

    def _build_thematic_steps(self, *, theme: str, query_terms: list[str]) -> list[str]:
        focus = "、".join(query_terms[:3]) if query_terms else "当前问题"
        mapping = {
            "product_selection": [
                "先做市场需求验证，确认目标人群、使用场景和销量趋势。",
                "再评估竞争强度，重点看卖家数量、评价密度和价格带。",
                "核算利润空间，至少覆盖采购、物流、平台费用和广告成本。",
                "检查供应链与合规风险，避免侵权、禁运和交期不稳定。",
                "用小批量测试验证结果，再按数据迭代选品策略。",
            ],
            "listing": [
                "先确定核心关键词与主卖点，明确标题和主图表达重点。",
                "优化标题、五点和详情页，确保信息一致且可搜索。",
                "补齐转化要素，如价格策略、评价素材和信任信息。",
                "上线后跟踪曝光、点击和转化，定位弱项逐项优化。",
            ],
            "advertising": [
                "先拆分投放目标与预算，区分引流词和转化词。",
                "按词分组设置出价与匹配方式，避免高耗低转化。",
                "持续清理无效词，保留高相关高转化词。",
                "结合 ACOS 与利润线做调价，维持可持续投放。",
            ],
            "logistics": [
                "先确认发货模式与时效要求，匹配对应物流方案。",
                "核对包装、标签与报关资料，降低异常和退件风险。",
                "按体积重与实重核算运费，控制单位物流成本。",
                "建立节点追踪和异常预案，保证交付稳定。",
            ],
            "customer_service": [
                "先设定回复时效标准，保证关键消息及时处理。",
                "按问题类型准备标准话术，提高首轮解决率。",
                "针对差评先定位根因，再给出可执行补救方案。",
                "沉淀高频问题与处理结果，持续优化客服 SOP。",
            ],
            "inventory": [
                "先建立安全库存阈值，明确补货触发条件。",
                "按销量趋势和补货周期预测需求，减少断货风险。",
                "将库存分层管理，优先保障核心 SKU。",
                "每周复盘周转与缺货率，动态调整补货计划。",
            ],
            "promotion": [
                "先明确活动目标与人群，再选择合适促销机制。",
                "控制折扣力度与毛利底线，避免只增量不增利。",
                "将活动与广告节奏联动，提升流量承接效率。",
                "活动后复盘转化和利润，保留有效方案。",
            ],
            "conversion": [
                "先定位转化漏斗卡点，区分流量问题与页面问题。",
                "优化主图、卖点和价格策略，提升点击后成交率。",
                "补齐评价与问答等信任要素，降低决策阻力。",
                "按周跟踪转化率和客单价，持续做 A/B 优化。",
            ],
            "account_security": [
                "先梳理账号环境与操作规范，避免高风险行为。",
                "隔离设备、网络与权限，降低关联概率。",
                "建立异常监控与告警，及时处理风控信号。",
                "保留合规证据链，便于申诉与风险复盘。",
            ],
            "general": [
                f"先明确“{focus}”的目标、约束和成功标准。",
                "优先使用可核验的数据或规则做判断，避免仅凭单一片段决策。",
                "把关键结论拆成步骤执行，并在执行后复盘结果再迭代。",
            ],
        }
        return mapping.get(theme, mapping["general"])

    def _extract_evidence(self, *, query: str, selected: list[Any], limit: int) -> list[str]:
        query_terms = self._query_terms(query)
        query_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", query))
        action_markers = ("需要", "建议", "应", "可", "先", "再", "避免", "评估", "分析", "优化")
        scored: list[tuple[float, str]] = []
        seen_sentences: set[str] = set()

        for rank, item in enumerate(selected, start=1):
            content = str(getattr(item, "content", ""))
            if not content:
                continue
            for sentence in self._split_sentences(content):
                if query_has_cjk and not re.search(r"[\u4e00-\u9fff]{2,}", sentence):
                    continue
                key = sentence.lower()
                if key in seen_sentences:
                    continue
                seen_sentences.add(key)
                hit_count = sum(1 for term in query_terms if term and term in sentence.lower())
                marker_bonus = 1 if any(marker in sentence for marker in action_markers) else 0
                source_bonus = max(0.0, 0.5 - 0.1 * (rank - 1))
                length_penalty = 0.2 if len(sentence) > 90 else 0.0
                score = hit_count + marker_bonus + source_bonus - length_penalty
                scored.append((score, sentence))

        scored.sort(key=lambda row: row[0], reverse=True)
        evidence = [sentence for _, sentence in scored[:limit]]
        return evidence

    def _split_sentences(self, text: str) -> list[str]:
        raw_parts = re.split(r"[。！？\n]+", text.replace("\r", "\n"))
        sentences: list[str] = []
        for part in raw_parts:
            line = part.strip()
            if not line:
                continue
            line = re.sub(r"[\t ]+", " ", line)
            if len(line) < 10:
                continue
            if line.startswith(("#", "|", "-", "```", "http")):
                continue
            if line.count("|") >= 2:
                continue
            if not self._is_readable_sentence(line):
                continue
            sentences.append(line)
        return sentences

    def _query_terms(self, query: str) -> list[str]:
        terms = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", query.lower())
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped

    def _is_readable_sentence(self, line: str) -> bool:
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", line)
        if not cleaned:
            return False
        lowered = cleaned.lower()
        noise_markers = ("flatedecode", "xref", "obj", "endobj", "stream", "/filter", "/length")
        if any(marker in lowered for marker in noise_markers):
            return False
        readable = re.findall(r"[A-Za-z0-9\u4e00-\u9fff，。！？；：、（）()《》“”‘’\- ]", cleaned)
        ratio = len(readable) / max(len(cleaned), 1)
        if ratio < 0.75:
            return False
        has_cjk_word = bool(re.search(r"[\u4e00-\u9fff]{2,}", cleaned))
        if has_cjk_word:
            return True
        english_words = re.findall(r"[A-Za-z]{2,}", cleaned)
        return len(english_words) >= 3
