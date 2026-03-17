from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.core.search.fts_retriever import FTSRetriever
from src.core.search.vec_retriever import VecRetriever
from src.RAG.config.kbase_config import KBaseConfig


class HybridRetriever:
    def __init__(self, db_path: str, config: KBaseConfig):
        self.fts = FTSRetriever(db_path)
        self.vec = VecRetriever(db_path, config)

    def retrieve(
        self,
        *,
        query: str,
        fts_limit: int,
        vec_limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        fts_results: list[dict[str, Any]] = []
        vec_results: list[dict[str, Any]] = []
        vec_meta: dict[str, Any] = {}
        branch_errors: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            fts_future = executor.submit(self.fts.retrieve, query, fts_limit)
            vec_future = executor.submit(self.vec.retrieve, query, vec_limit)
            try:
                fts_results = fts_future.result()
            except Exception as exc:
                branch_errors["fts"] = str(exc)
                fts_results = []
            try:
                vec_results, vec_meta = vec_future.result()
            except Exception as exc:
                branch_errors["vec"] = str(exc)
                vec_results = []
                vec_meta = {"candidate_pool": 0, "error": str(exc)}
        return fts_results, vec_results, {"vector_meta": vec_meta, "branch_errors": branch_errors}
