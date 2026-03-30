from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from src.config import Config


def _write_config(payload: dict) -> Path:
    base = Path("DB") / "tmp_runtime" / f"config-test-{uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_search_config_reads_tavily_env_and_provider(monkeypatch) -> None:
    config_path = _write_config(
        {
            "search": {
                "web_search_provider": "mock",
                "web_search_max_results": 6,
            }
        },
    )
    monkeypatch.setenv("ECBOT_WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("ECBOT_TAVILY_API_KEY", "env-tavily-key")
    monkeypatch.setenv("ECBOT_WEB_SEARCH_MAX_RESULTS", "12")
    monkeypatch.setenv("ECBOT_WEB_SEARCH_DEPTH", "advanced")
    monkeypatch.setenv("ECBOT_WEB_SEARCH_RETRIES", "3")
    monkeypatch.setenv("ECBOT_PHASE_A_RAG_CONFIDENCE_THRESHOLD", "0.66")
    monkeypatch.setenv("ECBOT_L1_TRIGGER_THRESHOLD", "0.72")
    monkeypatch.setenv("ECBOT_L1_TEMPLATE_ENABLED", "false")
    monkeypatch.setenv("ECBOT_L2_MAX_TOP_K", "11")
    monkeypatch.setenv("ECBOT_MERGE_WEB_TRIGGER_REQUIRES_RAG_GAP", "false")
    monkeypatch.setenv("ECBOT_MERGE_TRIGGER_ON_KB_EMPTY", "true")
    monkeypatch.setenv("ECBOT_MERGE_TRIGGER_ON_LOW_CONFIDENCE", "false")
    monkeypatch.setenv("ECBOT_MERGE_STEP_MIN_EVIDENCE", "2")
    monkeypatch.setenv("ECBOT_MERGE_EVIDENCE_RAG_TOP_K", "4")
    monkeypatch.setenv("ECBOT_MERGE_EVIDENCE_SEARCH_TOP_K", "3")
    monkeypatch.setenv("ECBOT_KB_AUTO_INIT_ON_STARTUP", "true")
    monkeypatch.setenv("ECBOT_KB_INIT_BLOCKING", "true")
    monkeypatch.setenv("ECBOT_KB_INIT_FAIL_OPEN", "false")

    cfg = Config(str(config_path))

    assert cfg.search.web_search_provider == "tavily"
    assert cfg.search.web_search_tavily_api_key == "env-tavily-key"
    assert cfg.search.web_search_max_results == 12
    assert cfg.search.web_search_depth == "advanced"
    assert cfg.search.web_search_retries == 3
    assert cfg.search.phase_a_rag_confidence_threshold == 0.66
    assert cfg.search.l1_trigger_threshold == 0.72
    assert cfg.search.l1_template_enabled is False
    assert cfg.search.l2_max_top_k == 11
    assert cfg.search.merge_web_trigger_requires_rag_gap is False
    assert cfg.search.merge_trigger_on_kb_empty is True
    assert cfg.search.merge_trigger_on_low_confidence is False
    assert cfg.search.merge_step_min_evidence == 2
    assert cfg.search.merge_evidence_rag_top_k == 4
    assert cfg.search.merge_evidence_search_top_k == 3
    assert cfg.knowledge_base.auto_init_on_startup is True
    assert cfg.knowledge_base.init_blocking is True
    assert cfg.knowledge_base.init_fail_open is False


def test_search_config_uses_defaults_when_values_missing(monkeypatch) -> None:
    config_path = _write_config({})
    monkeypatch.setenv("ECBOT_TAVILY_API_KEY", "")
    monkeypatch.delenv("ECBOT_WEB_SEARCH_DEPTH", raising=False)
    monkeypatch.delenv("ECBOT_WEB_SEARCH_MAX_RESULTS", raising=False)
    monkeypatch.delenv("ECBOT_WEB_SEARCH_RETRIES", raising=False)
    monkeypatch.delenv("ECBOT_PHASE_A_RAG_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("ECBOT_L1_TRIGGER_THRESHOLD", raising=False)
    monkeypatch.delenv("ECBOT_L1_TEMPLATE_ENABLED", raising=False)
    monkeypatch.delenv("ECBOT_L2_MAX_TOP_K", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_WEB_TRIGGER_REQUIRES_RAG_GAP", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_TRIGGER_ON_KB_EMPTY", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_TRIGGER_ON_LOW_CONFIDENCE", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_STEP_MIN_EVIDENCE", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_EVIDENCE_RAG_TOP_K", raising=False)
    monkeypatch.delenv("ECBOT_MERGE_EVIDENCE_SEARCH_TOP_K", raising=False)
    monkeypatch.delenv("ECBOT_KB_AUTO_INIT_ON_STARTUP", raising=False)
    monkeypatch.delenv("ECBOT_KB_INIT_BLOCKING", raising=False)
    monkeypatch.delenv("ECBOT_KB_INIT_FAIL_OPEN", raising=False)

    cfg = Config(str(config_path))

    assert cfg.search.web_search_tavily_api_key == ""
    assert cfg.search.web_search_tavily_base_url == "https://api.tavily.com/search"
    assert cfg.search.web_search_max_results == 8
    assert cfg.search.web_search_depth == "basic"
    assert cfg.search.web_search_retries == 1
    assert cfg.search.phase_a_rag_confidence_threshold == 0.58
    assert cfg.search.l1_trigger_threshold == 0.58
    assert cfg.search.l1_template_enabled is True
    assert cfg.search.l2_max_top_k == 8
    assert cfg.search.merge_web_trigger_requires_rag_gap is True
    assert cfg.search.merge_trigger_on_kb_empty is True
    assert cfg.search.merge_trigger_on_low_confidence is True
    assert cfg.search.merge_step_min_evidence == 1
    assert cfg.search.merge_evidence_rag_top_k == 3
    assert cfg.search.merge_evidence_search_top_k == 2
    assert cfg.knowledge_base.auto_init_on_startup is False
    assert cfg.knowledge_base.init_blocking is False
    assert cfg.knowledge_base.init_fail_open is True
