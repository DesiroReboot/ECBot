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

    cfg = Config(str(config_path))

    assert cfg.search.web_search_provider == "tavily"
    assert cfg.search.web_search_tavily_api_key == "env-tavily-key"
    assert cfg.search.web_search_max_results == 12
    assert cfg.search.web_search_depth == "advanced"
    assert cfg.search.web_search_retries == 3


def test_search_config_uses_defaults_when_values_missing(monkeypatch) -> None:
    config_path = _write_config({})
    monkeypatch.setenv("ECBOT_TAVILY_API_KEY", "")
    monkeypatch.delenv("ECBOT_WEB_SEARCH_DEPTH", raising=False)
    monkeypatch.delenv("ECBOT_WEB_SEARCH_MAX_RESULTS", raising=False)
    monkeypatch.delenv("ECBOT_WEB_SEARCH_RETRIES", raising=False)

    cfg = Config(str(config_path))

    assert cfg.search.web_search_tavily_api_key == ""
    assert cfg.search.web_search_tavily_base_url == "https://api.tavily.com/search"
    assert cfg.search.web_search_max_results == 8
    assert cfg.search.web_search_depth == "basic"
    assert cfg.search.web_search_retries == 1
