from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.config import Config
from src.fastapi_gateway.app import create_app


class _DummyAgent:
    def run_sync(self, query: str, include_trace: bool = False):  # noqa: ARG002
        class _Resp:
            answer = "ok"
            retrieval_confidence = 1.0
            trace = {"search": {"fts_hits": 1, "vec_hits": 1}}

        if include_trace:
            return _Resp()
        return _Resp()


class _DummyAPIClient:
    def __init__(self) -> None:
        self._dialog = {"ok": True, "missing_tokens": [], "message": ""}
        self.reply_calls = 0
        self.message_calls = 0

    def token_dialog_payload(self):
        return self._dialog

    def openapi_base_url_ok(self):
        return True

    def validate_credentials(self):
        return {
            "ok": True,
            "error": "",
            "feishu_code": 0,
            "feishu_msg": "",
            "expire": 7200,
            "tenant_access_token_ready": True,
        }

    def send_reply_text(self, *, message_id: str, text: str):  # noqa: ARG002
        self.reply_calls += 1
        class _Resp:
            ok = True
            error = ""
            data = {}

        return _Resp()

    def send_message_text(self, *, receive_id: str, text: str, receive_id_type: str = "chat_id"):  # noqa: ARG002
        self.message_calls += 1
        class _Resp:
            ok = True
            error = ""
            data = {}

        return _Resp()


def _build_config(tmp_path: Path, extra: dict | None = None) -> Config:
    payload = {
        "gateway": {
            "feishu": {
                "enabled": True,
                "receive_mode": "webhook",
                "openapi_base_url": "https://open.feishu.cn/open-apis",
                "app_id": "app-id",
                "app_secret": "app-secret",
                "verification_token": "verify-token",
                "webhook_path": "/webhook/feishu",
            }
        }
    }
    if extra:
        payload["gateway"]["feishu"].update(extra["gateway"]["feishu"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return Config(str(path))


def test_create_app_rejects_long_connection_mode(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path, extra={"gateway": {"feishu": {"receive_mode": "long_connection"}}})
    with pytest.raises(RuntimeError, match="receive_mode=long_connection"):
        create_app(cfg)


def test_health_and_startup_check(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    startup = client.get("/gateway/startup-check")
    assert startup.status_code == 200
    payload = startup.json()
    assert payload["webhook_path"] == "/webhook/feishu"
    assert payload["credential_validation"]["ok"] is True


def test_self_check_and_fullchain_visualize(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    self_check = client.get("/gateway/self-check")
    assert self_check.status_code == 200
    self_payload = self_check.json()
    assert self_payload["summary"]["total"] >= 1
    assert isinstance(self_payload["checks"], list)
    self_check_ts = datetime.fromisoformat(self_payload["timestamp"])
    assert self_check_ts.tzinfo is not None
    assert self_check_ts.utcoffset() == timedelta(0)

    fullchain = client.post("/gateway/fullchain-visualize", json={"query": "hello"})
    assert fullchain.status_code == 200
    chain_payload = fullchain.json()
    assert chain_payload["ok"] is True
    assert chain_payload["query"] == "hello"
    assert isinstance(chain_payload["stages"], list)
    fullchain_ts = datetime.fromisoformat(chain_payload["timestamp"])
    assert fullchain_ts.tzinfo is not None
    assert fullchain_ts.utcoffset() == timedelta(0)


def test_url_verification_token_check(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    bad = client.post(
        "/webhook/feishu",
        json={"type": "url_verification", "challenge": "abc", "token": "bad-token"},
    )
    assert bad.status_code == 200
    assert bad.json()["success"] is False
    assert bad.json()["fallback_type"] == "verification_token_invalid"

    ok = client.post(
        "/webhook/feishu",
        json={"type": "url_verification", "challenge": "abc", "token": "verify-token"},
    )
    assert ok.status_code == 200
    assert ok.json() == {"challenge": "abc"}


def test_event_reply_pipeline(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    resp = client.post(
        "/webhook/feishu",
        json={
            "type": "event_callback",
            "event": {
                "message": {"message_id": "om_xxx", "text": "hello"},
                "sender": {"sender_type": "user"},
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["reply_ok"] is True


def test_duplicate_event_callback_is_ignored(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    app = create_app(cfg)
    app.state.event_service.agent = _DummyAgent()
    app.state.event_service.api_client = _DummyAPIClient()

    client = TestClient(app)
    event_payload = {
        "type": "event_callback",
        "event_id": "evt_dedup_001",
        "event": {
            "message": {"message_id": "om_dup_001", "text": "hello"},
            "sender": {"sender_type": "user"},
        },
    }
    first = client.post("/webhook/feishu", json=event_payload)
    second = client.post("/webhook/feishu", json=event_payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["success"] is True
    assert first.json()["reply_ok"] is True
    assert second.json()["success"] is True
    assert second.json()["message"] == "ignored_duplicate_event"
    assert second.json()["duplicate"] is True
    assert app.state.event_service.api_client.reply_calls == 1
