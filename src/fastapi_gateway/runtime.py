from __future__ import annotations

from src.config import Config
from src.fastapi_gateway.app import create_app
from src.fastapi_gateway.long_connection import run_long_connection_client


def resolve_receive_mode(config: Config) -> str:
    mode = str(config.gateway.feishu.receive_mode or "").strip().lower()
    if mode not in {"long_connection", "webhook"}:
        return "long_connection"
    return mode


def run_webhook_server(config: Config | None = None) -> None:
    cfg = config or Config()
    app = create_app(cfg)

    # Keep a single FastAPI runtime path while long_connection mode is deprecated.
    import uvicorn

    uvicorn.run(
        app,
        host=str(cfg.gateway.feishu.webhook_host or "127.0.0.1"),
        port=cfg.gateway.feishu.webhook_port,
    )


def run_gateway(config: Config | None = None) -> None:
    cfg = config or Config()
    mode = resolve_receive_mode(cfg)
    if mode == "long_connection":
        run_long_connection_client(cfg)
        return
    run_webhook_server(cfg)
