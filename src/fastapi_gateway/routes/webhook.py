from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.fastapi_gateway.services.event_service import FeishuEventService


def create_webhook_router(path: str, event_service: FeishuEventService) -> APIRouter:
    router = APIRouter()

    @router.post(path)
    async def feishu_webhook(request: Request) -> JSONResponse:
        body = await request.body()
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            return JSONResponse(status_code=400, content={"error": "invalid_json"})

        headers = {
            "X-Lark-Request-Timestamp": request.headers.get("X-Lark-Request-Timestamp", ""),
            "X-Lark-Request-Nonce": request.headers.get("X-Lark-Request-Nonce", ""),
            "X-Lark-Signature": request.headers.get("X-Lark-Signature", ""),
        }
        result = event_service.handle_event(payload, headers=headers, raw_body=body)
        return JSONResponse(status_code=200, content=result)

    return router
