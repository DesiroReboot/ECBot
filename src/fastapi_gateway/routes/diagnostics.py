from __future__ import annotations

from fastapi import APIRouter, Body

from src.fastapi_gateway.services.event_service import FeishuEventService


def create_diagnostics_router(event_service: FeishuEventService) -> APIRouter:
    router = APIRouter()

    @router.get("/gateway/self-check")
    def self_check() -> dict:
        return event_service.run_self_check()

    @router.post("/gateway/fullchain-visualize")
    def fullchain_visualize(payload: dict | None = Body(default=None)) -> dict:
        body = payload or {}
        query = str(body.get("query", "")).strip()
        return event_service.visualize_fullchain(query)

    return router
