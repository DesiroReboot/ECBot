from __future__ import annotations

from fastapi import APIRouter

from src.fastapi_gateway.services.event_service import FeishuEventService


def create_startup_check_router(event_service: FeishuEventService) -> APIRouter:
    router = APIRouter()

    @router.get("/gateway/startup-check")
    def startup_check() -> dict:
        return event_service.validate_startup()

    return router
