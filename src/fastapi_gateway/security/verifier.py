from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Any


class FeishuAuthVerifier:
    @staticmethod
    def verify_signature(
        *,
        headers: dict[str, Any],
        raw_body: bytes,
        encrypt_key: str,
    ) -> bool:
        if not encrypt_key:
            return True

        timestamp = str(headers.get("X-Lark-Request-Timestamp", "")).strip()
        nonce = str(headers.get("X-Lark-Request-Nonce", "")).strip()
        signature = str(headers.get("X-Lark-Signature", "")).strip()
        if not timestamp or not nonce or not signature:
            return False

        payload = f"{timestamp}{nonce}{encrypt_key}{raw_body.decode('utf-8')}"
        digest = hmac.new(
            encrypt_key.encode("utf-8"),
            payload.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        expected = base64.b64encode(digest).decode("utf-8")
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def verify_verification_token(payload: dict[str, Any], expected_token: str) -> bool:
        if not expected_token:
            return True
        token = str(payload.get("token", "")).strip()
        return hmac.compare_digest(token, expected_token)
