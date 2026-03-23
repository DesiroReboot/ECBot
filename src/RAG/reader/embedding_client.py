from __future__ import annotations

from array import array
import hashlib
import json
import math
from time import sleep
from typing import Any
from urllib import parse, request

from src.RAG.config.kbase_config import KBaseConfig


class EmbeddingClient:
    def __init__(self, config: KBaseConfig):
        self.config = config
        self.dimension = int(config.vector_dimension)
        self.provider = config.embedding_provider
        self.base_url = config.embedding_base_url.rstrip("/")
        self.api_key = config.embedding_api_key
        self.remote_model = config.embedding_model
        self.batch_size = max(1, int(config.embedding_batch_size))
        self.timeout = max(1, int(config.embedding_timeout))
        self.max_retries = max(0, int(config.embedding_max_retries))
        self.use_remote = self._should_use_remote()
        self.model_name = self.remote_model if self.use_remote else f"{self.remote_model}@local-sha256"

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else self._local_vector("")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.use_remote:
            try:
                return self._embed_texts_remote(texts)
            except Exception:
                # Query/index path can degrade to lexical-only when network is unavailable.
                return [self._local_vector(text) for text in texts]
        return [self._local_vector(text) for text in texts]

    def serialize(self, vector: list[float]) -> bytes:
        return array("f", vector).tobytes()

    def deserialize(self, payload: bytes | bytearray | memoryview) -> list[float]:
        buf = array("f")
        buf.frombytes(bytes(payload))
        values = list(buf)
        if len(values) >= self.dimension:
            return values[: self.dimension]
        return values + [0.0] * (self.dimension - len(values))

    def cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 <= 1e-12 or n2 <= 1e-12:
            return 0.0
        return dot / (n1 * n2)

    def _should_use_remote(self) -> bool:
        if self.provider.lower() in {"mock", "local"}:
            return False
        if not self.base_url:
            return False
        if not self.api_key:
            return False
        return True

    def _embed_texts_remote(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_vectors.extend(self._embed_batch_remote(batch))
        return all_vectors

    def _embed_batch_remote(self, texts: list[str]) -> list[list[float]]:
        url = self._safe_http_url(f"{self.base_url}/embeddings")
        payload = json.dumps(
            {
                "model": self.remote_model,
                "input": texts,
                "encoding_format": "float",
            }
        ).encode("utf-8")

        req = request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                    body = resp.read().decode("utf-8", errors="ignore")
                    data = json.loads(body)
                    rows = data.get("data", [])
                    vectors = [self._normalize_vector(row.get("embedding", [])) for row in rows]
                    if len(vectors) != len(texts):
                        raise RuntimeError("remote embedding response size mismatch")
                    return vectors
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    sleep(min(1.5 * (attempt + 1), 4.0))
                    continue
                break
        raise RuntimeError(f"remote embedding failed: {last_error}")

    def _safe_http_url(self, raw_url: str) -> str:
        parsed = parse.urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported url scheme: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError("missing url host")
        return raw_url

    def _local_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        seed = digest
        while len(values) < self.dimension:
            seed = hashlib.sha256(seed).digest()
            for idx in range(0, len(seed), 4):
                chunk = seed[idx : idx + 4]
                if len(chunk) < 4:
                    continue
                integer = int.from_bytes(chunk, byteorder="little", signed=False)
                values.append((integer / 2**32) * 2.0 - 1.0)
                if len(values) >= self.dimension:
                    break
        return self._l2_normalize(values)

    def _normalize_vector(self, vec: list[Any]) -> list[float]:
        values = [float(v) for v in vec][: self.dimension]
        if len(values) < self.dimension:
            values.extend([0.0] * (self.dimension - len(values)))
        return self._l2_normalize(values)

    def _l2_normalize(self, vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-12:
            return vec
        return [v / norm for v in vec]
