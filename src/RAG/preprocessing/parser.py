from __future__ import annotations

from pathlib import Path
from typing import Any

from src.RAG.config.kbase_config import KBaseConfig


class DocumentParser:
    CODE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".cs",
        ".php",
    }

    def __init__(self, config: KBaseConfig):
        self.config = config

    def parse(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self._parse_pdf(file_path)
        if ext in self.CODE_EXTENSIONS:
            content = self._read_text(file_path)
            return content, {"type": "code", "language": ext.lstrip(".")}

        content = self._read_text(file_path)
        return content, {"type": "text", "language": "plain"}

    def extract_text_chunks(self, content: str, chunk_size: int, overlap: int) -> list[str]:
        if not content:
            return []
        step = max(1, chunk_size - overlap)
        return [content[i : i + chunk_size] for i in range(0, len(content), step)]

    def _read_text(self, file_path: Path) -> str:
        if not file_path.exists():
            return ""

        for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                return file_path.read_text(encoding=encoding, errors="strict")
            except UnicodeDecodeError:
                continue
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _parse_pdf(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        # MVP: keep deterministic behavior without mandatory PDF dependencies.
        if not file_path.exists():
            return "", {"type": "pdf", "language": "pdf"}
        # Best-effort extraction: many text PDFs still contain readable streams.
        raw = file_path.read_bytes()
        text = raw.decode("utf-8", errors="ignore")
        return text, {"type": "pdf", "language": "pdf"}
