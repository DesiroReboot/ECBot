from __future__ import annotations

from pathlib import Path


class FileScanner:
    def __init__(self, supported_extensions: tuple[str, ...]):
        self.supported_extensions = tuple(ext.lower() for ext in supported_extensions)

    def scan(self, source_dir: str) -> list[Path]:
        root = Path(source_dir)
        if not root.exists() or not root.is_dir():
            return []

        files: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.supported_extensions:
                continue
            files.append(path)
        files.sort(key=lambda p: str(p).lower())
        return files
