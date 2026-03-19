from __future__ import annotations

import re


class Chunker:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        if not text:
            return []

        units = self._semantic_units(text)
        if not units:
            return []

        chunks: list[str] = []
        current = ""
        for unit in units:
            if not unit:
                continue
            if not current:
                current = unit
                continue
            if len(current) + 1 + len(unit) <= self.chunk_size:
                current = f"{current}\n{unit}"
                continue

            chunks.append(current)
            overlap = current[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
            overlap = overlap.strip()
            current = f"{overlap}\n{unit}".strip() if overlap else unit
            while len(current) > self.chunk_size:
                chunks.append(current[: self.chunk_size])
                tail = current[self.chunk_size - self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current = tail.strip()
        if current:
            chunks.append(current)

        return [chunk for chunk in chunks if chunk.strip()]

    def _semantic_units(self, text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        paragraphs = [para.strip() for para in normalized.split("\n\n") if para.strip()]

        units: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                units.append(paragraph)
                continue
            sentences = self._split_sentences(paragraph)
            if not sentences:
                sentences = [paragraph]
            units.extend(self._rebalance_long_sentences(sentences))
        return units

    def _split_sentences(self, paragraph: str) -> list[str]:
        parts = re.split(r"(?<=[。！？!?；;])\s+|(?<=[。！？!?；;])(?=[^\s])", paragraph)
        sentences = [part.strip() for part in parts if part.strip()]
        if not sentences:
            return [paragraph.strip()] if paragraph.strip() else []
        return sentences

    def _rebalance_long_sentences(self, sentences: list[str]) -> list[str]:
        units: list[str] = []
        for sentence in sentences:
            if len(sentence) <= self.chunk_size:
                units.append(sentence)
                continue
            words = re.split(r"(\s+)", sentence)
            current = ""
            for word in words:
                if not word:
                    continue
                if len(current) + len(word) <= self.chunk_size:
                    current += word
                    continue
                if current.strip():
                    units.append(current.strip())
                current = word.strip()
            if current.strip():
                units.append(current.strip())
        return units
