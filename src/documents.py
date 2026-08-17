from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None

    @property
    def label(self) -> str:
        return f"{self.source}" + (f", page {self.page}" if self.page else "")


def _chunk_text(text: str, source: str, page: int | None, size: int = 180, overlap: int = 35) -> list[Chunk]:
    words = text.split()
    chunks: list[Chunk] = []
    for start in range(0, len(words), size - overlap):
        part = " ".join(words[start : start + size]).strip()
        if len(part) >= 80:
            chunks.append(Chunk(text=part, source=source, page=page))
    return chunks


def _read_pdf(file) -> Iterable[Chunk]:
    reader = PdfReader(file)
    for page_number, page in enumerate(reader.pages, start=1):
        yield from _chunk_text(page.extract_text() or "", file.name, page_number)


def load_documents(files) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file in files:
        suffix = Path(file.name).suffix.lower()
        file.seek(0)
        if suffix == ".pdf":
            chunks.extend(_read_pdf(file))
        else:
            chunks.extend(_chunk_text(file.getvalue().decode("utf-8", errors="ignore"), file.name, None))
    return chunks
