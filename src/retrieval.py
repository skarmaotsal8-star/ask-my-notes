from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.documents import Chunk


class KnowledgeBase:
    """In-memory FAISS index for a user-uploaded set of note chunks."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []
        self.stats = {"documents": 0, "chunks": 0}

    def build(self, chunks: list[Chunk]) -> None:
        vectors = self.model.encode([chunk.text for chunk in chunks], convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks
        self.stats = {"documents": len({chunk.source for chunk in chunks}), "chunks": len(chunks)}

    def search(self, question: str, k: int = 5) -> list[dict]:
        if not self.index:
            raise RuntimeError("Index your notes before asking a question.")
        query = self.model.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query)
        scores, indexes = self.index.search(query, min(k, len(self.chunks)))
        return [
            {
                "id": position + 1,
                "label": self.chunks[index].label,
                "text": self.chunks[index].text,
                "preview": self.chunks[index].text[:280] + "…",
                "score": float(score),
            }
            for position, (score, index) in enumerate(zip(scores[0], indexes[0]))
            if index >= 0
        ]
