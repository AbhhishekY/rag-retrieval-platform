"""FAISS IndexFlatIP wrapper for cosine similarity over L2-normalized vectors.

Exhaustive inner-product search. For L2-normalized vectors this equals cosine.
Optimal for <100K vectors — brute force is faster than HNSW/ANN here because
there's no graph-traversal overhead and zero recall loss.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np


class FaissFlatIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunk_ids: list[str] = []

    def build(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        assert vectors.dtype == np.float32, f"expected float32, got {vectors.dtype}"
        assert vectors.shape == (len(chunk_ids), self.dim), (
            f"shape mismatch: vectors={vectors.shape} expected ({len(chunk_ids)}, {self.dim})"
        )
        self.chunk_ids = list(chunk_ids)
        self.index.add(vectors)

    def search(self, query_vec: np.ndarray, k: int = 100) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vec.astype(np.float32), k)
        return [
            (self.chunk_ids[indices[0][i]], float(scores[0][i]))
            for i in range(k)
            if indices[0][i] != -1
        ]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "faiss.index"))
        with open(dir_path / "chunk_ids.pkl", "wb") as f:
            pickle.dump(self.chunk_ids, f)

    @classmethod
    def load(cls, dir_path: Path) -> "FaissFlatIndex":
        index = faiss.read_index(str(dir_path / "faiss.index"))
        with open(dir_path / "chunk_ids.pkl", "rb") as f:
            chunk_ids = pickle.load(f)
        inst = cls(index.d)
        inst.index = index
        inst.chunk_ids = chunk_ids
        return inst
