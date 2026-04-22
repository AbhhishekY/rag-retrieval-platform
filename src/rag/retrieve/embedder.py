"""FastEmbed (ONNX) embedder for query + doc vectors.

Uses sentence-transformers/all-MiniLM-L6-v2 weights via ONNX Runtime — no
PyTorch dependency. Outputs 384-dim L2-normalized vectors (so cosine
similarity == inner product in FAISS IndexFlatIP).
"""
from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)
        probe = next(iter(self.model.embed(["probe"])))
        self.dim = int(np.asarray(probe).shape[-1])

    def encode_query(self, query: str) -> np.ndarray:
        vec = next(iter(self.model.embed([query])))
        return np.asarray(vec, dtype=np.float32)

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = list(self.model.embed(texts, batch_size=batch_size))
        return np.vstack(vecs).astype(np.float32)
