"""Cross-encoder reranker via FastEmbed (ONNX, no PyTorch)."""
from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag import constants as C


class Reranker:
    def __init__(self, model_name: str = C.RERANKER_MODEL):
        self.model_name = model_name
        self.model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int = C.TOP_K_FINAL,
        batch_size: int = C.RERANK_BATCH_SIZE,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        docs = [text for _, text in candidates]
        scores = list(self.model.rerank(query, docs, batch_size=batch_size))
        scored = [(cid, float(s)) for (cid, _), s in zip(candidates, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
