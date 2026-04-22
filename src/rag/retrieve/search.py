"""Async search pipeline: BM25 + dense retrieval in parallel -> fusion -> rerank -> top-k.

Holds indices + models in memory (load once per process). Query path kicks
off embed + BM25 on a threadpool so the two CPU-bound retrievers run in
parallel instead of serializing on the event loop.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rag.index.bm25_index import BM25Index
from rag.index.vector_index import FaissFlatIndex
from rag.retrieve.embedder import Embedder
from rag.retrieve.fusion import rrf_fuse, weighted_alpha_fuse
from rag.retrieve.reranker import Reranker
from rag.types import SearchHit


class SearchEngine:
    def __init__(
        self,
        index_dir: Path,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ):
        self.bm25 = BM25Index.load(index_dir / "bm25.pkl")
        self.vector = FaissFlatIndex.load(index_dir / "faiss")
        self.embedder = embedder
        self.reranker = reranker
        self.chunk_map = self._load_chunks(index_dir / "chunks.jsonl")

    @staticmethod
    def _load_chunks(path: Path) -> dict[str, dict]:
        chunks: dict[str, dict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                chunks[row["chunk_id"]] = row
        return chunks

    async def search(
        self,
        query: str,
        top_k_retrieve: int = 100,
        top_k_rerank: int = 20,
        top_k_final: int = 5,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        use_rerank: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        loop = asyncio.get_running_loop()

        embed_fut = loop.run_in_executor(None, self.embedder.encode_query, query)
        bm25_fut = loop.run_in_executor(None, self.bm25.search, query, top_k_retrieve)
        query_vec, bm25_hits = await asyncio.gather(embed_fut, bm25_fut)

        dense_hits = self.vector.search(query_vec, k=top_k_retrieve)

        if fusion_method == "rrf":
            fused = rrf_fuse(bm25_hits, dense_hits, k=rrf_k, top_k=top_k_rerank)
        elif fusion_method == "weighted":
            fused = weighted_alpha_fuse(bm25_hits, dense_hits, alpha=alpha, top_k=top_k_rerank)
        elif fusion_method == "semantic_only":
            fused = dense_hits[:top_k_rerank]
        elif fusion_method == "bm25_only":
            fused = bm25_hits[:top_k_rerank]
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")

        bm25_by_id = dict(bm25_hits)
        dense_by_id = dict(dense_hits)
        fused_by_id = dict(fused)

        if filters:
            fused = [
                (cid, s) for cid, s in fused
                if _metadata_matches(self.chunk_map.get(cid, {}).get("metadata", {}), filters)
            ]

        if use_rerank and self.reranker is not None and fused:
            candidates = [
                (cid, self.chunk_map[cid]["text"]) for cid, _ in fused if cid in self.chunk_map
            ]
            reranked = await loop.run_in_executor(
                None, self.reranker.rerank, query, candidates, top_k_final
            )
        else:
            reranked = [(cid, 0.0) for cid, _ in fused[:top_k_final]]

        rerank_by_id = dict(reranked)

        hits: list[SearchHit] = []
        for cid, _ in reranked:
            chunk = self.chunk_map.get(cid)
            if not chunk:
                continue
            final_score = (
                rerank_by_id[cid]
                if use_rerank and self.reranker is not None
                else fused_by_id.get(cid, 0.0)
            )
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    doc_id=chunk["doc_id"],
                    text=chunk["text"],
                    scores={
                        "bm25": bm25_by_id.get(cid, 0.0),
                        "semantic": dense_by_id.get(cid, 0.0),
                        "hybrid_fused": fused_by_id.get(cid, 0.0),
                        "rerank": rerank_by_id.get(cid, 0.0) if use_rerank else 0.0,
                        "final": final_score,
                    },
                    metadata=chunk.get("metadata", {}),
                )
            )
        return hits


def _metadata_matches(meta: dict, filters: dict) -> bool:
    for key, want in filters.items():
        got = meta.get(key)
        if isinstance(want, list):
            if got not in want:
                return False
        else:
            if got != want:
                return False
    return True
