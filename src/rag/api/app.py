"""FastAPI service exposing POST /search."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from rag.api.schemas import SearchRequest, SearchResponse, SearchResult, Scores
from rag.config import get_settings
from rag.retrieve.embedder import Embedder
from rag.retrieve.reranker import Reranker
from rag.retrieve.search import SearchEngine


_engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    settings = get_settings()
    index_dir = settings.index_dir / "default"
    if not (index_dir / "faiss").exists():
        raise RuntimeError(f"No index at {index_dir}. Run scripts/ingest.py first.")

    print("Loading embedder...")
    embedder = Embedder(settings.embedding_model)
    print("Loading reranker...")
    reranker = Reranker(settings.reranker_model)
    print("Loading indices...")
    _engine = SearchEngine(index_dir, embedder, reranker)

    print("Warming up...")
    await _engine.search("warmup query", top_k_rerank=5, top_k_final=2)
    print("API ready.")
    yield


app = FastAPI(title="RAG Retrieval Platform", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not loaded")
    t0 = time.perf_counter()
    hits = await _engine.search(
        query=req.query,
        top_k_retrieve=req.top_k_retrieve,
        top_k_rerank=req.top_k_rerank,
        top_k_final=req.top_k,
        fusion_method=req.fusion_method,
        alpha=req.alpha,
        rrf_k=req.rrf_k,
        use_rerank=req.use_rerank,
        filters=req.filters,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return SearchResponse(
        query=req.query,
        results=[
            SearchResult(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                text=h.text,
                scores=Scores(**h.scores),
                metadata=h.metadata,
            )
            for h in hits
        ],
        latency_ms=round(latency_ms, 2),
        config={
            "fusion_method": req.fusion_method,
            "alpha": req.alpha,
            "use_rerank": req.use_rerank,
            "top_k_rerank": req.top_k_rerank,
            "top_k_retrieve": req.top_k_retrieve,
        },
    )
