"""Pydantic models for /search request and response."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag import constants as C


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"query": "What did Goldman Sachs report about quarterly earnings?"}
        }
    )

    query: str = Field(..., min_length=1)
    top_k: int = Field(C.TOP_K_FINAL, ge=1, le=50)
    top_k_retrieve: int = Field(C.TOP_K_RETRIEVE, ge=5, le=500)
    top_k_rerank: int = Field(C.TOP_K_RERANK, ge=1, le=100)
    fusion_method: str = Field(C.FUSION_METHOD, pattern="^(rrf|weighted|semantic_only|bm25_only)$")
    alpha: float = Field(C.HYBRID_ALPHA, ge=0.0, le=1.0)
    rrf_k: int = Field(C.RRF_K, ge=1, le=500)
    use_rerank: bool = C.USE_RERANK_DEFAULT
    filters: dict[str, Any] | None = None


class Scores(BaseModel):
    bm25: float
    semantic: float
    hybrid_fused: float
    rerank: float
    final: float


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    scores: Scores
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    latency_ms: float
    config: dict[str, Any]
