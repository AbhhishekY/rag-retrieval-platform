"""Pydantic models for /search request and response."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    top_k_retrieve: int = Field(100, ge=5, le=500)
    top_k_rerank: int = Field(20, ge=1, le=100)
    fusion_method: str = Field("rrf", pattern="^(rrf|weighted|semantic_only|bm25_only)$")
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(60, ge=1, le=500)
    use_rerank: bool = True
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
