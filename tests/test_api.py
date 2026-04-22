"""Integration test for /search. Skipped unless an index exists at indices/default."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


INDEX_PATH = Path("indices/default/faiss")
SKIP_REASON = "Integration test needs an index at indices/default. Run scripts/ingest.py."


@pytest.mark.skipif(not INDEX_PATH.exists(), reason=SKIP_REASON)
def test_search_endpoint_returns_score_breakdown():
    from rag.api.app import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["engine_loaded"] is True

        r = client.post("/search", json={"query": "climate change impact", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "climate change impact"
        assert len(body["results"]) <= 3
        assert body["latency_ms"] > 0
        assert "fusion_method" in body["config"]
        if body["results"]:
            hit = body["results"][0]
            assert set(hit["scores"].keys()) == {"bm25", "semantic", "hybrid_fused", "rerank", "final"}
            assert "chunk_id" in hit and "doc_id" in hit and "text" in hit
            assert isinstance(hit["metadata"], dict)


@pytest.mark.skipif(not INDEX_PATH.exists(), reason=SKIP_REASON)
def test_search_with_filter_and_no_rerank():
    from rag.api.app import app

    with TestClient(app) as client:
        r = client.post("/search", json={
            "query": "business news",
            "top_k": 2,
            "use_rerank": False,
            "filters": {"category": "business"},
        })
        assert r.status_code == 200
        body = r.json()
        for hit in body["results"]:
            assert hit["metadata"].get("category") == "business"
