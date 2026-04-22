"""Tests for score fusion (RRF and weighted-alpha)."""
from rag.retrieve.fusion import rrf_fuse, weighted_alpha_fuse


def test_rrf_single_list_ranking_preserved():
    bm25 = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
    dense = []
    fused = rrf_fuse(bm25, dense, k=60, top_k=10)
    assert [cid for cid, _ in fused] == ["a", "b", "c"]


def test_rrf_combines_both_lists():
    bm25 = [("a", 10.0), ("b", 9.0), ("c", 8.0)]
    dense = [("b", 0.9), ("d", 0.85), ("a", 0.8)]
    fused = rrf_fuse(bm25, dense, k=60, top_k=10)
    fused_ids = [cid for cid, _ in fused]
    assert fused_ids[0] == "b"
    assert set(fused_ids) == {"a", "b", "c", "d"}


def test_rrf_handles_empty_inputs():
    assert rrf_fuse([], [], k=60, top_k=5) == []


def test_weighted_alpha_extremes_correct():
    bm25 = [("a", 100.0), ("b", 50.0)]
    dense = [("b", 0.99), ("a", 0.01)]
    assert [c for c, _ in weighted_alpha_fuse(bm25, dense, alpha=1.0, top_k=2)] == ["a", "b"]
    assert [c for c, _ in weighted_alpha_fuse(bm25, dense, alpha=0.0, top_k=2)] == ["b", "a"]


def test_weighted_alpha_missing_ids_treated_as_zero():
    bm25 = [("b", 10.0)]
    dense = [("a", 0.99), ("b", 0.5)]
    out = weighted_alpha_fuse(bm25, dense, alpha=0.3, top_k=2)
    assert len(out) == 2
    assert {c for c, _ in out} == {"a", "b"}
