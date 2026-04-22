"""Tests for retrieval metrics: precision@k, recall@k, NDCG@k."""
import math

from rag.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k


def test_precision_all_relevant():
    assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_precision_none_relevant():
    assert precision_at_k(["a", "b"], {"c"}, k=2) == 0.0


def test_precision_partial():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == 0.5


def test_precision_at_k_truncates():
    assert precision_at_k(["a", "b", "x", "y"], {"a", "b", "x"}, k=2) == 1.0


def test_recall_at_k():
    assert recall_at_k(["a", "x", "b", "y", "z"], {"a", "b", "c"}, k=5) == 2 / 3


def test_recall_at_k_no_relevant_returns_zero():
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_ndcg_zero_when_no_hits():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_ndcg_partial():
    val = ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3)
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert abs(val - expected) < 1e-6
