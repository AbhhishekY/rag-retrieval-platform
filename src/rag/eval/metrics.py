"""Retrieval metrics: precision@k, recall@k, NDCG@k (binary relevance).

All expect:
  retrieved: list[str]  -- doc_ids in ranked order
  relevant:  set[str]   -- ground-truth relevant doc_ids
"""
from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for d in topk if d in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for d in topk if d in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k == 0:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(retrieved[:k]):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
