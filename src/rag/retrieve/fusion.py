"""Score fusion strategies for hybrid retrieval.

- rrf_fuse: rank-based Reciprocal Rank Fusion; no score-scale assumptions.
- weighted_alpha_fuse: per-query min-max normalize both lists, then
  alpha*bm25 + (1-alpha)*dense. Missing IDs treated as zero after normalization.
"""
from __future__ import annotations


def rrf_fuse(
    list_a: list[tuple[str, float]],
    list_b: list[tuple[str, float]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for results in (list_a, list_b):
        for rank, (cid, _) in enumerate(results):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ordered[:top_k]


def weighted_alpha_fuse(
    bm25_list: list[tuple[str, float]],
    dense_list: list[tuple[str, float]],
    alpha: float = 0.5,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    bm25_norm = _minmax_normalize(bm25_list)
    dense_norm = _minmax_normalize(dense_list)
    all_ids = set(bm25_norm) | set(dense_norm)
    fused = {
        cid: alpha * bm25_norm.get(cid, 0.0) + (1.0 - alpha) * dense_norm.get(cid, 0.0)
        for cid in all_ids
    }
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ordered[:top_k]


def _minmax_normalize(scored: list[tuple[str, float]]) -> dict[str, float]:
    if not scored:
        return {}
    vals = [s for _, s in scored]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {cid: 1.0 for cid, _ in scored}
    return {cid: (s - lo) / (hi - lo) for cid, s in scored}
