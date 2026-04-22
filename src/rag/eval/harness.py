"""Eval harness: run queries through SearchEngine with bounded concurrency,
compute P@k / R@k / NDCG@k, capture per-query latency (cold-first, then warm).
"""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from rag.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k
from rag.eval.qrels import EvalQuery
from rag.retrieve.search import SearchEngine


@dataclass
class PerQueryResult:
    query_id: str
    query: str
    latency_ms: float
    retrieved_doc_ids: list[str]
    relevant_doc_ids: set[str]
    precision_at_5: float
    recall_at_5: float
    ndcg_at_5: float


@dataclass
class EvalReport:
    config_name: str
    n_queries: int
    precision_at_5_mean: float
    recall_at_5_mean: float
    ndcg_at_5_mean: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    first_query_latency_ms: float
    per_query: list[PerQueryResult] = field(default_factory=list)


async def _run_one(
    engine: SearchEngine, eq: EvalQuery, search_kwargs: dict[str, Any]
) -> PerQueryResult:
    kwargs = dict(search_kwargs)
    # use_metadata_filter is a harness-level flag, not a SearchEngine param.
    # When set, we pass the query's ground-truth category as a hard filter.
    if kwargs.pop("use_metadata_filter", False) and eq.category:
        kwargs["filters"] = {"category": eq.category}
    t0 = time.perf_counter()
    hits = await engine.search(eq.query, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    # Dedupe to doc-level (retrieval unit is chunk; eval unit is doc).
    # Preserve rank order — keep first occurrence of each doc_id.
    seen: set[str] = set()
    retrieved: list[str] = []
    for h in hits:
        if h.doc_id not in seen:
            seen.add(h.doc_id)
            retrieved.append(h.doc_id)
    return PerQueryResult(
        query_id=eq.query_id,
        query=eq.query,
        latency_ms=latency_ms,
        retrieved_doc_ids=retrieved,
        relevant_doc_ids=eq.relevant_doc_ids,
        precision_at_5=precision_at_k(retrieved, eq.relevant_doc_ids, k=5),
        recall_at_5=recall_at_k(retrieved, eq.relevant_doc_ids, k=5),
        ndcg_at_5=ndcg_at_k(retrieved, eq.relevant_doc_ids, k=5),
    )


async def run_eval(
    engine: SearchEngine,
    queries: list[EvalQuery],
    config_name: str,
    search_kwargs: dict[str, Any],
    concurrency: int = 8,
) -> EvalReport:
    if not queries:
        raise ValueError("No queries provided")

    # First query in isolation captures cold-start latency
    first = await _run_one(engine, queries[0], search_kwargs)

    # For p95 to be meaningful, we must not let queries queue on each other.
    # concurrency=1 gives true per-request latency.
    # Higher concurrency measures throughput, not latency.
    if concurrency <= 1:
        rest = []
        for eq in queries[1:]:
            rest.append(await _run_one(engine, eq, search_kwargs))
    else:
        sem = asyncio.Semaphore(concurrency)

        async def _guarded(eq: EvalQuery):
            async with sem:
                return await _run_one(engine, eq, search_kwargs)

        rest = list(await asyncio.gather(*(_guarded(q) for q in queries[1:])))

    all_results = [first] + rest

    latencies_warm = [r.latency_ms for r in all_results[1:]] or [first.latency_ms]

    return EvalReport(
        config_name=config_name,
        n_queries=len(all_results),
        precision_at_5_mean=statistics.mean(r.precision_at_5 for r in all_results),
        recall_at_5_mean=statistics.mean(r.recall_at_5 for r in all_results),
        ndcg_at_5_mean=statistics.mean(r.ndcg_at_5 for r in all_results),
        latency_p50_ms=_percentile(latencies_warm, 50),
        latency_p95_ms=_percentile(latencies_warm, 95),
        latency_p99_ms=_percentile(latencies_warm, 99),
        first_query_latency_ms=first.latency_ms,
        per_query=all_results,
    )


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
