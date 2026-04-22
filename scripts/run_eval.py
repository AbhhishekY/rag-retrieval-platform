"""Run one eval config end-to-end."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rag.config import get_settings
from rag.eval.harness import run_eval
from rag.eval.qrels import load_multihop_eval
from rag.eval.reports import save_report
from rag.retrieve.embedder import Embedder
from rag.retrieve.reranker import Reranker
from rag.retrieve.search import SearchEngine


def build_search_kwargs(config: str, settings) -> dict:
    base = {
        "top_k_retrieve": settings.top_k_retrieve,
        "top_k_rerank": settings.top_k_rerank,
        "top_k_final": settings.top_k_final,
        "rrf_k": settings.rrf_k,
    }
    if config == "semantic_only":
        return {**base, "fusion_method": "semantic_only", "use_rerank": False}
    if config == "hybrid":
        return {**base, "fusion_method": "rrf", "use_rerank": False}
    if config == "hybrid+rerank":
        return {**base, "fusion_method": "rrf", "use_rerank": True}
    if config == "bm25_only":
        return {**base, "fusion_method": "bm25_only", "use_rerank": False}
    raise ValueError(f"Unknown config: {config}")


async def main_async(args) -> int:
    settings = get_settings()
    index_dir = settings.index_dir / args.index_subdir
    if not (index_dir / "faiss").exists():
        print(f"No index at {index_dir}. Run scripts/ingest.py first.", file=sys.stderr)
        return 1

    print("Loading models and indices...")
    embedder = Embedder(settings.embedding_model)
    reranker = Reranker(settings.reranker_model) if args.config.endswith("rerank") else None
    engine = SearchEngine(index_dir, embedder, reranker)

    queries = load_multihop_eval(settings.data_dir / "multihop_rag_queries")
    if args.limit:
        queries = queries[: args.limit]
    print(f"Evaluating {len(queries)} queries on config={args.config}")

    search_kwargs = build_search_kwargs(args.config, settings)
    report = await run_eval(
        engine, queries, config_name=args.config,
        search_kwargs=search_kwargs, concurrency=args.concurrency,
    )

    save_report(report, settings.output_dir / "runs")
    print(
        f"P@5={report.precision_at_5_mean:.4f} "
        f"R@5={report.recall_at_5_mean:.4f} "
        f"NDCG@5={report.ndcg_at_5_mean:.4f} "
        f"p50={report.latency_p50_ms:.0f}ms "
        f"p95={report.latency_p95_ms:.0f}ms "
        f"p99={report.latency_p99_ms:.0f}ms "
        f"cold={report.first_query_latency_ms:.0f}ms"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True,
        choices=["semantic_only", "hybrid", "hybrid+rerank", "bm25_only"],
    )
    parser.add_argument("--index-subdir", default="default")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="1 = sequential (accurate p95); >1 = throughput mode")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
