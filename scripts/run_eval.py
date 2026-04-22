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


def build_search_kwargs(config: str, settings, alpha: float | None = None) -> dict:
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
    if config == "weighted":
        a = alpha if alpha is not None else settings.hybrid_alpha
        return {**base, "fusion_method": "weighted", "alpha": a, "use_rerank": False}
    if config == "hybrid+metadata_filter":
        # use_metadata_filter is consumed by the harness (_run_one), not passed to SearchEngine.
        # The harness injects filters={"category": eq.category} per-query from ground truth.
        return {**base, "fusion_method": "weighted", "alpha": settings.hybrid_alpha,
                "use_rerank": False, "use_metadata_filter": True}
    raise ValueError(f"Unknown config: {config}")


async def main_async(args) -> int:
    settings = get_settings()
    index_dir = settings.index_dir / args.index_subdir
    if not (index_dir / "faiss").exists():
        print(f"No index at {index_dir}. Run scripts/ingest.py first.", file=sys.stderr)
        return 1

    print("Loading models and indices...")
    embedder = Embedder(settings.embedding_model)
    reranker = Reranker(settings.reranker_model) if "rerank" in args.config and not args.config.endswith("metadata_filter") else None
    engine = SearchEngine(index_dir, embedder, reranker)

    queries = load_multihop_eval(settings.data_dir / "multihop_rag_queries")
    if args.limit:
        queries = queries[: args.limit]
    print(f"Evaluating {len(queries)} queries on config={args.config}")

    search_kwargs = build_search_kwargs(args.config, settings, alpha=args.alpha)
    if args.max_chunks_per_doc is not None:
        search_kwargs["max_chunks_per_doc"] = args.max_chunks_per_doc
    if args.top_k_final is not None:
        search_kwargs["top_k_final"] = args.top_k_final
    if args.top_k_rerank is not None:
        search_kwargs["top_k_rerank"] = args.top_k_rerank
    if args.top_k_retrieve is not None:
        search_kwargs["top_k_retrieve"] = args.top_k_retrieve
    config_label = f"{args.config}(a={args.alpha})" if args.alpha is not None else args.config
    report = await run_eval(
        engine, queries, config_name=config_label,
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
        choices=["semantic_only", "hybrid", "hybrid+rerank", "bm25_only", "weighted",
                 "hybrid+metadata_filter"],
    )
    parser.add_argument("--alpha", type=float, default=None,
                        help="Alpha for weighted fusion (0.0=pure dense, 1.0=pure BM25)")
    parser.add_argument("--max-chunks-per-doc", type=int, default=None,
                        help="Cap chunks from the same doc in final results (diversity)")
    parser.add_argument("--top-k-final", type=int, default=None,
                        help="Override top_k_final (number of results returned)")
    parser.add_argument("--top-k-rerank", type=int, default=None,
                        help="Override top_k_rerank (fusion pool size)")
    parser.add_argument("--top-k-retrieve", type=int, default=None,
                        help="Override top_k_retrieve (candidates from each retriever)")
    parser.add_argument("--index-subdir", default="default")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="1 = sequential (accurate p95); >1 = throughput mode")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
