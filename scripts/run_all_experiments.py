"""Run the experiment matrix and emit SUMMARY.md.

Tier 1 (required): semantic_only / hybrid / hybrid+rerank @ recursive-512-10%
Tier 2 (optional): chunk-size sweep (needs re-ingest — skipped by default)
Tier 3 (optional): alpha sweep via weighted fusion
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rag.config import get_settings
from rag.eval.harness import run_eval
from rag.eval.qrels import load_multihop_eval
from rag.eval.reports import combine_reports_table, save_report
from rag.retrieve.embedder import Embedder
from rag.retrieve.reranker import Reranker
from rag.retrieve.search import SearchEngine


async def run_tier1(embedder, reranker, queries, settings, reports, limit: int | None):
    index_dir = settings.index_dir / "default"
    q = queries[:limit] if limit else queries

    for config_name, kwargs, needs_reranker in [
        ("tier1_semantic_only", {"fusion_method": "semantic_only", "use_rerank": False}, False),
        ("tier1_hybrid", {"fusion_method": "rrf", "use_rerank": False}, False),
        ("tier1_hybrid_rerank", {"fusion_method": "rrf", "use_rerank": True}, True),
        ("tier1_hybrid_metadata_filter", {
            "fusion_method": "weighted", "alpha": settings.hybrid_alpha,
            "use_rerank": False, "use_metadata_filter": True,
        }, False),
    ]:
        engine = SearchEngine(
            index_dir, embedder, reranker if needs_reranker else None
        )
        base = {
            "top_k_retrieve": 100,
            "top_k_rerank": settings.top_k_rerank,
            "top_k_final": 5,
            "rrf_k": 60,
            **kwargs,
        }
        print(f"\n=== {config_name} (N={len(q)}) ===")
        report = await run_eval(engine, q, config_name, base, concurrency=1)
        save_report(report, settings.output_dir / "runs")
        reports.append(report)
        print(
            f"    P@5={report.precision_at_5_mean:.4f} "
            f"R@5={report.recall_at_5_mean:.4f} "
            f"NDCG@5={report.ndcg_at_5_mean:.4f} "
            f"p50={report.latency_p50_ms:.0f}ms "
            f"p95={report.latency_p95_ms:.0f}ms"
        )


async def run_tier3_alpha_sweep(embedder, queries, settings, reports, subset: int):
    """Alpha sweep on default index — no re-ingest needed, no rerank for speed."""
    engine = SearchEngine(settings.index_dir / "default", embedder, None)
    q = queries[:subset]
    for alpha in (0.3, 0.7):
        name = f"tier3_alpha_{alpha}"
        kwargs = {
            "top_k_retrieve": 100, "top_k_rerank": 20, "top_k_final": 5,
            "fusion_method": "weighted", "alpha": alpha, "use_rerank": False,
        }
        print(f"\n=== {name} (N={len(q)}) ===")
        report = await run_eval(engine, q, name, kwargs, concurrency=1)
        save_report(report, settings.output_dir / "runs")
        reports.append(report)
        print(
            f"    P@5={report.precision_at_5_mean:.4f} "
            f"R@5={report.recall_at_5_mean:.4f} "
            f"NDCG@5={report.ndcg_at_5_mean:.4f} "
            f"p50={report.latency_p50_ms:.0f}ms "
            f"p95={report.latency_p95_ms:.0f}ms"
        )


async def main_async(args):
    settings = get_settings()
    print("Loading models...")
    embedder = Embedder(settings.embedding_model)
    reranker = Reranker(settings.reranker_model) if 1 in args.tiers or 2 in args.tiers else None

    queries = load_multihop_eval(settings.data_dir / "multihop_rag_queries")
    print(f"{len(queries)} eval queries available")

    reports = []
    if 1 in args.tiers:
        await run_tier1(embedder, reranker, queries, settings, reports, limit=args.limit)
    if 3 in args.tiers:
        await run_tier3_alpha_sweep(embedder, queries, settings, reports, subset=args.limit or 200)

    table = combine_reports_table(reports)
    summary_path = settings.output_dir / "runs" / "SUMMARY.md"
    summary_path.write_text(
        "# Experiment Matrix Summary\n\n"
        f"Corpus: MultiHop-RAG 609 articles + AG News 400 = 1,009 docs, 19,817+ chunks (recursive-512-10%).\n"
        f"Evaluated on {args.limit or len(queries)} queries per config.\n\n"
        + table
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSUMMARY written to {summary_path}")
    print("\n" + table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="+", type=int, default=[1],
                        help="Which tiers to run (default: 1)")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max queries per config (default 200 for speed)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
