"""Find queries where BM25-only beats semantic-only by the largest NDCG@5 margin.

Answers the hard-mode signal: "When does BM25 beat semantic? Show the failure modes."
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rag.config import get_settings
from rag.eval.harness import run_eval
from rag.eval.qrels import load_multihop_eval
from rag.retrieve.embedder import Embedder
from rag.retrieve.search import SearchEngine


async def main():
    settings = get_settings()
    queries = load_multihop_eval(settings.data_dir / "multihop_rag_queries")[:200]
    embedder = Embedder(settings.embedding_model)
    engine = SearchEngine(settings.index_dir / "default", embedder, None)

    base = {"top_k_retrieve": 100, "top_k_rerank": 20, "top_k_final": 5, "rrf_k": 60}

    print(f"Running BM25-only on {len(queries)} queries...")
    bm25_rep = await run_eval(
        engine, queries, "bm25_only_analysis",
        {**base, "fusion_method": "bm25_only", "use_rerank": False},
    )

    print("Running semantic-only...")
    sem_rep = await run_eval(
        engine, queries, "semantic_only_analysis",
        {**base, "fusion_method": "semantic_only", "use_rerank": False},
    )

    bm25_by_qid = {r.query_id: r for r in bm25_rep.per_query}
    sem_by_qid = {r.query_id: r for r in sem_rep.per_query}
    diffs = []
    for qid, b in bm25_by_qid.items():
        s = sem_by_qid.get(qid)
        if s is None:
            continue
        diffs.append({
            "query": b.query,
            "bm25_ndcg": b.ndcg_at_5,
            "semantic_ndcg": s.ndcg_at_5,
            "delta": b.ndcg_at_5 - s.ndcg_at_5,
        })

    diffs.sort(key=lambda d: d["delta"], reverse=True)
    top_wins = [d for d in diffs[:10] if d["delta"] > 0]

    out = settings.output_dir / "runs" / "bm25_wins.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(top_wins, indent=2), encoding="utf-8")

    print(f"\nOverall: bm25_NDCG={bm25_rep.ndcg_at_5_mean:.4f} "
          f"semantic_NDCG={sem_rep.ndcg_at_5_mean:.4f}")
    print(f"\nTop {len(top_wins)} queries where BM25 beat semantic (by NDCG@5 delta):\n")
    for d in top_wins:
        print(f"  delta={d['delta']:+.3f}  bm25={d['bm25_ndcg']:.3f} sem={d['semantic_ndcg']:.3f}")
        print(f"    Q: {d['query'][:120]}")


if __name__ == "__main__":
    asyncio.run(main())
