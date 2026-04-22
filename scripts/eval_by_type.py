"""
Stratified eval: break down P@5 / R@5 / NDCG@5 by question_type and n_relevant.

Best config: weighted alpha=0.7, max_chunks_per_doc=1, top_k_retrieve=100,
             top_k_rerank=50, top_k_final=5  (no cross-encoder rerank).
"""
from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict

from rag.config import get_settings
from rag.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k
from rag.eval.qrels import load_multihop_eval
from rag.retrieve.embedder import Embedder
from rag.retrieve.search import SearchEngine

N_QUERIES = 200          # keep runtime under ~3 min
ALPHA     = 0.7
TOP_K_RETRIEVE = 100
TOP_K_RERANK   = 50
TOP_K_FINAL    = 5

SEARCH_KWARGS = dict(
    fusion_method="weighted",
    alpha=ALPHA,
    use_rerank=False,
    top_k_retrieve=TOP_K_RETRIEVE,
    top_k_rerank=TOP_K_RERANK,
    top_k_final=TOP_K_FINAL,
    max_chunks_per_doc=1,
)


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _print_group_table(
    title: str,
    groups: dict[str, list[tuple[float, float, float]]],
) -> None:
    # groups: key -> list of (p5, r5, ndcg5) tuples
    print(f"\n  {'─'*62}")
    print(f"  {title}")
    print(f"  {'─'*62}")
    header = f"  {'Group':<22}{'N':>5}{'P@5':>10}{'R@5':>10}{'NDCG@5':>10}"
    print(header)
    print(f"  {'─'*62}")
    for key in sorted(groups):
        rows = groups[key]
        n = len(rows)
        p5   = statistics.mean(r[0] for r in rows)
        r5   = statistics.mean(r[1] for r in rows)
        ndcg = statistics.mean(r[2] for r in rows)
        print(f"  {key:<22}{n:>5}  {_fmt(p5):>9}  {_fmt(r5):>9}  {_fmt(ndcg):>9}")
    # overall
    all_rows = [r for rows in groups.values() for r in rows]
    n = len(all_rows)
    p5   = statistics.mean(r[0] for r in all_rows)
    r5   = statistics.mean(r[1] for r in all_rows)
    ndcg = statistics.mean(r[2] for r in all_rows)
    print(f"  {'─'*62}")
    print(f"  {'OVERALL':<22}{n:>5}  {_fmt(p5):>9}  {_fmt(r5):>9}  {_fmt(ndcg):>9}")


async def main() -> None:
    settings = get_settings()
    index_dir = settings.index_dir / "default"

    print("Loading engine (weighted alpha=0.7, max_chunks_per_doc=1)…")
    embedder = Embedder(settings.embedding_model)
    engine   = SearchEngine(index_dir, embedder, None)

    queries = load_multihop_eval(settings.data_dir / "multihop_rag_queries")[:N_QUERIES]
    print(f"Evaluating {len(queries)} queries…\n")

    # Accumulators
    by_type:     dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_nrel:     dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_type_nrel: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for i, eq in enumerate(queries, 1):
        hits = await engine.search(eq.query, **SEARCH_KWARGS)

        seen: set[str] = set()
        retrieved: list[str] = []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                retrieved.append(h.doc_id)

        p5   = precision_at_k(retrieved, eq.relevant_doc_ids, k=5)
        r5   = recall_at_k(retrieved, eq.relevant_doc_ids, k=5)
        ndcg = ndcg_at_k(retrieved, eq.relevant_doc_ids, k=5)
        triple = (p5, r5, ndcg)

        qtype = eq.question_type or "unknown"
        nrel  = len(eq.relevant_doc_ids)
        nrel_key = f"n_relevant={nrel}"
        combo_key = f"{qtype} | n_rel={nrel}"

        by_type[qtype].append(triple)
        by_nrel[nrel_key].append(triple)
        by_type_nrel[combo_key].append(triple)

        if i % 50 == 0:
            print(f"  … {i}/{len(queries)} done")

    print("\n" + "═" * 64)
    print("  STRATIFIED EVAL RESULTS")
    print("═" * 64)

    _print_group_table("By question_type", by_type)
    _print_group_table("By n_relevant docs", by_nrel)
    _print_group_table("Cross-tab: question_type × n_relevant", by_type_nrel)

    print("\n  Notes:")
    print("  · P@5 ceiling = n_relevant / 5  (e.g. n_rel=2 → max 0.40)")
    print("  · NDCG@5 is the most meaningful metric here (rank-sensitive)")
    print("  · comparison_query tends to be 2-hop; inference_query often 3-hop")
    print()


if __name__ == "__main__":
    asyncio.run(main())
