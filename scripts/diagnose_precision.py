"""
Diagnose why P@5 is low.

Three hypotheses:
  H1: Multiple chunks from the same doc cluster in top-5,
      leaving fewer slots for the 2nd/3rd relevant doc.
  H2: MultiHop queries genuinely need 2-3 relevant docs —
      P@5 denominator (5) is just too small.
  H3: The 2nd relevant doc is retrievable but sits at rank 6-15,
      meaning a simple top_k increase would recover it.
"""
from __future__ import annotations

import asyncio
import statistics
from collections import Counter
from pathlib import Path

from rag.config import get_settings
from rag.eval.qrels import load_multihop_eval
from rag.retrieve.embedder import Embedder
from rag.retrieve.search import SearchEngine

N_QUERIES = 200
TOP_K_RETRIEVE = 100   # wide funnel
TOP_K_FINAL    = 20    # look deeper than 5


async def main() -> None:
    settings = get_settings()
    index_dir = settings.index_dir / "default"

    print("Loading engine...")
    embedder = Embedder(settings.embedding_model)
    engine   = SearchEngine(index_dir, embedder, None)

    queries = load_multihop_eval(settings.data_dir / "multihop_rag_queries")[:N_QUERIES]

    # Per-query stats
    n_relevant_per_query:   list[int]   = []
    unique_docs_in_top5:    list[int]   = []
    relevant_found_at_rank: list[int]   = []   # rank where each relevant doc first appears
    first_miss_rank:        list[int]   = []   # earliest rank where a relevant doc is NOT in top-5

    found_at_5  = 0   # queries where ALL relevant docs are in top-5 unique doc_ids
    found_at_10 = 0
    found_at_20 = 0

    for eq in queries:
        hits = await engine.search(
            eq.query,
            top_k_retrieve=TOP_K_RETRIEVE,
            top_k_rerank=TOP_K_FINAL,
            top_k_final=TOP_K_FINAL,
            fusion_method="weighted",
            alpha=0.7,
            use_rerank=False,
        )

        # Build ranked unique doc list (up to top-20)
        seen: set[str] = set()
        ranked_docs: list[str] = []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked_docs.append(h.doc_id)

        n_rel = len(eq.relevant_doc_ids)
        n_relevant_per_query.append(n_rel)

        # H1: how many unique docs in top-5 chunks?
        seen5: set[str] = set()
        for h in hits[:5]:
            seen5.add(h.doc_id)
        unique_docs_in_top5.append(len(seen5))

        # H2/H3: at what rank does each relevant doc appear?
        for rel_id in eq.relevant_doc_ids:
            if rel_id in ranked_docs:
                r = ranked_docs.index(rel_id) + 1   # 1-indexed
                relevant_found_at_rank.append(r)
            else:
                relevant_found_at_rank.append(999)  # not found in top-20

        # Coverage at different k
        top5_ids  = set(ranked_docs[:5])
        top10_ids = set(ranked_docs[:10])
        top20_ids = set(ranked_docs[:20])
        if eq.relevant_doc_ids <= top5_ids:  found_at_5  += 1
        if eq.relevant_doc_ids <= top10_ids: found_at_10 += 1
        if eq.relevant_doc_ids <= top20_ids: found_at_20 += 1

    N = len(queries)
    print("\n" + "═" * 60)
    print("  PRECISION DIAGNOSTIC")
    print("═" * 60)

    print(f"\n── H2: MultiHop relevant-set size ──")
    print(f"  Avg relevant docs / query : {statistics.mean(n_relevant_per_query):.2f}")
    print(f"  Distribution              : {dict(sorted(Counter(n_relevant_per_query).items()))}")

    print(f"\n── H1: Chunk clustering in top-5 ──")
    print(f"  Avg unique doc_ids in top-5 chunks: {statistics.mean(unique_docs_in_top5):.2f}")
    print(f"  (if <5, multiple chunks from same doc are clustering)")

    print(f"\n── H3: Where do relevant docs actually land? ──")
    found   = [r for r in relevant_found_at_rank if r < 999]
    missing = [r for r in relevant_found_at_rank if r == 999]
    if found:
        print(f"  Found in top-20 : {len(found)}/{len(relevant_found_at_rank)}  ({100*len(found)/len(relevant_found_at_rank):.1f}%)")
        print(f"  Missing from top-20: {len(missing)}")
        print(f"  Avg rank of found relevant docs : {statistics.mean(found):.1f}")
        rank_dist = Counter(min(r, 10) for r in found)
        print(f"  Rank distribution (1-10, 10+=bucketed):")
        for rank in sorted(rank_dist):
            bar = "█" * rank_dist[rank]
            label = f"{rank}+" if rank == 10 else str(rank)
            print(f"    rank {label:>3}: {rank_dist[rank]:>4} relevant docs  {bar}")

    print(f"\n── Coverage: all relevant docs found by k ──")
    print(f"  All relevant in top-5  : {found_at_5}/{N}  ({100*found_at_5/N:.1f}%)")
    print(f"  All relevant in top-10 : {found_at_10}/{N}  ({100*found_at_10/N:.1f}%)")
    print(f"  All relevant in top-20 : {found_at_20}/{N}  ({100*found_at_20/N:.1f}%)")

    print(f"\n── Implication for P@5 ──")
    avg_unique_in_5 = statistics.mean(unique_docs_in_top5)
    avg_rel         = statistics.mean(n_relevant_per_query)
    theoretical_max_p5 = min(avg_rel, avg_unique_in_5) / 5
    print(f"  Avg unique docs in top-5 chunks : {avg_unique_in_5:.2f}")
    print(f"  Avg relevant docs per query     : {avg_rel:.2f}")
    print(f"  Theoretical max P@5 (if perfect recall in top-5): {theoretical_max_p5:.4f}")
    print(f"  → Perfect retrieval can still only achieve P@5 ≈ {theoretical_max_p5:.2f}")
    print(f"    because relevant set ≈ {avg_rel:.1f} docs and denominator is always 5")
    print("\n" + "═" * 60)


if __name__ == "__main__":
    asyncio.run(main())
