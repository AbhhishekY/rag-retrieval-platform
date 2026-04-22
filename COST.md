# Cost per 1,000 Queries

## Retrieval path (all-local, no external API)

| Component | Per-query cost | 1K queries |
|---|---|---|
| FastEmbed query embed (MiniLM on CPU via ONNX) | ~8–15 ms compute | **$0** |
| BM25 (in-process over ~20K chunks) | ~250–350 ms compute | **$0** |
| FAISS IndexFlatIP (in-process) | <2 ms | **$0** |
| RRF fusion | <1 ms | **$0** |
| Cross-encoder rerank top-20 (optional, CPU) | ~900 ms compute | **$0** |

**Total external spend per 1,000 queries: $0.**

Every component — embedder, BM25, vector index, cross-encoder reranker — runs in-process via CPU. No network calls, no API keys, no rate limits.

## Hardware amortization

The only real cost is the machine running the service. On a commodity c6i.xlarge (~$0.17/hr ≈ $124/mo), sustained at 1 QPS that's **~$0.004 per 1,000 queries**. A rounding error.

Run local on a developer laptop and it's free (ignoring electricity).

## Ingest cost (one-time, per corpus)

All-local — no external calls during ingestion either. Wall-time on MultiHop-RAG (609 docs → 19,817 chunks) was well under one minute of embedding + indexing on CPU. **$0** external.

Incremental ingest (`ingest_manifest.db` SHA-256 tracking) means re-running on unchanged docs is O(1) — no re-embedding. Adding 100 new docs to a 609-doc corpus re-embeds only those 100.

## Summary

- **Retrieval:** $0 per 1K queries.
- **Ingest:** $0 external.
- **Hardware:** ~$0.004 per 1K queries amortized on a $125/mo VM (negligible).
- **No vendor lock-in:** the stack is pip-installable and fully open.
