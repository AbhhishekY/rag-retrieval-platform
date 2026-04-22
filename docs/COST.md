# Cost per 1,000 Queries

## Retrieval path (all-local, no external API)

| Component | Per-query cost | 1K queries |
|---|---|---|
| FastEmbed query embed (MiniLM on CPU via ONNX) | ~8–15 ms compute | **$0** |
| BM25 (in-process over ~20K chunks) | ~250–350 ms compute | **$0** |
| FAISS IndexFlatIP (in-process) | <2 ms | **$0** |
| RRF fusion | <1 ms | **$0** |
| Cross-encoder rerank top-20 (optional, CPU) | ~900 ms compute | **$0** |

**Retrieval subtotal: $0 external spend per 1,000 queries.**

The only cost is amortized hardware. On a commodity c6i.xlarge (~$0.17/hr = ~$124/mo), sustained at 1 QPS that's ~$0.004 per 1,000 queries. Still a rounding error.

## Optional answer synthesis via Azure OpenAI (`gpt-4_1_dev_1`)

If the system is extended with a RAG answer-synthesis step, per-query cost depends on context size and output length. Illustrative: 500 tokens of retrieved context + 100 tokens query + 200 tokens answer.

Fill in from your actual Azure pricing sheet for the `gpt-4.1` family:

| Component | Tokens / 1K queries | Rate (from Azure) | Cost / 1K |
|---|---:|---:|---:|
| Input | 600,000 | $___/1M | ~$___ |
| Output | 200,000 | $___/1M | ~$___ |
| **Total / 1K queries** | — | — | **~$___** |

As of public pricing for `gpt-4o` (analog of 4.1-class models): ~$2.50/1M input + $10.00/1M output → ~$3.50/1K queries for this context budget. Generation dominates the cost structure if added.

## Ingest (one-time, per corpus)

All-local — no external calls during ingestion either. Wall-time on MultiHop-RAG (609 docs → 19,817 chunks) was well under one minute of embedding + indexing on CPU. **$0** external.

## Summary

- **Status-quo retrieval:** $0 per 1K queries, all in-process.
- **Add generation:** ~$3.50 per 1K queries (assuming 4o-class rates, typical context).
- **Hardware:** rounding error ($0.004/1K at 1 QPS sustained on a small VM).
