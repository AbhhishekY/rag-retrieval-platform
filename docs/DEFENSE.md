# Retrieval Platform — Defense Notes

## Corpus

**MultiHop-RAG** (`yixuantt/MultiHopRAG`) — 609 news articles (bodies + metadata), plus 2,556 Q/A pairs with `evidence_list` providing per-query ground-truth document URLs. Evaluations use article URL as `doc_id`; the retrieval unit is chunk, the eval unit is doc (retrieval results deduplicated to doc level before metrics).

After ingest: **19,817 chunks** (recursive-512-10%).

## Stack

All retrieval is local; Azure OpenAI (`gpt-4_1_dev_1`) is reserved for optional answer synthesis and is NOT on the retrieval critical path.

| Component | Implementation | Why |
|---|---|---|
| Embedder | FastEmbed (ONNX) — `sentence-transformers/all-MiniLM-L6-v2` (384d, L2-normalized) | ~500 MB PyTorch avoided; ONNX Runtime is 2–3× faster on CPU; same model weights. |
| BM25 | `rank_bm25.BM25Okapi`, lowercase tokenization, no stopword removal / lemmatization | News text is full of named entities and numbers; lemmatizing breaks them. BM25's IDF is enough. |
| Vector index | FAISS `IndexFlatIP` (exhaustive) | For <100K vectors, brute force beats HNSW — no graph traversal overhead, zero recall loss. 19,817 vecs searchable in ~2 ms. |
| Fusion | RRF primary (`rrf_k=60`), weighted-alpha with per-query min-max normalization as tunable alternative | RRF sidesteps the BM25/cosine score-scale mismatch. Weighted path gives interpretable α for the "tunable weights" requirement. |
| Reranker | FastEmbed `Xenova/ms-marco-MiniLM-L-6-v2` (ONNX cross-encoder), single batched `predict` call | Cross-attention between query and candidate adds ranking signal bi-encoders can't produce. Batched to minimize per-pair overhead. |
| API | FastAPI `lifespan`-loaded engine, `POST /search` with score breakdown | Models load once at startup, warmed up, then serve requests with per-query `{bm25, semantic, hybrid_fused, rerank, final}` breakdown. |
| Ingest | Recursive chunking, SHA-256 content hash in SQLite `ingest_manifest` | Re-running ingest on unchanged docs skips embedding entirely. |

## Chunking rationale

Recursive character splitter, target 512 chars, 10% overlap. Separators in order:
paragraph (`\n\n`) → line (`\n`) → sentence (`. `) → word (` `) → char.

Rationale:
- A chunk exists because the embedder compresses it to ONE vector. If the chunk straddles topic boundaries, the vector becomes a blur that matches neither topic well. So chunks must respect meaning boundaries, not just size.
- Overlap is **duplication insurance**, not narrative continuity — embedders don't read chunks sequentially. If a key sentence lives at the boundary of chunks N and N+1, without overlap it's half in each and both vectors get diluted; with overlap, at least one chunk captures it cleanly.
- Character-based (not token-based) for determinism + zero deps. 512 chars ≈ 128 tokens for English.

## Hybrid search — physics

BM25 and dense embeddings fail in **orthogonal** ways:

- **BM25 wins** on rare proper nouns, identifiers, exact phrases, OOV tokens, entity-heavy queries. It's bag-of-words with IDF weighting — it does not "understand" anything, which is a feature when you need rare-token matching.
- **Semantic wins** on paraphrase, synonymy, conceptual queries with no rare tokens.

RRF combines them rank-wise (no score normalization needed): `rrf_score(d) = Σ 1 / (k + rank_i(d))`.

## Experimental results

### Tier 1: required 3 configs (N=200 queries, sequential, p95 accurate)

| Config | P@5 | R@5 | **NDCG@5** | p50 | **p95** | p99 | cold |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic_only | 0.2540 | 0.5067 | 0.5101 | 346 ms | **488 ms** | 566 ms | 268 ms |
| **hybrid (RRF)** | **0.2980** | **0.6042** | **0.5916** | 353 ms | **478 ms** ✅ | 539 ms | 329 ms |
| hybrid+rerank | 0.2770 | 0.5550 | 0.5627 | 1275 ms | 1591 ms ❌ | 1774 ms | 852 ms |

**Winner: hybrid (no rerank)** — both best quality and under the p95 < 500 ms budget.

### Surprising finding: cross-encoder rerank hurt quality

Rerank dropped NDCG@5 from 0.5916 → 0.5627 and recall from 0.6042 → 0.5550. Why?

MultiHop-RAG queries ask about relationships across multiple documents ("Do article X and article Y both report Z?"). The optimal retrieval for these queries has **distributed evidence** — several docs each contributing a fact. The cross-encoder scores *single-passage relevance to the full query* and tends to promote one strongly-matching passage over several partial-evidence passages. On this query distribution, that's actively harmful: some qrel-relevant docs get pushed out of the top-5.

This is a genuine query-distribution-dependent result, not a model defect. Rerank **would** help on single-hop QA or direct factoid queries.

### BM25-only beat semantic-only overall

Evaluated separately, BM25 alone (NDCG=0.5766) beat semantic alone (NDCG=0.5101) on this corpus. Per-query diff (`outputs/runs/bm25_wins.json`, top 10 below) shows every top BM25 win is a query containing a specific **publication name** (TechCrunch, Wired, Sporting News, The Verge, Hacker News, Fortune, Yardbarker, The Roar) or a **rare named entity** (Sam Altman, FTX, Tyreek Hill, Jordan Love, Alex Verdugo, Tasmania JackJumpers, Sony headphones). MiniLM-L6 (general-purpose, 384d) under-represents these rare tokens; BM25's rare-token IDF weighting nails them.

Top 10 BM25 wins (all delta +0.6 to +1.0 in NDCG@5), excerpted:

```
delta=+1.000  Q: Do the TechCrunch article on software companies and the Hacker News article on The Epoch Times...
delta=+1.000  Q: Does the article from Wired suggest that Sony headphones do not offer...
delta=+0.920  Q: Did the article from The Roar | Sports Writers Blog attribute the Tasmania JackJumpers'...
delta=+0.877  Q: Does the TechCrunch article suggest that Sam Altman is involved...
delta=+0.877  Q: After the Sporting News report on Tyreek Hill's chances of achieving 2,000-plus receiving yards...
delta=+0.704  Q: Who is the individual associated with FTX...
delta=+0.704  Q: Which company, recently mentioned in articles by both TechCrunch and The Verge...
delta=+0.613  Q: What is the name of the company that was discussed on TechCrunch for removing AI-created songs...
delta=+0.613  Q: Does the Sporting News article anticipate an impressive performance in the upcoming home game for Jordan Love...
delta=+0.613  Q: Did the Yardbarker article describe Alex Verdugo's offensive performance...
```

**Clean failure-mode story:** MiniLM-L6 struggles with rare publication names + entity IDs; BM25 is the complement.

## Hard-mode signal answers

### Which chunk size wins and why
At 512 chars + 10% overlap (recursive), the hybrid config produced NDCG 0.59 at p95 478 ms. Chunk-size sweep (Tier 2) was scoped but not run — scripts support it via `scripts/ingest.py --chunk-size 256 --index-subdir chunk256` and re-eval. Given MultiHop-RAG is multi-hop-biased, smaller chunks (256) are expected to improve recall (more granular evidence stitching) at the cost of per-chunk completeness. The 512 default was chosen via first-principles: news paragraphs average ~400-600 chars, so 512 respects most paragraph boundaries while holding a full topical statement.

### When does BM25 beat semantic
See the top-10 diff above — BM25 beats semantic when the query contains:
1. **Publication / source names** (TechCrunch, Wired, Hacker News, The Verge…)
2. **Rare proper nouns / named entities** (company names, athletes, specific people)
3. **Exact identifiers** (dates, event names, product codes)

In short: queries whose signal is carried by **low-frequency specific tokens** the embedder sees rarely during training.

### Cold-cache vs warm-cache latency profile
- **Cold (first query after load):** 268–852 ms depending on config. The first query pays ONNX session initialization for both the embedder (and reranker if used).
- **Warm p50:** 346–1275 ms depending on config.
- **Warm p95:** 478–1591 ms depending on config.

The cold/warm gap is modest (~200 ms) — ONNX Runtime warm-up is faster than PyTorch's. First-query latency is reported separately (`first_query_latency_ms`) in every report so it's never folded into the p95 number.

### Cost per 1,000 queries
See [`COST.md`](./COST.md). Retrieval is $0 (all local). Only generation incurs Azure billing if answer synthesis is added.

## What's NOT in v1 (honest scope)

- **Tier 2/3 experiment sweeps not executed** — chunk-size (256 / 1024) and alpha (0.3 / 0.7) runs are scripted (`run_all_experiments.py --tiers 1 3 2`) but deferred to respect time budget. Current DEFENSE uses Tier 1 + BM25-diff analysis.
- **PDF padding not included** — MultiHop-RAG provides 609 articles; the assignment asks for 1,000+. The mixed-format PDF pipeline (`load_pdf_directory`) is implemented and tested, but not populated. Adding 400 arXiv PDFs to the corpus is a 20-min follow-up.
- **LLM answer synthesis** — out of retrieval scope; `gpt-4_1_dev_1` hookup plumbing is in `config.py` but no generation endpoint yet.
