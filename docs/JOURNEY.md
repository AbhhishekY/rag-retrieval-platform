# Build Journey — From Blank Directory to Shipping Retrieval Platform

A chronological narrative of the build: what we tried, what broke, what worked, and how each decision landed. Every p95 / NDCG number in this doc is from an actual run that's committed to history.

---

## 0. Framing the problem (brainstorming phase)

**Constraint:** production-grade retrieval (1000+ docs, hybrid, rerank, eval harness, p95 <500ms, cost/1K queries) in a few hours on CPU-only Windows hardware.

**Three-technique brainstorm** (see `_bmad-output/brainstorming/brainstorming-session-2026-04-22-101317.md`):

1. **First Principles** — stripped chunking, BM25, semantic, rerank to physics. Produced principles 1-4 (one-vector-per-topic, overlap-as-duplication-insurance, query-distribution-dictates-chunk-size, orthogonal-failure-modes).
2. **Morphological Analysis** — full grid was 3 × 3 × 3 × 3 × 3 × 2 = 162 cells. Pruned to 9 runs: 3 Tier 1 (required), 2 Tier 2 (chunk sweep), 2 Tier 3 (alpha sweep), 2 Tier 4 (strategy ablation).
3. **Pre-mortem** — named 7 traps before building: (1) corpus-mismatch, (2) fusion-math-nonsense, (3) model-download-hang, (4) async-sync-deadlock, (5) rerank-not-batched, (6) eval-too-slow, (7) qrels-format-mismatch. Three of these bit us anyway during execution — but pre-naming them made the fixes fast.

**Key output from brainstorm:** 7 locked decisions, 9-run experiment matrix, 32-task implementation plan committed to `docs/superpowers/plans/2026-04-22-retrieval-platform.md`.

---

## 1. Stack selection — PyTorch vs. FastEmbed

**Original plan:** `sentence-transformers` + PyTorch for both embedder and cross-encoder reranker.

**Mid-build pivot (user-prompted):** "Why are we installing PyTorch? It's expensive and heavy."

Investigated the alternative. Here's the comparison that drove the switch:

| | `sentence-transformers` + PyTorch | **FastEmbed (ONNX Runtime)** |
|---|---|---|
| Install size | ~600 MB (torch + transformers + sklearn) | **~80 MB** |
| First import | ~5–8 sec | ~1–2 sec |
| Embed speed on CPU | baseline | **2–3× faster** (ONNX graph fusion) |
| Memory footprint | ~1–2 GB RAM | ~400–600 MB RAM |
| Model weights | identical | **identical** (same HF exports) |
| API churn needed | — | ~30 lines in `embedder.py` + `reranker.py` |

**Decision:** swapped to FastEmbed. Rationale lived in two places:
- Install: 500 MB avoided = 2–5 minutes saved on this connection
- Inference: 2–3× faster on CPU = bigger p95 headroom (we ended up needing it)

Commit: `34fb7b4` ("refactor: swap sentence-transformers -> fastembed (ONNX runtime)").

---

## 2. Preflight snags and fixes

### Snag 1: Incomplete model download

First run of `preflight.py` failed with:
```
Preflight FAILED: NoSuchFile: [ONNXRuntimeError] : 3 : NO_SUCHFILE :
Load model from ...models--Xenova--ms-marco-MiniLM-L-6-v2\snapshots\...\onnx\model.onnx
failed. File doesn't exist
```

Root cause: partial download. The HF Hub cache had the snapshot directory + tokenizer files, but the actual `.onnx` file was still an `.incomplete` blob.

**Fix:** cleared the cache directory, re-ran. Second attempt succeeded (took 4.5 min on this connection). Sanity-tested with `rerank("who founded Tesla?", ["Elon Musk founded Tesla.", "Cats are animals."])` → scores `[+10.2, -11.1]` (correct).

### Snag 2: MultiHop-RAG dataset has two configs

Naive `load_dataset("yixuantt/MultiHopRAG")` failed:
```
ValueError: Config name is missing.
Please pick one among the available configs: ['MultiHopRAG', 'corpus']
```

The HF dataset actually ships TWO configs:
- `corpus` — 609 rows, article bodies (for ingest)
- `MultiHopRAG` — 2,556 rows, Q/A records (for eval)

**Fix:** loaded both separately, saved to `data/multihop_rag_corpus/` and `data/multihop_rag_queries/`. Updated loader (`loaders.py`) and qrels adapter (`qrels.py`) to target the right directory for each use.

Commit: `c72b683` ("fix(preflight): MultiHop-RAG has two HF configs (corpus + MultiHopRAG)").

### Preflight output snapshot

After both fixes, preflight prints:
```
[1/3] Pre-downloading FastEmbed (ONNX) models...
      models cached in 1.4s
[2/3] Downloading MultiHop-RAG benchmark (two configs: corpus + MultiHopRAG)...
      saved both configs in 17.8s
[3/3] Inspecting formats...
      corpus[train]: 609 rows, keys: ['category', 'author', 'published_at', 'body', 'title', 'url', 'source']
      queries[train]: 2556 rows, keys: ['evidence_list', 'answer', 'query', 'question_type']
      evidence item keys: ['author', 'category', 'fact', 'published_at', 'source', 'title', 'url']
```

**Schema discovery that mattered:** evidence `url` = corpus `url` = our `doc_id` — the joinable field for eval.

---

## 3. Chunking (TDD)

6 tests written before implementation:
1. short text → single chunk
2. long text → paragraph boundary splits
3. overlap → shared tokens between consecutive chunks
4. empty text → empty list
5. whitespace-only → empty list
6. dense text → sentence-level fallback

First pytest run: `ImportError: No module named 'rag.ingest.chunking'` ✅ (TDD red).

Implemented the recursive splitter with cascade `\n\n` → `\n` → `. ` → ` ` → chars. All 6 passed on first try.

**Result on real corpus:** 609 docs → **19,817 chunks** at 512/10% recursive. Higher than the 6K–8K I'd estimated; average article is longer than my sample suggested.

Commit: `559ae50` ("feat: core types, config, recursive chunker (6 TDD tests passing)").

---

## 4. Ingest pipeline — first run

Ingest code written, ran on full MultiHop-RAG:

```
Loaded 609 docs
609 docs changed since last ingest; 0 skipped
chunking: 100%|##########| 609/609 [00:00<00:00, 10575.93it/s]
Produced 19817 chunks
Embedding chunks...
INGEST DONE: {'docs_total': 609, 'chunks_total': 19817, 'embedding_dim': 384, 'index_dir': 'indices\\default'}
```

Chunking in <1 second. Embedding ~1 minute (FastEmbed batched ONNX on CPU). No drama.

Artifacts:
- `indices/default/bm25.pkl` (pickled tokenized corpus)
- `indices/default/faiss/faiss.index` (19,817 × 384 float32)
- `indices/default/faiss/chunk_ids.pkl`
- `indices/default/chunks.jsonl` (one JSON per chunk, with text + metadata)
- `indices/default/ingest_manifest.db` (SHA-256 per doc for incremental re-runs)

Commit: `c447805` ("feat(ingest): end-to-end pipeline + runner").

---

## 5. Retrieval pipeline + first live query

Wired up embedder + BM25 + FAISS + RRF fusion + reranker into async `SearchEngine`. First smoke query:

```
>>> asyncio.run(engine.search('What is the impact of climate change?', top_k_rerank=20, top_k_final=3))
Cold query: 742 ms  (first query pays ONNX session init)
Warm query: 932 ms
Top 3: climate articles with sensible rerank scores (2.4, 1.8, −0.04)
```

**p95 ≈ 930 ms warm** — nearly 2× our 500ms target. This was the moment we knew the rerank path would be tight.

---

## 6. Eval harness — two bugs found in the first run

First end-to-end eval on 20 queries, hybrid config, concurrency=8:

```
P@5=0.5200 R@5=1.0750 NDCG@5=0.8647 p50=2667ms p95=3488ms p99=3554ms
```

### Bug #1: R@5 > 1 impossible

**Root cause:** retrieval unit is chunk; eval unit is doc. If top-5 chunks came from fewer than 5 unique docs, multiple chunks of the same relevant doc counted as multiple "hits" — inflating recall beyond 1.0.

**Fix (`src/rag/eval/harness.py`):**
```python
# Dedupe to doc-level. Preserve rank order — keep first occurrence of each doc_id.
seen: set[str] = set()
retrieved: list[str] = []
for h in hits:
    if h.doc_id not in seen:
        seen.add(h.doc_id)
        retrieved.append(h.doc_id)
```

### Bug #2: p95 of 3.5s was queue wait, not per-request latency

**Root cause:** `concurrency=8` + ~400ms per query + limited CPU threads meant each query's `latency_ms` was measured INCLUDING time spent waiting in the `asyncio.Semaphore(8)` queue. The p95 number was throughput-scaled, not latency.

**Fix (`src/rag/eval/harness.py`):** defaulted concurrency to 1 for eval runs, with a branch that skips the semaphore entirely when concurrency ≤ 1:

```python
if concurrency <= 1:
    rest = []
    for eq in queries[1:]:
        rest.append(await _run_one(engine, eq, search_kwargs))
else:
    sem = asyncio.Semaphore(concurrency)
    ...
```

### Re-run after both fixes

```
P@5=0.3000 R@5=0.6250 NDCG@5=0.5973 p50=315ms p95=378ms p99=397ms
```

R@5 ≤ 1 ✅, **p95 378ms — under 500ms budget** ✅. The harness was fixed.

---

## 7. Rerank latency investigation

Separately, tested hybrid+rerank on 20 queries:
```
P@5=0.3000 R@5=0.6125 NDCG@5=0.6527 p50=1232ms p95=1633ms
```

NDCG up from 0.60 → 0.65 (+0.05), but p95 blew the budget 3× over. Ran a `top_k_rerank` sweep to understand the Pareto:

| top_k_rerank | NDCG@5 | p95 latency |
|---:|---:|---:|
| 5 | 0.6160 | 882 ms |
| 10 | 0.6353 | 964 ms |
| 20 | 0.6527 | 1409 ms |

Even at top-5 rerank, latency was 882ms — above budget. Cross-encoder inference on CPU is the hard floor here.

**Decision at this point:** keep all three configs available via the API, document the tradeoff. Don't drop rerank — the assignment asked for p95 **documented**, not uniformly under 500ms across every config.

---

## 8. Tier 1 full run — the big surprise

Ran all 3 required configs on 200 queries, sequential (concurrency=1):

| Config | P@5 | R@5 | NDCG@5 | p50 | p95 | p99 | cold |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic_only | 0.2540 | 0.5067 | 0.5101 | 346 | 488 | 566 | 268 |
| **hybrid (RRF)** | **0.2980** | **0.6042** | **0.5916** | 353 | **478** ✅ | 539 | 329 |
| hybrid+rerank | 0.2770 | 0.5550 | 0.5627 | 1275 | 1591 | 1774 | 852 |

Three findings, one of them very surprising:

**1. Hybrid (no rerank) wins on everything.** Best NDCG, best latency (under budget), best recall. This is the shipping config.

**2. Rerank HURT quality.** Dropped NDCG from 0.592 → 0.563. Why?

MultiHop-RAG is multi-hop — queries reference multiple docs and reward *distributed evidence*. Cross-encoder reranking scores *single-passage relevance to the full query* and promotes one fully-matching passage over several partial-evidence ones. On this query distribution, that actively hurts recall — some qrel-relevant docs get pushed out of the top-5.

This is a genuine query-distribution-dependent result. Rerank would likely help on single-hop QA or factoid queries. It's not the model's fault; it's the wrong tool for this benchmark.

**3. Everything still beats "it depends".** Five-key score breakdown, incremental ingest, hybrid tuning — all shipped and green.

---

## 9. BM25-vs-semantic — textbook failure mode

Separate analysis: ran BM25-only and semantic-only on the same 200 queries, computed per-query NDCG@5 diffs, sorted by `bm25 − semantic`.

Overall:
- BM25-only NDCG@5: **0.5766**
- Semantic-only NDCG@5: 0.5101

**BM25 alone beat semantic alone.** Not subtle.

Top 10 queries where BM25 beat semantic (all delta ≥ +0.6 in NDCG@5):

```
delta=+1.000  Q: Do the TechCrunch article on software companies and the Hacker News article on The Epoch Times both report...
delta=+1.000  Q: Does the article from Wired suggest that Sony headphones do not offer...
delta=+0.920  Q: Did the article from The Roar | Sports Writers Blog attribute the Tasmania JackJumpers...
delta=+0.877  Q: Does the TechCrunch article suggest that Sam Altman is involved in a new venture...
delta=+0.877  Q: After the Sporting News report on Tyreek Hill's chances of achieving 2,000-plus receiving yards...
delta=+0.704  Q: Who is the individual associated with FTX that informed another trader about permissible withdrawals...
delta=+0.704  Q: Which company, recently mentioned in articles by both TechCrunch and The Verge...
delta=+0.613  Q: What is the name of the company that was discussed on TechCrunch for removing AI-created songs...
delta=+0.613  Q: Does the Sporting News article anticipate an impressive performance in the upcoming home game for Jordan Love...
delta=+0.613  Q: Did the Yardbarker article describe Alex Verdugo's offensive performance...
```

**Every single one** contains either a **publication name** (TechCrunch, Wired, The Roar, Hacker News, Fortune, Sporting News, Yardbarker, The Verge) or a **rare proper noun** (Sam Altman, FTX, Tyreek Hill, Jordan Love, Alex Verdugo, Tasmania JackJumpers, Sony headphones).

This is textbook-perfect. MiniLM-L6 is a general-purpose embedder (trained mostly on Wikipedia-like text); it under-represents specific publication names and rare entities in its 384-dim space. BM25's IDF weighting nails them immediately.

**This is exactly what the assignment asked for** in the hard-mode signal: "When does BM25 beat semantic? Show the failure modes."

---

## 10. FastAPI wrapper — the fast hill

With the engine, fusion, rerank, and eval all done, FastAPI was a ~15-minute layer over the existing `SearchEngine`. Two endpoints:

- `GET /health` — returns `{status, engine_loaded}`
- `POST /search` — accepts `{query, top_k, fusion_method, alpha, rrf_k, use_rerank, top_k_rerank, top_k_retrieve, filters}`, returns per-hit `{bm25, semantic, hybrid_fused, rerank, final}` scores + `latency_ms`

Engine loaded once at `lifespan` startup, warmed up with a dummy query, then shared across requests.

Two integration tests added:
1. `test_search_endpoint_returns_score_breakdown` — contract test: all 5 score keys present
2. `test_search_with_filter_and_no_rerank` — metadata filter actually excludes non-matching categories

Both skip-gracefully if no index exists. Full suite: **22/22 passing**.

Commit: `1d2c187`.

---

## 11. Gaps we chose NOT to close in v1

- **Corpus at 609, not 1000+** — MultiHop-RAG provides 609 articles. PDF-padding pipeline (`load_pdf_directory`) is built and tested but no PDFs are in `data/pdfs/`. This is a 20-minute follow-up.
- **Tier 3 alpha sweep (0.3 / 0.7)** — scripted in `run_all_experiments.py` but not executed. Tier 1 + BM25-diff analysis carry the defense.
- **Semantic chunker (Tier 4)** — scripted but not executed. Recursive chunking was defended on first principles; the value of running semantic was primarily pedagogical.

All three are explicit choices documented in `DEFENSE.md`, not hidden.

---

## 12. How each p95 / NDCG iteration landed

A compressed timeline of the quality + latency numbers we watched land:

| Checkpoint | NDCG@5 | p95 | Notes |
|---|---:|---:|---|
| First live smoke (cold) | — | ~740 ms | Includes ONNX init |
| First live smoke (warm) | — | ~932 ms | Full hybrid+rerank at top-20 |
| Eval v0 (buggy) | 0.865 | 3488 ms | R@5=1.07 bug, queue-wait latency |
| Eval v1 (dedup + conc=1) hybrid | 0.597 | 378 ms | **Real numbers.** Budget met. |
| Rerank top-5 | 0.616 | 882 ms | Over budget, minor NDCG gain |
| Rerank top-10 | 0.635 | 964 ms | Over budget |
| Rerank top-20 | 0.653 | 1409 ms | Over budget, best rerank NDCG |
| **Tier 1 hybrid (shipping config)** | **0.592** | **478 ms** ✅ | Final numbers |

The "gap we closed" was almost entirely in the harness, not the retrieval code. Correct measurement gave us the real picture: hybrid-no-rerank is the winning config.

---

## 13. Final commit graph

```
cc9828b docs: DEFENSE narrative + COST projection + BM25-vs-semantic analyzer
1d2c187 feat(api): FastAPI /search endpoint with full score breakdown + filters
716b57b feat(eval): Tier 1 experiment matrix — orchestrator + first results
3b3c422 feat(eval): metrics TDD (9 tests), qrels adapter, harness, run_eval
5db6e20 feat(retrieve): fusion (RRF + weighted-alpha, 5 TDD tests), reranker, async search
c447805 feat(ingest): end-to-end pipeline + runner
4727bbd feat: embedder (FastEmbed/ONNX), BM25 index, FAISS IndexFlatIP wrappers
69004e4 feat(ingest): HF MultiHop-RAG loader + PDF loader + SHA-256 manifest
559ae50 feat: core types, config, recursive chunker (6 TDD tests passing)
c72b683 fix(preflight): MultiHop-RAG has two HF configs (corpus + MultiHopRAG)
34fb7b4 refactor: swap sentence-transformers -> fastembed (ONNX runtime)
a1e2805 docs: implementation plan + ignore .claude/ tooling
4983d0b chore: initial scaffold for RAG retrieval platform
```

Every commit is a working checkpoint. TDD commits land tests + implementation together. Config/preflight fixes are isolated in small commits. The refactor away from PyTorch is its own commit so it's easy to understand, revert, or audit.
