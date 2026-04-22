# Architecture

## 30-second version

```
Query ──▶ [BM25]  ─┐
   │               ├─▶ Fusion ─▶ Rerank? ─▶ Top-5
   └─▶ [Embed]─▶ [FAISS] ─┘
```

All retrieval components run in one Python process. No external services, no network dependency. Ingest-time and query-time paths are decoupled via index artifacts on disk.

---

## Components

### `src/rag/types.py`
Three dataclasses shared across the system: `Document` (ingested), `Chunk` (post-split), `SearchHit` (query result with scores + metadata). Keep this file small — all types here are used everywhere.

### `src/rag/config.py`
Pydantic-settings over `.env`. Single source of truth for chunking defaults (512/10%), retrieval defaults (top_k_retrieve=100, rerank=20, final=5), fusion (`rrf` by default, `alpha=0.5`), model names. Tests read the same file — no config duplication.

### `src/rag/ingest/`
- `chunking.py::recursive_chunk` — pure function, 6 TDD tests. Char-based (deterministic); cascade on `\n\n → \n → . → space → chars`.
- `loaders.py::load_multihop_from_hf` — streams 609 articles from HF disk, yields `Document` with metadata preserved.
- `loaders.py::load_pdf` / `load_pdf_directory` — PyMuPDF extraction, graceful failure for unreadable PDFs.
- `loaders.py::load_csv` / `load_csv_directory` — flexible CSV loader; maps common column name variants (`text`/`body`/`content`, `url`/`id`, `category`/`label`, etc.) to `Document` fields. Drop any CSV into `data/csvs/`, re-run `ingest.py`.
- `manifest.py::IngestManifest` — SQLite table keyed on `doc_id` → `(content_sha256, chunk_count, indexed_at)`. `is_unchanged(doc_id, hash)` lets the pipeline skip re-embedding stable docs.
- `pipeline.py::run_ingest` — orchestrator. Loads docs → chunks → embeds → builds BM25 + FAISS → writes all artifacts atomically to an `index_dir` subdir (`default/`, `chunk256/`, etc.).

### `src/rag/index/`
- `bm25_index.py::BM25Index` — thin wrapper around `rank_bm25.BM25Okapi`. Lowercase word-character tokenization (no stemming, no stopword removal — news text preserves named entities and numbers). Persisted via pickle (chunk_ids + tokenized corpus, rebuild BM25Okapi on load).
- `vector_index.py::FaissFlatIndex` — wrapper around `faiss.IndexFlatIP`. For L2-normalized vectors, inner product ≡ cosine similarity. Persisted as `faiss.index` + `chunk_ids.pkl` side-by-side.

### `src/rag/retrieve/`
- `embedder.py::Embedder` — FastEmbed (ONNX Runtime) over `sentence-transformers/all-MiniLM-L6-v2`. Probes `dim` at init by doing one throwaway embed. Returns L2-normalized float32 vectors.
- `reranker.py::Reranker` — FastEmbed cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`). Always calls `rerank(query, docs, batch_size=N)` — never loops. One batched forward pass per query.
- `fusion.py::rrf_fuse` / `::weighted_alpha_fuse` — pure functions, 5 TDD tests. RRF is rank-based (no score scale issues). Weighted applies per-query min-max normalization before the `α·bm25 + (1−α)·dense` sum.
- `search.py::SearchEngine` — lifespan-held container that owns loaded indices + models. `async search()` kicks BM25 and embed onto the default thread executor in parallel (both are CPU-bound sync libs), then FAISS, then fusion, optional metadata filter, optional rerank. Returns `SearchHit` with `{bm25, semantic, hybrid_fused, rerank, final}` score dict per result.

### `src/rag/api/`
- `schemas.py` — Pydantic models: `SearchRequest` (validated ranges for top_k, alpha, fusion_method), `SearchResult` (scores + metadata), `SearchResponse` (results + latency_ms + config echo).
- `app.py` — FastAPI `lifespan`-managed engine. `/health` returns engine-loaded status; `POST /search` delegates to `SearchEngine.search(...)` and wraps the response with timing.

### `src/rag/eval/`
- `metrics.py` — `precision_at_k`, `recall_at_k`, `ndcg_at_k`. Pure, binary-relevance, 9 TDD tests.
- `qrels.py::load_multihop_eval` — reads the MultiHop-RAG queries config, maps each `evidence_list` → `set[url]` as the relevant-doc ground truth. Also extracts `category` (most common across evidence items) into `EvalQuery.category` — used by the `hybrid+metadata_filter` eval config.
- `harness.py::run_eval` — runs a batch of queries, isolates the first query's latency (cold-start), captures percentiles on the rest. Supports `concurrency=1` (accurate p95) and `concurrency>1` (throughput).
- `reports.py` — `save_report` (JSON + markdown per config) and `combine_reports_table` (multi-config summary).

### `scripts/`
- `preflight.py` — Hour-0 ritual: pre-download FastEmbed models + MultiHop-RAG corpus + queries + inspect formats.
- `ingest.py` — CLI wrapper over `pipeline.run_ingest` with `--chunk-size`, `--overlap`, `--index-subdir`, `--force` flags.
- `run_eval.py` — run one config (`semantic_only` | `hybrid` | `hybrid+rerank` | `bm25_only` | `hybrid+metadata_filter`), save one report.
- `run_all_experiments.py` — matrix runner with `--tiers {1,2,3}` selector.
- `analyze_bm25_wins.py` — per-query BM25-vs-semantic diff, writes top-10 BM25-wins to JSON.

---

## Data flow

### Ingest path (once per corpus, incremental on re-run)

```
 raw docs (MultiHop-RAG JSON + optional PDFs)
       │
       ▼
  Document dataclass  ── sha256(body) ──▶  manifest check
       │                                      │
       │                                      ├── unchanged → skip
       │                                      └── changed    ▼
       │                                                   chunk
       ▼                                                     │
   chunks (text + chunk_id=doc_id::chunk::N + metadata)     │
       │                                                     │
       ▼                                                     │
  FastEmbed.encode_docs(batch=64)  ◀───────────────────── (vectors)
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │ indices/<subdir>/                                   │
  │   bm25.pkl                (chunk_ids + tokenized)   │
  │   faiss/faiss.index       (N × 384 float32)         │
  │   faiss/chunk_ids.pkl     (row → chunk_id map)      │
  │   chunks.jsonl            (chunk_id → text+metadata)│
  │   ingest_manifest.db      (doc_id → sha256)         │
  └─────────────────────────────────────────────────────┘
```

### Query path (per-request)

```
  query string
       │
       ▼
  asyncio.gather(
       embedder.encode_query (threadpool) ──▶ 384d vec
       bm25.search(query, k=100) (threadpool) ─▶ [(chunk_id, bm25_score), ...]
  )
       │
       ▼
  faiss.search(vec, k=100)  ──▶ [(chunk_id, cosine), ...]
       │
       ▼
  fusion (RRF or weighted-α)  ──▶ top_k_rerank candidates
       │
       ▼
  metadata filter (optional)  ──▶ filtered candidates
       │
       ▼
  cross_encoder.rerank(query, [chunk_texts]) (threadpool)  ──▶ top_k_final
       │
       ▼
  render as SearchHit[] with {bm25, semantic, hybrid_fused, rerank, final}
```

Critical design choice: **BM25 and embedding both run on the threadpool in parallel** via `asyncio.gather(loop.run_in_executor(...))`. On CPU-only hardware this saves ~80–120ms per query vs serializing.

---

## Why these component boundaries

Each file has one clear responsibility — you can read it in isolation and understand it:

- **Chunking is pure** → trivial to test, trivial to swap for a different strategy
- **BM25 + FAISS have the same interface shape** (`build / search / save / load`) — easy to add a new retriever (e.g., ColBERT) later
- **Fusion lives outside search** → can be tested with hand-built rank lists, no need for a real index
- **Metrics live outside the harness** → can be unit-tested against small synthetic rankings
- **Search engine is the only place that touches everything** → one integration point, no hidden dependencies elsewhere
- **FastAPI is a wrapper** → the engine works without it; eval bypasses it entirely

This makes debugging fast: when a number looks weird, you can usually narrow to one file by thinking about which layer owns that number.

---

## Threading and async model

Everything retrieval-side is CPU-bound synchronous code (ONNX, rank-bm25, faiss, cross-encoder). The async surface exists to let callers `await` without blocking the event loop, and to parallelize the BM25 + embed legs via `loop.run_in_executor(None, sync_fn, ...)`.

No multiprocessing, no ray, no celery — we fit in one process comfortably on ~20K chunks with ~500 MB resident memory.

---

## Persistence model

Indices live on disk under `indices/<subdir>/`. Different configs (default, chunk256, chunk1024) get separate subdirs so experiments don't cross-contaminate.

`ingest_manifest.db` is per-subdir — re-running `ingest.py --index-subdir default` with unchanged docs is a no-op; changing the corpus re-embeds only the changed docs and rebuilds BM25+FAISS once (both are cheap to rebuild vs. maintaining inserts, so we just regenerate).

---

## Extension points

- **Swap embedder**: change `EMBEDDING_MODEL` in `src/rag/constants.py` or `.env` → FastEmbed downloads the new model → re-run ingest
- **Swap reranker**: change `RERANKER_MODEL` in `constants.py` or `.env` (fastembed supports `BAAI/bge-reranker-base`, `jinaai/jina-reranker-v1-tiny-en`, etc.)
- **Add a new retriever** (e.g., ColBERT): implement `build / search / save / load` like `BM25Index`, add it to `SearchEngine.__init__`, extend `fusion.py` to three-way fuse
- **Switch to ANN** if corpus grows past ~100K chunks: `faiss.IndexFlatIP` → `faiss.IndexHNSWFlat`, everything else unchanged
- **Tune retrieval knobs**: edit `src/rag/constants.py` — one file holds chunk size, overlap, top-k at each stage, fusion method, alpha, RRF k, batch sizes, model names
