# RAG Retrieval Platform

Production-grade hybrid retrieval: BM25 (sparse) + semantic dense vectors + optional cross-encoder reranking + metadata filtering. CPU-only, zero cloud dependencies, sub-200ms p95 latency.

---

## First time on a new machine — do this in order

### Step 1 — Run `run.py` (downloads, indexes, tests, evals)

```bash
git clone <repo-url>
cd rag-retrieval-platform
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

pip install -e ".[dev]"
python run.py
```

This single command does everything sequentially:

```
Step 1  Preflight    — downloads MiniLM encoder + MultiHop-RAG corpus from HuggingFace (~5 min)
Step 2  Ingest       — chunks (512 chars) → embeds (384-dim) → builds BM25 + FAISS indices (~10 min)
Step 3  Tests        — runs 22 unit + integration tests
Step 4  Eval ×4      — semantic_only / bm25_only / hybrid / hybrid+rerank (200 queries each)
Step 5  Summary      — prints NDCG@5, P@5, R@5, p95 latency table
```

First run: ~15-20 min (model downloads + embedding 19,817 chunks). Re-runs are fast — unchanged docs are skipped via SHA-256 manifest.

Useful flags:
```bash
python run.py --skip-preflight          # skip download (already cached)
python run.py --skip-ingest             # skip embedding (indices already built)
python run.py --skip-tests              # skip pytest
python run.py --limit 50                # 50 queries per config instead of 200
python run.py --verbose                 # stream subprocess output live
```

---

### Step 2 — Start the FastAPI server

After `run.py` completes (indices must exist), start the API:

```bash
uvicorn rag.api.app:app --reload
```

You'll see:
```
Loading embedder...
Loading reranker...
Loading indices...
Warming up...
API ready.
INFO: Uvicorn running on http://127.0.0.1:8000
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI, or use curl:

```bash
# Basic search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Goldman Sachs Q3 earnings"}'

# Search with metadata filter (only technology articles)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI chip shortage", "filters": {"category": "technology"}}'

# Search with custom fusion
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "FTX trial fraud", "fusion_method": "weighted", "alpha": 0.7}'

# Health check
curl http://localhost:8000/health
```

Response includes per-result score breakdown: `bm25`, `semantic`, `hybrid_fused`, `rerank`, `final`.

---

## One-shot bootstrap (alternative)

**Mac / Linux:**
```bash
bash bootstrap.sh
```

**Windows:**
```bat
bootstrap.bat
```

Does the same as the manual setup above — check Python 3.11+, create `.venv`, pip install, run `run.py`.

---

## Architecture

| Component | What it is |
|---|---|
| **BM25** | Sparse keyword index — exact match, great for named entities and publication names |
| **FAISS** | Dense vector database — 384-dim MiniLM embeddings, cosine similarity |
| **Encoder** | `all-MiniLM-L6-v2` via FastEmbed (ONNX, no PyTorch, no GPU needed) |
| **Fusion** | RRF (rank-based) or weighted alpha (`α·BM25 + (1-α)·dense`) |
| **Metadata filter** | Hard filter on `category`, `source_name`, `published_at`, or any metadata field |
| **Reranker** | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder (optional, disabled by default) |
| **Chunker** | Recursive char-level, 512 chars, 51-char overlap |
| **Manifest** | SQLite SHA-256 store — skips unchanged docs on re-ingest |

All tunables live in one file: `src/rag/constants.py`.

---

## Evaluation configs

Four retrieval configurations evaluated head-to-head:

| Config | What it tests |
|---|---|
| `semantic_only` | Dense vectors alone — FAISS + MiniLM |
| `hybrid` | BM25 + FAISS fused via RRF |
| `hybrid+rerank` | Hybrid + cross-encoder reranking |
| `hybrid+metadata_filter` | Weighted hybrid filtered by ground-truth category per query |

Run any single config:
```bash
python scripts/run_eval.py --config hybrid+metadata_filter --limit 200
```

Run all four at once:
```bash
python scripts/run_all_experiments.py --tiers 1 --limit 200
```

---

## Results (best config: weighted α=0.7)

| Metric | Value |
|---|---|
| NDCG@5 | **0.657** |
| P@5 | 0.354 |
| R@5 | 0.715 |
| p95 latency (warm) | **194ms** |
| p95 latency (cold first query) | 120ms |
| Cost per 1K queries | **$0.00** (no API calls, all local) |

Config: weighted alpha = 0.7, diversity filter (max 1 chunk per source doc), top_k_retrieve = 100.

Full results and charts: [`docs/DEFENSE.md`](docs/DEFENSE.md)

---

## Metadata filtering

Every chunk carries metadata from its source article:

```json
{
  "category": "technology",
  "published_at": "2023-10-01 14:00:29",
  "author": "Kyle Wiggers",
  "source_name": "TechCrunch"
}
```

Filter by any field via the API:

```bash
# Only business articles
curl -X POST http://localhost:8000/search \
  -d '{"query": "quarterly earnings", "filters": {"category": "business"}}'

# Multiple filter values (OR within a key)
curl -X POST http://localhost:8000/search \
  -d '{"query": "AI research", "filters": {"category": ["technology", "science"]}}'
```

The eval config `hybrid+metadata_filter` tests filtering with oracle category labels (ground-truth category from evidence_list) — this gives the upper bound of what metadata filtering can achieve on this dataset.

---

## Supported document formats

| Format | Loader | Notes |
|---|---|---|
| MultiHop-RAG JSON (HuggingFace) | `load_multihop_from_hf` | Default corpus |
| PDF | `load_pdf` / `load_pdf_directory` | Drop into `data/pdfs/`, re-run `ingest.py` |
| CSV | `load_csv` / `load_csv_directory` | Drop into `data/csvs/`, flexible column mapping |

Add supplementary CSVs (optional):
```bash
python scripts/fetch_supplementary.py   # downloads 400 AG News articles → data/csvs/
python scripts/ingest.py                # incremental — only embeds new docs
```

---

## Project layout

```
run.py                      one-shot pipeline runner (start here)
bootstrap.sh / .bat         fresh-machine setup scripts
scripts/
  preflight.py              download models + corpus
  ingest.py                 chunk → embed → index (--csv-dir, --pdf-dir flags)
  run_eval.py               single config eval
  run_all_experiments.py    full experiment matrix
  eval_by_type.py           stratified eval by question type
  fetch_supplementary.py    download AG News CSV supplement
  analyze_bm25_wins.py      BM25 vs semantic failure mode analysis
src/rag/
  constants.py              all tunables in one place
  ingest/                   chunker, loaders (HF+PDF+CSV), pipeline, manifest
  index/                    BM25Index, FaissFlatIndex
  retrieve/                 embedder, reranker, fusion, search
  eval/                     metrics, harness, qrels, reports
  api/                      FastAPI app + schemas
docs/
  DEFENSE.md                experiment results + design justifications
  ARCHITECTURE.md           component deep-dive + data flow diagrams
  BENCHMARK.md              why MultiHop-RAG, dataset shape, eval methodology
  JOURNEY.md                experiment narrative
indices/default/            built indices (after ingest)
data/                       corpus + queries (after preflight)
outputs/runs/               eval reports (JSON + markdown)
```
