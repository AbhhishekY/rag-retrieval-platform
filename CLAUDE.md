# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Lint
ruff check src tests
ruff format src tests

# Test all
pytest

# Test single file or test
pytest tests/test_fusion.py
pytest tests/test_fusion.py::test_rrf_combines_both_lists

# First-time setup (~5 min, downloads models + corpus)
python scripts/preflight.py

# Ingest corpus (must run before starting API)
python scripts/ingest.py

# Ingest with non-default chunk size (creates separate index subdir)
python scripts/ingest.py --chunk-size 256 --index-subdir chunk256

# Start API
uvicorn rag.api.app:app --reload

# Run a single eval config
python scripts/run_eval.py --config hybrid+rerank   # semantic_only | hybrid | hybrid+rerank | bm25_only

# Run full experiment matrix
python scripts/run_all_experiments.py --tiers 1
```

## Architecture

The system is a single-process, CPU-only hybrid retrieval pipeline with no external service dependencies.

**Three shared dataclasses** (`src/rag/types.py`): `Document` → `Chunk` → `SearchHit`. Everything passes these types across module boundaries.

**One file for all tunables** (`src/rag/constants.py`): chunk size, overlap, top-k at each stage, fusion method, alpha, RRF k, batch sizes, model names. `src/rag/config.py` layers `.env` overrides on top via pydantic-settings. Never hardcode tuning knobs elsewhere.

**Ingest path** (`scripts/ingest.py` → `src/rag/ingest/pipeline.py`):
- Loads docs (MultiHop-RAG JSON from HuggingFace or local PDFs)
- Chunks via `recursive_chunk` (char-based cascade: `\n\n → \n → . → space → chars`)
- Embeds in batches via FastEmbed (ONNX, no PyTorch)
- Builds BM25 (pickle) + FAISS `IndexFlatIP` (inner product ≡ cosine on L2-normalized vecs)
- Writes all artifacts to `indices/<subdir>/`; `ingest_manifest.db` (SQLite) tracks sha256 per doc so unchanged docs are skipped on re-run

**Query path** (`src/rag/retrieve/search.py::SearchEngine`):
- BM25 tokenization + FAISS embed both run on the thread executor **in parallel** via `asyncio.gather(loop.run_in_executor(...))` — saves ~80–120ms vs serializing
- Fusion: RRF (rank-based, default) or weighted-α (per-query min-max normalized before `α·bm25 + (1−α)·dense`)
- Optional cross-encoder rerank via FastEmbed (`Xenova/ms-marco-MiniLM-L-6-v2`), always batched
- Returns `SearchHit` with a `scores` dict showing `{bm25, semantic, hybrid_fused, rerank, final}` at each stage

**API** (`src/rag/api/app.py`): FastAPI with `lifespan`-managed `SearchEngine`. Index must exist before startup (`POST /search`, `GET /health`). Eval scripts bypass the API and call `SearchEngine` directly.

**Multiple index subdirs** (`indices/default/`, `indices/chunk256/`, etc.): different chunking configs get separate subdirs so experiments don't cross-contaminate. The API always loads `indices/default/`.

**BM25 tokenization**: lowercase word-character only — no stemming, no stopword removal. Preserves named entities and numbers (important for news corpus).

## Key extension points

- **Swap models**: edit `EMBEDDING_MODEL` or `RERANKER_MODEL` in `src/rag/constants.py` (or `.env`), then re-run ingest
- **Scale to >100K chunks**: swap `faiss.IndexFlatIP` → `faiss.IndexHNSWFlat` in `src/rag/index/vector_index.py`; nothing else changes
- **Add a retriever**: implement `build / search / save / load` like `BM25Index`, add to `SearchEngine.__init__`, extend `fusion.py`

## Test coverage notes

- `recursive_chunk`: 6 TDD tests in `tests/test_chunking.py`
- `rrf_fuse` / `weighted_alpha_fuse`: 5 TDD tests in `tests/test_fusion.py`
- `precision_at_k` / `recall_at_k` / `ndcg_at_k`: 9 TDD tests in `tests/test_metrics.py`
- `POST /search` API: async tests in `tests/test_api.py` (pytest-asyncio, `asyncio_mode = "auto"`)
