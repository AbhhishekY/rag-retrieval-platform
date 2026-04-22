# RAG Retrieval Platform

Production-grade retrieval system: hybrid BM25 + semantic + cross-encoder reranking, over 1,000+ mixed-format documents, with evaluation harness and sub-500ms p95 latency budget.

## Quick start

```bash
# 1. Activate venv
source .venv/Scripts/activate         # Windows Git Bash
# .venv\Scripts\activate              # Windows CMD
# source .venv/bin/activate           # Linux/Mac

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Copy env template (optional — all defaults in src/rag/constants.py)
cp .env.example .env

# 4. Run preflight (downloads models + benchmark corpus, ~5 min)
python scripts/preflight.py

# 5. Ingest corpus
python scripts/ingest.py

# 6. Start API
uvicorn rag.api.app:app --reload

# 7. Run evaluation
python scripts/run_eval.py --config hybrid+rerank
```

## Architecture

- **Embeddings:** FastEmbed (ONNX) `sentence-transformers/all-MiniLM-L6-v2` (384d, CPU, ~8ms/query)
- **BM25:** local `rank_bm25`
- **Vector index:** FAISS `IndexFlatIP` (exhaustive; optimal for <100K vectors)
- **Reranker:** FastEmbed (ONNX) `Xenova/ms-marco-MiniLM-L-6-v2` (CPU, batched)
- **Fusion:** RRF primary, α-weighted alternative (tunable via API or `constants.py`)
- **Tuning:** every knob (chunk size, overlap, top-k, fusion, batch sizes, models) lives in one file — `src/rag/constants.py`

## Deliverables

- `POST /search` — hybrid retrieval with score breakdown
- Evaluation across 3 configs (semantic / hybrid / hybrid+rerank) using MultiHop-RAG ground truth
- 9-run experiment matrix (chunk-size sweep, α sweep, strategy ablation)
- p95 latency profile (cold/warm)
- Cost-per-1K-queries projection

See [`docs/superpowers/plans/2026-04-22-retrieval-platform.md`](docs/superpowers/plans/2026-04-22-retrieval-platform.md) for the full implementation plan (32 tasks) and `_bmad-output/brainstorming/` for the design session that produced it.
