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

# 3. Copy env template and edit if using Azure generation (optional)
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

- **Embeddings:** local `sentence-transformers/all-MiniLM-L6-v2` (384d, CPU, ~8ms/query)
- **BM25:** local `rank_bm25`
- **Vector index:** FAISS `IndexFlatIP` (exhaustive; optimal for <100K vectors)
- **Reranker:** local `cross-encoder/ms-marco-MiniLM-L-6-v2` (~25ms/pair CPU, batched)
- **Fusion:** RRF primary, α-weighted alternative (tunable via API)
- **Generation:** optional Azure OpenAI `gpt-4_1_dev_1` (retrieval doesn't need it)

## Deliverables

- `POST /search` — hybrid retrieval with score breakdown
- Evaluation across 3 configs (semantic / hybrid / hybrid+rerank) using MultiHop-RAG ground truth
- 9-run experiment matrix (chunk-size sweep, α sweep, strategy ablation)
- p95 latency profile (cold/warm)
- Cost-per-1K-queries projection

See [`docs/PLAN.md`](docs/PLAN.md) for the full implementation plan and [`_bmad-output/brainstorming/`](../) for the design session that produced it.
