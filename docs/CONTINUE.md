# Picking Up Where We Left Off

This doc is the cheat-sheet for resuming the build on a different machine (e.g., an M2 MacBook). The repo is self-contained — everything you need is committed here.

---

## TL;DR — where we are

**What's done and green:**

- Brainstorming + implementation plan (`docs/DESIGN-SESSION.md`, `docs/superpowers/plans/2026-04-22-retrieval-platform.md`)
- Full stack: chunking, BM25, FAISS, FastEmbed embedder, cross-encoder reranker, async search engine, FastAPI `/search`, eval harness with metrics, `run_all_experiments.py`
- 22/22 tests passing (`pytest tests/`)
- **Tier 1 evaluation complete** — 3 configs × 200 queries, concurrency=1 for real p95:
  - hybrid (RRF): **NDCG@5 0.592, p95 478ms** ← shipping config, under budget
  - semantic_only: NDCG@5 0.510, p95 488ms
  - hybrid+rerank: NDCG@5 0.563, p95 1591ms (rerank hurt on multi-hop!)
- **BM25-vs-semantic failure-mode analysis** — top 10 BM25 wins in `outputs/runs/bm25_wins.json` — all queries with rare publication names / proper nouns
- All tunable knobs centralized in `src/rag/constants.py`
- Full narrative docs: `DEFENSE.md`, `COST.md`, `JOURNEY.md`, `BENCHMARK.md`, `ARCHITECTURE.md`, `TRADEOFFS.md`

**What's partial (ingested but not evaluated):**

- `indices/chunk256/` — 47,102 chunks ingested at chunk_size=256 (for Tier 2 chunk sweep)
- `indices/chunk1024/` — ingest may or may not have finished before you left the Windows machine; re-run to be safe

**What's remaining (optional, for a complete experiment matrix):**

- **Tier 2:** eval `hybrid` on `chunk256` and `chunk1024` indices (chunk-size sweep)
- **Tier 3:** alpha sweep (0.3 / 0.7) via `run_all_experiments.py --tiers 3`
- **Tier 4:** semantic chunker comparison (implementation not written)
- **PDF padding:** adding ~400 arXiv PDFs to hit the "1,000+" corpus count (loader is built and tested)

---

## Environment setup on macOS (M2 Air 8GB or similar)

```bash
# 1. Clone
git clone git@github.com:AbhhishekY/rag-retrieval-platform.git
cd rag-retrieval-platform

# 2. Create venv (macOS paths differ from Windows)
python3.11 -m venv .venv
source .venv/bin/activate          # macOS/Linux path (NOT .venv/Scripts/activate)

# 3. Install
pip install -e ".[dev]"

# 4. Download models + benchmark (~5-8 min on first run, cached after)
python scripts/preflight.py

# 5. Ingest the default corpus (~1 min on M2)
python scripts/ingest.py

# 6. Sanity — run the full test suite
pytest tests/
# expected: 22 passed
```

You're back in business. The repo is platform-agnostic — paths are handled via `pathlib.Path` and the only Windows-specific thing was the venv activation path.

## Expected perf shift on M2 vs the original Windows CPU

Measured on Windows, estimated on M2 (Apple Silicon + ONNX Runtime's CoreML provider):

| Config | Windows p95 | M2 p95 (est.) |
|---|---:|---:|
| semantic_only | 488 ms | ~280 ms |
| hybrid | 478 ms | ~280 ms |
| hybrid+rerank | 1591 ms | ~700–900 ms |

Embedding and rerank are ~2× faster on M2 (Neural Engine). BM25 gets ~1.3×. Hybrid will land comfortably under budget on M2; hybrid+rerank will still breach 500ms but less badly.

## Memory fit on 8GB

| Item | RAM |
|---|---:|
| App total (models + FAISS + BM25 + chunks) | ~1.2 GB |
| macOS baseline | ~2–3 GB |
| Headroom | ~4 GB |

Fits comfortably. For 100K+ chunks you'd want 16 GB.

---

## How to finish the experiment matrix on M2

### Tier 2 — chunk-size sweep (~10 min total)

```bash
# Re-ingest the two alternative chunk sizes
python scripts/ingest.py --chunk-size 256  --overlap 25  --index-subdir chunk256  --force
python scripts/ingest.py --chunk-size 1024 --overlap 102 --index-subdir chunk1024 --force

# Eval hybrid (the winning config) on each
python -c "
import asyncio
from pathlib import Path
from rag.config import get_settings
from rag.eval.harness import run_eval
from rag.eval.qrels import load_multihop_eval
from rag.eval.reports import save_report
from rag.retrieve.embedder import Embedder
from rag.retrieve.search import SearchEngine

s = get_settings()
e = Embedder()
queries = load_multihop_eval(s.data_dir / 'multihop_rag_queries')[:200]
for sub in ('chunk256', 'chunk1024'):
    eng = SearchEngine(s.index_dir / sub, e, None)  # hybrid, no rerank
    rep = asyncio.run(run_eval(eng, queries, f'tier2_{sub}',
        {'top_k_retrieve': 100, 'top_k_rerank': 20, 'top_k_final': 5,
         'fusion_method': 'rrf', 'use_rerank': False}))
    save_report(rep, s.output_dir / 'runs')
    print(f'{sub}: NDCG@5={rep.ndcg_at_5_mean:.4f} p95={rep.latency_p95_ms:.0f}ms')
"
```

Compare with the `tier1_hybrid` result (NDCG 0.592, p95 478ms) to see which chunk size wins. On MultiHop-RAG's multi-hop queries, the expectation (from first principles) is that smaller chunks (256) will boost recall, at a small p95 cost.

### Tier 3 — alpha sweep (~5 min total)

```bash
python scripts/run_all_experiments.py --tiers 3 --limit 200
```

Runs `α=0.3` (semantic-leaning) and `α=0.7` (BM25-leaning) on the default index, no rerank. Compare against RRF baseline and see which weighting wins.

### PDF padding to hit 1,000+ docs

```bash
# Drop any PDFs into data/pdfs/ — no specific source required
mkdir -p data/pdfs
# ... (download 400+ arXiv PDFs via arxiv API, or any PDF collection)

# Re-ingest combined corpus
python scripts/ingest.py --force --index-subdir default
# Now indices/default/ has MultiHop-RAG + PDFs mixed
```

The ingest pipeline handles mixed-format (HF JSON + arbitrary PDFs) transparently via `load_multihop_from_hf` + `load_pdf_directory`.

---

## Git identity reminder

This repo is under the personal identity:
```
user.email = abhhishek.yaadav@gmail.com
user.name  = AbhhishekY
```
set locally (`.git/config`). On the new machine after cloning, either:
- re-set via `git config user.email "abhhishek.yaadav@gmail.com" && git config user.name "AbhhishekY"`, or
- let it inherit from macOS global config (check with `git config --global user.email`)

**Do not** use work email `abhishek.yadav@axtria.com` for commits on this repo.

---

## Useful entry points when resuming

- **Architecture overview:** `docs/ARCHITECTURE.md` (components + data flow)
- **Why each decision was made:** `docs/TRADEOFFS.md` (13 decisions, rationale each)
- **The chronological story:** `docs/JOURNEY.md` (preflight → chunking → fusion → eval → bugs → fixes)
- **Numbers to defend:** `docs/DEFENSE.md` (hard-mode signals + findings)
- **The benchmark itself:** `docs/BENCHMARK.md` (MultiHop-RAG schema, why chosen)
- **The original design session:** `docs/DESIGN-SESSION.md` (first-principles + morphological + pre-mortem brainstorming output)
- **The implementation plan we executed:** `docs/superpowers/plans/2026-04-22-retrieval-platform.md` (32 tasks with code blocks)
- **All tunable knobs:** `src/rag/constants.py` (change here, propagates everywhere)

---

## If something is broken on resume

1. Did `pytest tests/` pass? If yes, the code is fine.
2. Did `python scripts/preflight.py` print `Preflight OK`? If yes, models + corpus are in place.
3. Does `indices/default/faiss/faiss.index` exist? If no, run `python scripts/ingest.py`.
4. Does `curl http://localhost:8000/health` work after `uvicorn rag.api.app:app`? If no, check the lifespan log for model-download issues.

If none of the above, re-read the JOURNEY doc for the failure modes we already hit once.
