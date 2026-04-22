"""Single source of truth for tunable parameters across the retrieval platform.

Every numeric default, model name, or fusion mode that a reviewer might want to
change lives here. `config.py` reads from here and layers `.env` overrides on
top — so you can tune via environment variables without editing code, but the
defaults all live in one file.

If you find yourself wanting to hardcode a tuning knob somewhere else in the
codebase, STOP and add it here instead.
"""
from __future__ import annotations

# ─── Models (local; FastEmbed / ONNX Runtime) ───────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384  # MiniLM-L6 vector dimension; used for FAISS index init

RERANKER_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"

# ─── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512           # characters per chunk (not tokens)
CHUNK_OVERLAP: int = 51         # ~10% of chunk_size — duplication insurance
CHUNK_STRATEGY: str = "recursive"  # recursive | fixed | semantic

# ─── Retrieval ──────────────────────────────────────────────────────────────
TOP_K_RETRIEVE: int = 100       # candidates pulled from each of BM25 and dense
TOP_K_RERANK: int = 20          # candidates passed to cross-encoder
TOP_K_FINAL: int = 5            # results returned to caller

# ─── Fusion ─────────────────────────────────────────────────────────────────
FUSION_METHOD: str = "rrf"      # rrf | weighted | semantic_only | bm25_only
HYBRID_ALPHA: float = 0.5       # used only when FUSION_METHOD == "weighted"
RRF_K: int = 60                 # RRF standard constant (rarely tuned)

# ─── Rerank behavior ────────────────────────────────────────────────────────
USE_RERANK_DEFAULT: bool = True

# ─── Batch sizes (throughput-influencing) ──────────────────────────────────
EMBED_BATCH_SIZE: int = 64      # docs per embedder forward pass
RERANK_BATCH_SIZE: int = 32     # (query, doc) pairs per cross-encoder forward pass

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR: str = "./data"
INDEX_DIR: str = "./indices"
OUTPUT_DIR: str = "./outputs"
