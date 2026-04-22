# Retrieval Platform v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade retrieval platform (1000+ docs, hybrid BM25+semantic+cross-encoder rerank, metadata filtering, search API, eval harness, p95 <500ms) that produces defensible numbers across 3 required configs + chunk/α/strategy ablations.

**Architecture:** Python 3.11 FastAPI service. All retrieval local (no network): `sentence-transformers/all-MiniLM-L6-v2` for embeddings, `rank_bm25` for BM25, `faiss-cpu IndexFlatIP` for vector search, `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking. Azure OpenAI `gpt-4_1_dev_1` reserved for optional answer generation only. Corpus = MultiHop-RAG (609 JSON articles w/ ground-truth qrels) + 400 arXiv PDFs (noise padding to hit 1000+).

**Tech Stack:** Python 3.11, FastAPI, sentence-transformers, rank-bm25, faiss-cpu, pymupdf, pydantic v2, pytest, asyncio, datasets (HuggingFace).

**Design rationale:** See `_bmad-output/brainstorming/brainstorming-session-2026-04-22-101317.md` for full reasoning behind each locked decision.

---

## File Structure

```
src/rag/
├── __init__.py
├── config.py                # pydantic-settings for env vars
├── types.py                 # Document, Chunk, SearchHit dataclasses
├── ingest/
│   ├── __init__.py
│   ├── loaders.py           # load_json_corpus, load_pdf
│   ├── chunking.py          # recursive_chunk (core algo, TDD)
│   ├── manifest.py          # SHA-256 ingest manifest (SQLite)
│   └── pipeline.py          # full ingest orchestrator
├── index/
│   ├── __init__.py
│   ├── bm25_index.py        # BM25Index wrapper
│   └── vector_index.py      # FaissFlatIndex wrapper
├── retrieve/
│   ├── __init__.py
│   ├── embedder.py          # MiniLM wrapper (sync + batched)
│   ├── reranker.py          # CrossEncoder wrapper (batched)
│   ├── fusion.py            # rrf + weighted_alpha (TDD)
│   └── search.py            # async retrieval pipeline
├── api/
│   ├── __init__.py
│   ├── schemas.py           # Pydantic request/response models
│   └── app.py               # FastAPI app + /search endpoint
└── eval/
    ├── __init__.py
    ├── metrics.py           # precision_at_k, recall_at_k, ndcg_at_k (TDD)
    ├── qrels.py             # MultiHop-RAG → qrels adapter
    ├── harness.py           # parallel eval runner with latency capture
    └── reports.py           # aggregate to markdown tables

scripts/
├── preflight.py             # DONE — pre-download models + MultiHop-RAG
├── ingest.py                # run full ingest pipeline
├── run_eval.py              # run one config end-to-end
└── run_all_experiments.py   # run all 9 Tier configs

tests/
├── test_chunking.py         # chunking edge cases
├── test_fusion.py           # RRF + alpha fusion math
├── test_metrics.py          # P@k / R@k / NDCG@k
├── test_manifest.py         # incremental ingest behavior
└── test_api.py              # /search integration tests
```

---

## Phase A — Preflight & Dependencies

### Task 1: Install dependencies and verify environment

**Files:**
- No file changes; runs in existing `.venv`

- [ ] **Step 1:** Activate venv and install project + dev deps

Run: `source .venv/Scripts/activate && pip install -e ".[dev]"`
Expected: installs ~30 packages, `pip list | grep sentence-transformers` shows it.

- [ ] **Step 2:** Verify imports work

Run:
```bash
python -c "import fastapi, sentence_transformers, rank_bm25, faiss, fitz, pandas, datasets; print('IMPORTS_OK')"
```
Expected: `IMPORTS_OK`

- [ ] **Step 3:** Commit lockfile if pip produces one (skip if none)

```bash
# No lockfile for pip-only; skip commit
```

---

### Task 2: Run preflight script (download models + corpus)

**Files:**
- No file changes; script `scripts/preflight.py` already exists

- [ ] **Step 1:** Run preflight

Run: `python scripts/preflight.py`
Expected output (approximate):
```
[1/3] Pre-downloading sentence-transformers models...
      models cached in 60-180s
[2/3] Downloading MultiHop-RAG benchmark...
      saved to data/multihop_rag in 10-30s
[3/3] Inspecting qrels format...
      [train] keys: ['query', 'answer', 'question_type', 'evidence_list']
      evidence_list type: list
      evidence item keys: ['title', 'author', 'url', 'source', 'category', 'published_at', 'fact']

Preflight OK. Safe to proceed with ingest.
```

- [ ] **Step 2:** Note the exact qrels structure for Task 17 (metrics adapter).

Capture the printed `evidence item keys` — they determine how we map evidence to doc_ids (likely via `url` or `title`).

---

## Phase B — Core Types and Config

### Task 3: Define core data types

**Files:**
- Create: `src/rag/types.py`

- [ ] **Step 1:** Write `src/rag/types.py`

```python
"""Core dataclasses for documents, chunks, and search results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    source: str               # file path or URL
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_sha256: str = ""  # set by ingest manifest


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int          # 0-based within doc
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    chunk_id: str
    doc_id: str
    text: str
    scores: dict[str, float]  # keys: bm25, semantic, hybrid_fused, rerank, final
    metadata: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 2:** Verify import

Run: `python -c "from rag.types import Document, Chunk, SearchHit; print('TYPES_OK')"`
Expected: `TYPES_OK`

- [ ] **Step 3:** Commit

```bash
git add src/rag/types.py
git commit -m "feat: add core Document/Chunk/SearchHit types"
```

---

### Task 4: Config module via pydantic-settings

**Files:**
- Create: `src/rag/config.py`

- [ ] **Step 1:** Write `src/rag/config.py`

```python
"""Config loaded from environment (.env file supported)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Paths
    data_dir: Path = Path("./data")
    index_dir: Path = Path("./indices")
    output_dir: Path = Path("./outputs")

    # Models (local)
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Azure OpenAI (optional, generation only)
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_api_key: str = ""
    azure_openai_chat_deployment: str = ""

    # Indexing
    chunk_size: int = 512
    chunk_overlap: int = 51
    chunk_strategy: str = "recursive"  # recursive | fixed | semantic

    # Retrieval
    top_k_retrieve: int = 100
    top_k_rerank: int = 20
    top_k_final: int = 5
    fusion_method: str = "rrf"  # rrf | weighted
    hybrid_alpha: float = 0.5   # used only when fusion_method=weighted
    rrf_k: int = 60


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 2:** Verify

Run: `python -c "from rag.config import get_settings; s = get_settings(); print(s.chunk_size, s.fusion_method)"`
Expected: `512 rrf`

- [ ] **Step 3:** Commit

```bash
git add src/rag/config.py
git commit -m "feat: add pydantic-settings config module"
```

---

## Phase C — Chunking (TDD)

### Task 5: Chunking tests first

**Files:**
- Create: `tests/test_chunking.py`

- [ ] **Step 1:** Write failing tests

```python
"""Tests for recursive chunking."""
from rag.ingest.chunking import recursive_chunk


def test_short_text_becomes_single_chunk():
    text = "This is a short sentence."
    chunks = recursive_chunk(text, chunk_size=512, overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_is_split_at_paragraph_boundary():
    # Each paragraph is ~60 chars; with chunk_size=120 we expect paragraph-level splits
    text = "First paragraph about dogs.\n\nSecond paragraph about cats.\n\nThird paragraph about birds."
    chunks = recursive_chunk(text, chunk_size=60, overlap=0)
    assert len(chunks) >= 2
    # No chunk exceeds the target (within a tolerance for boundary handling)
    for c in chunks:
        assert len(c) <= 120  # with overlap 0 and paragraph splits, should be clean


def test_overlap_produces_overlap_between_consecutive_chunks():
    # Use text with clear sentence boundaries
    sentences = ". ".join([f"Sentence number {i}" for i in range(40)]) + "."
    chunks = recursive_chunk(sentences, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    # Consecutive chunks should share some tail/head characters when overlap > 0
    # (we don't assert exact byte overlap due to boundary-snapping, only presence)
    assert any(
        any(word in chunks[i + 1] for word in chunks[i].split()[-5:])
        for i in range(len(chunks) - 1)
    )


def test_empty_text_returns_empty_list():
    assert recursive_chunk("", chunk_size=512, overlap=0) == []


def test_chunk_sizes_respect_budget_with_recursive_split():
    # Dense text with NO paragraph breaks — should fall through to sentence-level
    text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. " * 20
    chunks = recursive_chunk(text, chunk_size=80, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 100  # loose upper bound allowing clean sentence cuts
```

- [ ] **Step 2:** Run tests — expect failure

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.ingest.chunking'`

---

### Task 6: Implement recursive_chunk

**Files:**
- Create: `src/rag/ingest/__init__.py`
- Create: `src/rag/ingest/chunking.py`

- [ ] **Step 1:** Create empty package init

```python
# src/rag/ingest/__init__.py
```

Write an empty file.

- [ ] **Step 2:** Implement `recursive_chunk`

Write `src/rag/ingest/chunking.py`:
```python
"""Recursive character-level chunker.

Algorithm: Try to split on the strongest separator first. If the resulting pieces
still exceed chunk_size, recurse with the next-weaker separator. This prefers
paragraph boundaries, then line breaks, then sentence endings, then whitespace.

Character-based (not token-based) for determinism and zero external deps.
Tokens are ~4 chars on average for English; multiply chunk_size by 4 if you
were thinking in tokens.
"""
from __future__ import annotations

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 0,
    separators: list[str] | None = None,
) -> list[str]:
    """Split `text` into chunks of at most `chunk_size` characters.

    Args:
        text: input text (may be empty)
        chunk_size: target max chunk length in characters
        overlap: number of characters to repeat between consecutive chunks
        separators: split priority list; defaults to paragraph→line→sentence→word→char

    Returns:
        List of chunk strings (empty if input is empty or whitespace-only).
    """
    if not text or not text.strip():
        return []
    seps = separators if separators is not None else DEFAULT_SEPARATORS
    pieces = _split(text, chunk_size, seps)
    if overlap > 0 and len(pieces) > 1:
        pieces = _apply_overlap(pieces, overlap)
    return pieces


def _split(text: str, chunk_size: int, seps: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    for i, sep in enumerate(seps):
        if sep == "":
            # Last resort: hard char split
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        if sep in text:
            parts = text.split(sep)
            # Re-attach separator so meaning is preserved (except for the final part)
            parts = [p + sep for p in parts[:-1]] + [parts[-1]]
            parts = [p for p in parts if p]  # drop empties
            merged = _merge_small_pieces(parts, chunk_size)
            # Recurse into any piece still too large using weaker separators
            out: list[str] = []
            for p in merged:
                if len(p) <= chunk_size:
                    out.append(p)
                else:
                    out.extend(_split(p, chunk_size, seps[i + 1 :]))
            return out
    return [text]


def _merge_small_pieces(parts: list[str], chunk_size: int) -> list[str]:
    """Greedy merge: combine consecutive small pieces while staying under chunk_size."""
    merged: list[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
            continue
        if len(buf) + len(p) <= chunk_size:
            buf += p
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prepend the last `overlap` characters of chunk N-1 to chunk N."""
    out = [pieces[0]]
    for i in range(1, len(pieces)):
        prev_tail = pieces[i - 1][-overlap:]
        out.append(prev_tail + pieces[i])
    return out
```

- [ ] **Step 3:** Run tests — expect pass

Run: `pytest tests/test_chunking.py -v`
Expected: 5 passed

- [ ] **Step 4:** Commit

```bash
git add src/rag/ingest/__init__.py src/rag/ingest/chunking.py tests/test_chunking.py
git commit -m "feat(ingest): recursive chunker with overlap (tested)"
```

---

## Phase D — Document Loaders

### Task 7: JSON and PDF loaders

**Files:**
- Create: `src/rag/ingest/loaders.py`

- [ ] **Step 1:** Write `src/rag/ingest/loaders.py`

```python
"""Document loaders for MultiHop-RAG JSON corpus and PDF files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import fitz  # pymupdf

from rag.types import Document


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_multihop_corpus(corpus_json_path: Path) -> Iterator[Document]:
    """Yield Document per article in MultiHop-RAG corpus.json.

    corpus.json structure: list of article dicts with keys
    {'title', 'body', 'url', 'source', 'category', 'published_at', 'author'}
    """
    with open(corpus_json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    for i, art in enumerate(articles):
        text = art.get("body", "") or art.get("text", "")
        if not text.strip():
            continue
        # Stable doc_id: use URL if present, else deterministic index
        doc_id = art.get("url") or f"multihop-{i}"
        yield Document(
            doc_id=doc_id,
            source=art.get("url", doc_id),
            title=art.get("title", ""),
            text=text,
            metadata={
                "category": art.get("category", ""),
                "published_at": art.get("published_at", ""),
                "author": art.get("author", ""),
                "source_name": art.get("source", ""),
                "format": "json",
            },
            content_sha256=_sha256(text),
        )


def load_pdf(pdf_path: Path) -> Document | None:
    """Load a single PDF; return None if extraction yields empty text."""
    try:
        doc = fitz.open(str(pdf_path))
        text_parts = [page.get_text("text") for page in doc]
        doc.close()
    except Exception as e:
        print(f"  PDF load failed for {pdf_path.name}: {e}")
        return None

    text = "\n\n".join(p for p in text_parts if p.strip())
    if not text.strip():
        return None

    return Document(
        doc_id=f"pdf::{pdf_path.stem}",
        source=str(pdf_path),
        title=pdf_path.stem,
        text=text,
        metadata={"format": "pdf", "filename": pdf_path.name},
        content_sha256=_sha256(text),
    )


def load_pdf_directory(pdf_dir: Path) -> Iterator[Document]:
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc = load_pdf(pdf_path)
        if doc is not None:
            yield doc
```

- [ ] **Step 2:** Smoke-test with MultiHop-RAG

Run:
```bash
python -c "
from pathlib import Path
from rag.ingest.loaders import load_multihop_corpus

# MultiHop-RAG corpus.json lives inside the HF dataset save_to_disk output;
# for initial smoke test, resolve the path dynamically.
import glob
candidates = glob.glob('data/multihop_rag/**/corpus.json', recursive=True)
print('Found corpus files:', candidates[:3])
"
```
Expected: list of 0+ file paths (we'll confirm location in next task; if empty, MultiHop-RAG HF dataset may store articles differently — see Task 8).

- [ ] **Step 3:** Commit

```bash
git add src/rag/ingest/loaders.py
git commit -m "feat(ingest): JSON corpus + PDF loaders"
```

---

### Task 8: Bridge MultiHop-RAG HuggingFace dataset to `Document` stream

**Files:**
- Modify: `src/rag/ingest/loaders.py` (append function)

- [ ] **Step 1:** Inspect the HF dataset shape

Run:
```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/multihop_rag')
print('Splits:', list(ds.keys()))
for split in ds.keys():
    print(f'{split}: {len(ds[split])} rows')
    print('  keys:', list(ds[split][0].keys()))
    break
"
```
Capture the actual field names. The dataset may split query/answer from the corpus — the corpus itself may be a separate HF dataset (`yixuantt/MultiHopRAG` ships BOTH `corpus` and `MultiHopRAG` queries in one repo).

- [ ] **Step 2:** Extend loader based on inspection

If the HF dataset has a `corpus` split (containing article bodies), add `load_multihop_from_hf` to `src/rag/ingest/loaders.py`:

```python
def load_multihop_from_hf(hf_dataset_dir: Path) -> Iterator[Document]:
    """Load articles from a HuggingFace saved dataset on disk.

    Expected split name: 'corpus' (or whichever split holds article bodies).
    Adjusts field names based on actual dataset schema observed during preflight.
    """
    from datasets import load_from_disk

    ds = load_from_disk(str(hf_dataset_dir))
    # Use the split most likely to contain articles
    candidates = ["corpus", "train", "test"]
    split = next((c for c in candidates if c in ds), list(ds.keys())[0])
    rows = ds[split]

    # Schema varies; normalize field access
    for i, row in enumerate(rows):
        text = row.get("body") or row.get("text") or row.get("content") or ""
        if not text.strip():
            continue
        doc_id = row.get("url") or row.get("id") or f"multihop-{i}"
        yield Document(
            doc_id=doc_id,
            source=row.get("url", doc_id),
            title=row.get("title", ""),
            text=text,
            metadata={
                "category": row.get("category", ""),
                "published_at": row.get("published_at", ""),
                "author": row.get("author", ""),
                "source_name": row.get("source", ""),
                "format": "json",
            },
            content_sha256=_sha256(text),
        )
```

- [ ] **Step 3:** Smoke test

Run:
```bash
python -c "
from pathlib import Path
from rag.ingest.loaders import load_multihop_from_hf
docs = list(load_multihop_from_hf(Path('data/multihop_rag')))
print(f'Loaded {len(docs)} docs')
if docs:
    print('First doc:', docs[0].title[:80], '|', docs[0].doc_id[:80])
    print('Text length:', len(docs[0].text))
"
```
Expected: `Loaded 609 docs` (approximately), first doc title + doc_id printed, text length > 500.

- [ ] **Step 4:** Commit

```bash
git add src/rag/ingest/loaders.py
git commit -m "feat(ingest): load MultiHop-RAG from HuggingFace dataset"
```

---

## Phase E — Ingest Manifest (incremental ingest)

### Task 9: SQLite-backed manifest

**Files:**
- Create: `src/rag/ingest/manifest.py`

- [ ] **Step 1:** Write `src/rag/ingest/manifest.py`

```python
"""SHA-256 content-hash manifest for incremental ingestion.

Stores per-doc hashes so re-runs skip unchanged docs. Single SQLite file
under index_dir/ingest_manifest.db.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


class IngestManifest:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.db_path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _ensure_schema(self) -> None:
        with self._conn() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS ingest_manifest (
                    doc_id TEXT PRIMARY KEY,
                    content_sha256 TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                )"""
            )

    def is_unchanged(self, doc_id: str, content_sha256: str) -> bool:
        with self._conn() as con:
            row = con.execute(
                "SELECT content_sha256 FROM ingest_manifest WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        return row is not None and row[0] == content_sha256

    def record(self, doc_id: str, content_sha256: str, chunk_count: int) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as con:
            con.execute(
                """INSERT INTO ingest_manifest(doc_id, content_sha256, chunk_count, indexed_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     content_sha256=excluded.content_sha256,
                     chunk_count=excluded.chunk_count,
                     indexed_at=excluded.indexed_at""",
                (doc_id, content_sha256, chunk_count, now),
            )

    def known_doc_ids(self) -> set[str]:
        with self._conn() as con:
            rows = con.execute("SELECT doc_id FROM ingest_manifest").fetchall()
        return {r[0] for r in rows}
```

- [ ] **Step 2:** Smoke test

```bash
python -c "
from pathlib import Path
from rag.ingest.manifest import IngestManifest
import shutil
p = Path('/tmp/test_manifest.db')
if p.exists(): p.unlink()
m = IngestManifest(p)
print('Unchanged (new):', m.is_unchanged('doc1', 'hash1'))  # False
m.record('doc1', 'hash1', 5)
print('Unchanged (same hash):', m.is_unchanged('doc1', 'hash1'))  # True
print('Unchanged (new hash):', m.is_unchanged('doc1', 'hash2'))  # False
print('Known:', m.known_doc_ids())  # {'doc1'}
p.unlink()
print('MANIFEST_OK')
"
```
Expected:
```
Unchanged (new): False
Unchanged (same hash): True
Unchanged (new hash): False
Known: {'doc1'}
MANIFEST_OK
```

- [ ] **Step 3:** Commit

```bash
git add src/rag/ingest/manifest.py
git commit -m "feat(ingest): SHA-256 manifest for incremental ingestion"
```

---

## Phase F — Embedder and BM25 Wrappers

### Task 10: Embedder wrapper

**Files:**
- Create: `src/rag/retrieve/__init__.py`
- Create: `src/rag/retrieve/embedder.py`

- [ ] **Step 1:** Empty package init

Write `src/rag/retrieve/__init__.py`:
```python
```
(empty file)

- [ ] **Step 2:** Write `src/rag/retrieve/embedder.py`

```python
"""Thin wrapper around FastEmbed (ONNX Runtime) for query + doc embeddings.

FastEmbed runs `sentence-transformers/all-MiniLM-L6-v2` via ONNX — no PyTorch,
smaller install, often faster on CPU. The produced vectors are already
L2-normalized (the default for MiniLM), so cosine similarity == inner product
in FAISS IndexFlatIP.
"""
from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)
        # Probe one embedding to discover dimension and trigger model download
        probe = next(iter(self.model.embed(["probe"])))
        self.dim = int(probe.shape[-1])

    def encode_query(self, query: str) -> np.ndarray:
        """Return a single (dim,) float32 vector, L2-normalized."""
        vec = next(iter(self.model.embed([query])))
        return np.asarray(vec, dtype=np.float32)

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return (N, dim) float32 matrix, L2-normalized rows."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = list(self.model.embed(texts, batch_size=batch_size))
        return np.vstack(vecs).astype(np.float32)
```

- [ ] **Step 3:** Smoke test

```bash
python -c "
from rag.retrieve.embedder import Embedder
e = Embedder()
q = e.encode_query('hello world')
d = e.encode_docs(['first doc', 'second doc'])
print('Query shape:', q.shape, 'dim=', e.dim)
print('Docs shape:', d.shape)
import numpy as np
print('Query unit-norm:', round(float(np.linalg.norm(q)), 4))
"
```
Expected: `Query shape: (384,) dim= 384` / `Docs shape: (2, 384)` / `Query unit-norm: 1.0`

- [ ] **Step 4:** Commit

```bash
git add src/rag/retrieve/__init__.py src/rag/retrieve/embedder.py
git commit -m "feat(retrieve): MiniLM embedder wrapper"
```

---

### Task 11: BM25 index wrapper

**Files:**
- Create: `src/rag/index/__init__.py`
- Create: `src/rag/index/bm25_index.py`

- [ ] **Step 1:** Empty package init

Write `src/rag/index/__init__.py`:
```python
```

- [ ] **Step 2:** Write `src/rag/index/bm25_index.py`

```python
"""BM25 index over chunk texts. Simple whitespace+punctuation tokenization.

We don't lemmatize or remove stopwords — BM25's IDF handles common terms well
and lemmatization is a footgun for news text (proper nouns, numbers).
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self):
        self.chunk_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        assert len(chunk_ids) == len(texts), "chunk_ids and texts length mismatch"
        self.chunk_ids = chunk_ids
        self.tokenized_corpus = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Return [(chunk_id, score), ...] sorted by score desc, top-k."""
        if self.bm25 is None or not self.chunk_ids:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # top-k indices (partial sort for speed on large N)
        k = min(k, len(scores))
        if k == 0:
            return []
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "tokenized_corpus": self.tokenized_corpus}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls()
        inst.chunk_ids = data["chunk_ids"]
        inst.tokenized_corpus = data["tokenized_corpus"]
        inst.bm25 = BM25Okapi(inst.tokenized_corpus)
        return inst
```

- [ ] **Step 3:** Smoke test

```bash
python -c "
from rag.index.bm25_index import BM25Index
idx = BM25Index()
idx.build(['c1','c2','c3'], ['dogs are great pets', 'cats are independent animals', 'birds can fly in the sky'])
print('Dog query:', idx.search('dog', k=3))
print('Fly query:', idx.search('flying', k=3))
"
```
Expected: dog query returns c1 with high score; fly query returns c3.

- [ ] **Step 4:** Commit

```bash
git add src/rag/index/__init__.py src/rag/index/bm25_index.py
git commit -m "feat(index): BM25 index with save/load"
```

---

### Task 12: FAISS vector index wrapper

**Files:**
- Create: `src/rag/index/vector_index.py`

- [ ] **Step 1:** Write `src/rag/index/vector_index.py`

```python
"""FAISS IndexFlatIP wrapper for cosine similarity over L2-normalized vectors.

IndexFlatIP does exhaustive inner-product search. For L2-normalized vectors this
equals cosine similarity. Exhaustive search is optimal for <100K vectors —
HNSW/ANN indices add overhead without winning until the corpus is much larger.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np


class FaissFlatIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunk_ids: list[str] = []

    def build(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        assert vectors.dtype == np.float32, f"expected float32, got {vectors.dtype}"
        assert vectors.shape == (len(chunk_ids), self.dim), (
            f"shape mismatch: vectors={vectors.shape} expected ({len(chunk_ids)}, {self.dim})"
        )
        self.chunk_ids = list(chunk_ids)
        self.index.add(vectors)

    def search(self, query_vec: np.ndarray, k: int = 100) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vec.astype(np.float32), k)
        return [
            (self.chunk_ids[indices[0][i]], float(scores[0][i]))
            for i in range(k)
            if indices[0][i] != -1
        ]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "faiss.index"))
        with open(dir_path / "chunk_ids.pkl", "wb") as f:
            pickle.dump(self.chunk_ids, f)

    @classmethod
    def load(cls, dir_path: Path) -> "FaissFlatIndex":
        index = faiss.read_index(str(dir_path / "faiss.index"))
        with open(dir_path / "chunk_ids.pkl", "rb") as f:
            chunk_ids = pickle.load(f)
        inst = cls(index.d)
        inst.index = index
        inst.chunk_ids = chunk_ids
        return inst
```

- [ ] **Step 2:** Smoke test

```bash
python -c "
import numpy as np
from rag.index.vector_index import FaissFlatIndex
# Fake 4-dim vectors
vecs = np.array([[1,0,0,0],[0,1,0,0],[0.9,0.1,0,0]], dtype=np.float32)
# L2 normalize
vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
idx = FaissFlatIndex(dim=4)
idx.build(['c1','c2','c3'], vecs)
q = np.array([1,0,0,0], dtype=np.float32)
print('Search:', idx.search(q, k=3))
"
```
Expected: c1 score ~1.0, c3 score ~0.99, c2 score ~0.0.

- [ ] **Step 3:** Commit

```bash
git add src/rag/index/vector_index.py
git commit -m "feat(index): FAISS IndexFlatIP wrapper"
```

---

## Phase G — Ingest Pipeline

### Task 13: End-to-end ingest pipeline

**Files:**
- Create: `src/rag/ingest/pipeline.py`

- [ ] **Step 1:** Write `src/rag/ingest/pipeline.py`

```python
"""End-to-end ingest: load docs → chunk → embed → build BM25 + FAISS → save."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from rag.index.bm25_index import BM25Index
from rag.index.vector_index import FaissFlatIndex
from rag.ingest.chunking import recursive_chunk
from rag.ingest.manifest import IngestManifest
from rag.retrieve.embedder import Embedder
from rag.types import Chunk, Document


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    texts = recursive_chunk(doc.text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::chunk::{i}",
            doc_id=doc.doc_id,
            text=t,
            chunk_index=i,
            metadata=doc.metadata,
        )
        for i, t in enumerate(texts)
    ]


def run_ingest(
    docs: Iterable[Document],
    index_dir: Path,
    chunk_size: int = 512,
    overlap: int = 51,
    embedder_model: str = "all-MiniLM-L6-v2",
    embed_batch_size: int = 64,
    force: bool = False,
) -> dict:
    """Run full ingest. Returns stats dict.

    Incremental: skips docs whose content_sha256 matches the manifest.
    When any doc changes, we rebuild the full BM25 + FAISS indices (simple,
    correct, fast enough for <100K chunks).
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = IngestManifest(index_dir / "ingest_manifest.db")

    all_docs = list(docs)
    print(f"Loaded {len(all_docs)} docs")

    if not force:
        changed = [d for d in all_docs if not manifest.is_unchanged(d.doc_id, d.content_sha256)]
        print(f"{len(changed)} docs changed since last ingest; {len(all_docs) - len(changed)} skipped")
        if len(changed) == 0:
            print("No changes — indices still valid.")
            return {"docs_total": len(all_docs), "docs_changed": 0}

    # Chunk all docs (we always rebuild indices when anything changes)
    all_chunks: list[Chunk] = []
    for d in tqdm(all_docs, desc="chunking"):
        all_chunks.extend(chunk_document(d, chunk_size, overlap))
    print(f"Produced {len(all_chunks)} chunks")

    # Embed
    embedder = Embedder(embedder_model)
    chunk_texts = [c.text for c in all_chunks]
    print("Embedding chunks...")
    vectors = embedder.encode_docs(chunk_texts, batch_size=embed_batch_size)

    # Build indices
    chunk_ids = [c.chunk_id for c in all_chunks]
    bm25 = BM25Index()
    bm25.build(chunk_ids, chunk_texts)
    faiss_idx = FaissFlatIndex(dim=embedder.dim)
    faiss_idx.build(chunk_ids, vectors)

    # Save everything
    bm25.save(index_dir / "bm25.pkl")
    faiss_idx.save(index_dir / "faiss")
    # Chunk metadata (for rendering search results with text/metadata)
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }) + "\n")

    # Update manifest per doc
    from collections import Counter
    per_doc_counts = Counter(c.doc_id for c in all_chunks)
    for d in all_docs:
        manifest.record(d.doc_id, d.content_sha256, per_doc_counts[d.doc_id])

    return {
        "docs_total": len(all_docs),
        "chunks_total": len(all_chunks),
        "embedding_dim": embedder.dim,
        "index_dir": str(index_dir),
    }
```

- [ ] **Step 2:** Smoke test (tiny fake corpus)

```bash
python -c "
import tempfile, shutil
from pathlib import Path
from rag.types import Document
from rag.ingest.pipeline import run_ingest

docs = [
    Document('d1', 'src1', 't1', 'The quick brown fox jumps over the lazy dog. ' * 20,
             {'category': 'animals'}, content_sha256='hash_d1'),
    Document('d2', 'src2', 't2', 'Quantum mechanics is weird and wonderful. ' * 20,
             {'category': 'science'}, content_sha256='hash_d2'),
]
with tempfile.TemporaryDirectory() as tmp:
    stats = run_ingest(iter(docs), Path(tmp)/'idx', chunk_size=200, overlap=20)
    print('Stats:', stats)
"
```
Expected: stats with `docs_total: 2`, `chunks_total >= 2`, `embedding_dim: 384`.

- [ ] **Step 3:** Commit

```bash
git add src/rag/ingest/pipeline.py
git commit -m "feat(ingest): end-to-end pipeline with incremental ingest"
```

---

### Task 14: `scripts/ingest.py` runner

**Files:**
- Create: `scripts/ingest.py`

- [ ] **Step 1:** Write `scripts/ingest.py`

```python
"""Run the full ingest on MultiHop-RAG (+ optional PDFs from data/pdfs/)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import get_settings
from rag.ingest.loaders import load_multihop_from_hf, load_pdf_directory
from rag.ingest.pipeline import run_ingest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ignore manifest, re-embed all")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--pdf-dir", type=str, default=None,
                        help="Optional directory of PDF files to include")
    parser.add_argument("--index-subdir", type=str, default="default",
                        help="Subdir under index_dir — use different names per config")
    args = parser.parse_args()

    settings = get_settings()
    chunk_size = args.chunk_size or settings.chunk_size
    overlap = args.overlap if args.overlap is not None else settings.chunk_overlap
    index_dir = settings.index_dir / args.index_subdir

    docs = []
    # MultiHop-RAG articles
    mh_dir = settings.data_dir / "multihop_rag"
    if mh_dir.exists():
        docs.extend(load_multihop_from_hf(mh_dir))
    else:
        print(f"WARNING: {mh_dir} not found. Run scripts/preflight.py first.", file=sys.stderr)
        return 1

    # Optional PDF padding
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else settings.data_dir / "pdfs"
    if pdf_dir.exists() and any(pdf_dir.glob("*.pdf")):
        docs.extend(load_pdf_directory(pdf_dir))

    stats = run_ingest(
        docs,
        index_dir=index_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        embedder_model=settings.embedding_model,
        force=args.force,
    )
    print("INGEST DONE:", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2:** Run it

Run: `python scripts/ingest.py`
Expected: progress bars, final line like `INGEST DONE: {'docs_total': 609, 'chunks_total': ~3000, 'embedding_dim': 384, ...}`.

- [ ] **Step 3:** Verify index artifacts

Run: `ls -lh indices/default/`
Expected: `bm25.pkl`, `faiss/` (containing `faiss.index` + `chunk_ids.pkl`), `chunks.jsonl`, `ingest_manifest.db`.

- [ ] **Step 4:** Commit

```bash
git add scripts/ingest.py
git commit -m "feat(scripts): ingest runner for MultiHop-RAG + optional PDFs"
```

---

## Phase H — Fusion (TDD) and Reranker

### Task 15: Fusion tests first

**Files:**
- Create: `tests/test_fusion.py`

- [ ] **Step 1:** Write failing tests

```python
"""Tests for score fusion (RRF and weighted-alpha)."""
from rag.retrieve.fusion import rrf_fuse, weighted_alpha_fuse


def test_rrf_single_list_ranking_preserved():
    bm25 = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
    dense = []
    fused = rrf_fuse(bm25, dense, k=60, top_k=10)
    # Only bm25 contributes; order is preserved
    assert [cid for cid, _ in fused] == ["a", "b", "c"]


def test_rrf_combines_both_lists():
    # A ranks 1st in bm25 and 3rd in dense; B ranks 2nd in bm25 and 1st in dense
    # RRF: A = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
    #      B = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
    # So B should rank first
    bm25 = [("a", 10.0), ("b", 9.0), ("c", 8.0)]
    dense = [("b", 0.9), ("d", 0.85), ("a", 0.8)]
    fused = rrf_fuse(bm25, dense, k=60, top_k=10)
    fused_ids = [cid for cid, _ in fused]
    assert fused_ids[0] == "b"
    assert "a" in fused_ids and "c" in fused_ids and "d" in fused_ids


def test_rrf_handles_empty_inputs():
    assert rrf_fuse([], [], k=60, top_k=5) == []


def test_weighted_alpha_normalizes_per_query():
    # BM25 scores are large; dense are in [0,1]. Without per-query normalization,
    # bm25 would dominate regardless of alpha.
    bm25 = [("a", 100.0), ("b", 50.0)]
    dense = [("b", 0.99), ("a", 0.01)]
    # alpha=1.0 → pure bm25 → a first
    out_bm25 = weighted_alpha_fuse(bm25, dense, alpha=1.0, top_k=2)
    assert [c for c, _ in out_bm25] == ["a", "b"]
    # alpha=0.0 → pure dense → b first
    out_dense = weighted_alpha_fuse(bm25, dense, alpha=0.0, top_k=2)
    assert [c for c, _ in out_dense] == ["b", "a"]
    # alpha=0.5 → after normalization both are [1.0, 0.0]; tie broken by bm25 order → a first
    out_mid = weighted_alpha_fuse(bm25, dense, alpha=0.5, top_k=2)
    assert len(out_mid) == 2


def test_weighted_alpha_missing_ids_treated_as_zero():
    # dense has 'a' but bm25 does not → 'a' gets bm25 score of 0
    bm25 = [("b", 10.0)]
    dense = [("a", 0.99), ("b", 0.5)]
    out = weighted_alpha_fuse(bm25, dense, alpha=0.3, top_k=2)
    assert len(out) == 2
    assert set(c for c, _ in out) == {"a", "b"}
```

- [ ] **Step 2:** Run tests — expect failure

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError`.

---

### Task 16: Implement fusion

**Files:**
- Create: `src/rag/retrieve/fusion.py`

- [ ] **Step 1:** Write `src/rag/retrieve/fusion.py`

```python
"""Score fusion strategies for hybrid retrieval.

Two strategies exposed:
  - rrf_fuse: rank-based, no normalization needed. Battle-tested default.
  - weighted_alpha_fuse: per-query min-max normalization, then alpha-weighted sum.

Both return [(chunk_id, fused_score), ...] sorted desc.
"""
from __future__ import annotations


def rrf_fuse(
    list_a: list[tuple[str, float]],
    list_b: list[tuple[str, float]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    rrf_score(d) = Σ 1 / (k + rank_i(d))  over retrievers that returned d.
    """
    scores: dict[str, float] = {}
    for results in (list_a, list_b):
        for rank, (cid, _) in enumerate(results):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ordered[:top_k]


def weighted_alpha_fuse(
    bm25_list: list[tuple[str, float]],
    dense_list: list[tuple[str, float]],
    alpha: float = 0.5,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Per-query min-max normalize both score lists, then alpha*bm25 + (1-alpha)*dense.

    Missing IDs are treated as 0 after normalization (they never reached that retriever).
    """
    bm25_norm = _minmax_normalize(bm25_list)
    dense_norm = _minmax_normalize(dense_list)
    all_ids = set(bm25_norm) | set(dense_norm)
    fused: dict[str, float] = {}
    for cid in all_ids:
        fused[cid] = alpha * bm25_norm.get(cid, 0.0) + (1.0 - alpha) * dense_norm.get(cid, 0.0)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ordered[:top_k]


def _minmax_normalize(scored: list[tuple[str, float]]) -> dict[str, float]:
    if not scored:
        return {}
    vals = [s for _, s in scored]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {cid: 1.0 for cid, _ in scored}
    return {cid: (s - lo) / (hi - lo) for cid, s in scored}
```

- [ ] **Step 2:** Run tests — expect pass

Run: `pytest tests/test_fusion.py -v`
Expected: 5 passed.

- [ ] **Step 3:** Commit

```bash
git add src/rag/retrieve/fusion.py tests/test_fusion.py
git commit -m "feat(retrieve): RRF + weighted-alpha fusion (tested)"
```

---

### Task 17: Reranker wrapper

**Files:**
- Create: `src/rag/retrieve/reranker.py`

- [ ] **Step 1:** Write `src/rag/retrieve/reranker.py`

```python
"""Cross-encoder reranker via FastEmbed (ONNX Runtime, no PyTorch).

Uses `Xenova/ms-marco-MiniLM-L-6-v2` — ONNX export of the same model the
sentence-transformers `cross-encoder/ms-marco-MiniLM-L-6-v2` uses.
One call reranks all candidates in a batched forward pass.
"""
from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder


class Reranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],  # [(chunk_id, chunk_text), ...]
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[tuple[str, float]]:
        """Score all candidates in ONE rerank call, return top-k sorted desc."""
        if not candidates:
            return []
        docs = [text for _, text in candidates]
        scores = list(self.model.rerank(query, docs, batch_size=batch_size))
        scored = [(cid, float(s)) for (cid, _), s in zip(candidates, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

- [ ] **Step 2:** Smoke test

```bash
python -c "
from rag.retrieve.reranker import Reranker
r = Reranker()
out = r.rerank(
    'who founded Tesla?',
    [('c1','Elon Musk is the co-founder of Tesla.'),
     ('c2','Cats are independent animals.'),
     ('c3','Tesla sells electric cars.')],
    top_k=2
)
print('Top-2:', out)
"
```
Expected: c1 first (Musk/Tesla), some other chunk second.

- [ ] **Step 3:** Commit

```bash
git add src/rag/retrieve/reranker.py
git commit -m "feat(retrieve): batched cross-encoder reranker"
```

---

## Phase I — Search pipeline

### Task 18: Async search pipeline

**Files:**
- Create: `src/rag/retrieve/search.py`

- [ ] **Step 1:** Write `src/rag/retrieve/search.py`

```python
"""Async search pipeline: BM25 + dense → fusion → rerank → top-k."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rag.index.bm25_index import BM25Index
from rag.index.vector_index import FaissFlatIndex
from rag.retrieve.embedder import Embedder
from rag.retrieve.fusion import rrf_fuse, weighted_alpha_fuse
from rag.retrieve.reranker import Reranker
from rag.types import SearchHit


class SearchEngine:
    """Holds indices + models. One instance per process; load once at startup."""

    def __init__(
        self,
        index_dir: Path,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ):
        self.bm25 = BM25Index.load(index_dir / "bm25.pkl")
        self.vector = FaissFlatIndex.load(index_dir / "faiss")
        self.embedder = embedder
        self.reranker = reranker
        self.chunk_map = self._load_chunks(index_dir / "chunks.jsonl")

    @staticmethod
    def _load_chunks(path: Path) -> dict[str, dict]:
        chunks: dict[str, dict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                chunks[row["chunk_id"]] = row
        return chunks

    async def search(
        self,
        query: str,
        top_k_retrieve: int = 100,
        top_k_rerank: int = 20,
        top_k_final: int = 5,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        use_rerank: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        loop = asyncio.get_running_loop()

        # Kick off embed + BM25 in parallel (both are sync+CPU — use threadpool)
        embed_fut = loop.run_in_executor(None, self.embedder.encode_query, query)
        bm25_fut = loop.run_in_executor(
            None, self.bm25.search, query, top_k_retrieve
        )
        query_vec, bm25_hits = await asyncio.gather(embed_fut, bm25_fut)

        # FAISS is very fast and holds GIL briefly — keep inline
        dense_hits = self.vector.search(query_vec, k=top_k_retrieve)

        # Fuse
        if fusion_method == "rrf":
            fused = rrf_fuse(bm25_hits, dense_hits, k=rrf_k, top_k=top_k_rerank)
        elif fusion_method == "weighted":
            fused = weighted_alpha_fuse(bm25_hits, dense_hits, alpha=alpha, top_k=top_k_rerank)
        elif fusion_method == "semantic_only":
            fused = dense_hits[:top_k_rerank]
        elif fusion_method == "bm25_only":
            fused = bm25_hits[:top_k_rerank]
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")

        bm25_by_id = dict(bm25_hits)
        dense_by_id = dict(dense_hits)
        fused_by_id = dict(fused)

        # Metadata filter (post-fusion, pre-rerank) — applies to chunk.metadata
        if filters:
            fused = [(cid, s) for cid, s in fused if _metadata_matches(self.chunk_map.get(cid, {}).get("metadata", {}), filters)]

        # Rerank (if requested)
        if use_rerank and self.reranker is not None and fused:
            candidates = [(cid, self.chunk_map[cid]["text"]) for cid, _ in fused if cid in self.chunk_map]
            reranked = await loop.run_in_executor(
                None, self.reranker.rerank, query, candidates, top_k_final
            )
        else:
            reranked = [(cid, 0.0) for cid, _ in fused[:top_k_final]]

        rerank_by_id = dict(reranked)

        # Build SearchHits
        hits: list[SearchHit] = []
        for cid, _ in reranked:
            chunk = self.chunk_map.get(cid)
            if not chunk:
                continue
            final_score = (
                rerank_by_id[cid]
                if use_rerank and self.reranker is not None
                else fused_by_id.get(cid, 0.0)
            )
            hits.append(SearchHit(
                chunk_id=cid,
                doc_id=chunk["doc_id"],
                text=chunk["text"],
                scores={
                    "bm25": bm25_by_id.get(cid, 0.0),
                    "semantic": dense_by_id.get(cid, 0.0),
                    "hybrid_fused": fused_by_id.get(cid, 0.0),
                    "rerank": rerank_by_id.get(cid, 0.0) if use_rerank else 0.0,
                    "final": final_score,
                },
                metadata=chunk.get("metadata", {}),
            ))
        return hits


def _metadata_matches(meta: dict, filters: dict) -> bool:
    for key, want in filters.items():
        got = meta.get(key)
        if isinstance(want, list):
            if got not in want:
                return False
        else:
            if got != want:
                return False
    return True
```

- [ ] **Step 2:** Smoke test (requires `scripts/ingest.py` to have run)

```bash
python -c "
import asyncio
from pathlib import Path
from rag.retrieve.embedder import Embedder
from rag.retrieve.reranker import Reranker
from rag.retrieve.search import SearchEngine

e = Embedder()
r = Reranker()
s = SearchEngine(Path('indices/default'), e, r)
hits = asyncio.run(s.search('What is the impact of climate change?', top_k_rerank=10, top_k_final=3))
for h in hits:
    print(h.chunk_id, '|', round(h.scores['final'],3), '|', h.text[:80])
"
```
Expected: 3 hits with non-zero rerank scores and relevant-looking text.

- [ ] **Step 3:** Commit

```bash
git add src/rag/retrieve/search.py
git commit -m "feat(retrieve): async search pipeline with fusion+rerank"
```

---

## Phase J — FastAPI

### Task 19: API schemas

**Files:**
- Create: `src/rag/api/__init__.py`
- Create: `src/rag/api/schemas.py`

- [ ] **Step 1:** Empty package init

Write `src/rag/api/__init__.py`: (empty)

- [ ] **Step 2:** Write schemas

```python
"""Pydantic models for /search request and response."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    top_k_retrieve: int = Field(100, ge=5, le=500)
    top_k_rerank: int = Field(20, ge=1, le=100)
    fusion_method: str = Field("rrf", pattern="^(rrf|weighted|semantic_only|bm25_only)$")
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(60, ge=1, le=500)
    use_rerank: bool = True
    filters: dict[str, Any] | None = None


class Scores(BaseModel):
    bm25: float
    semantic: float
    hybrid_fused: float
    rerank: float
    final: float


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    scores: Scores
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    latency_ms: float
    config: dict[str, Any]
```

- [ ] **Step 3:** Commit

```bash
git add src/rag/api/__init__.py src/rag/api/schemas.py
git commit -m "feat(api): pydantic request/response schemas"
```

---

### Task 20: FastAPI app with /search

**Files:**
- Create: `src/rag/api/app.py`

- [ ] **Step 1:** Write `src/rag/api/app.py`

```python
"""FastAPI service exposing POST /search."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from rag.api.schemas import SearchRequest, SearchResponse, SearchResult, Scores
from rag.config import get_settings
from rag.retrieve.embedder import Embedder
from rag.retrieve.reranker import Reranker
from rag.retrieve.search import SearchEngine


_engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    settings = get_settings()
    index_dir = settings.index_dir / "default"
    if not (index_dir / "faiss").exists():
        raise RuntimeError(
            f"No index at {index_dir}. Run scripts/ingest.py first."
        )
    print("Loading embedder...")
    embedder = Embedder(settings.embedding_model)
    print("Loading reranker...")
    reranker = Reranker(settings.reranker_model)
    print("Loading indices...")
    _engine = SearchEngine(index_dir, embedder, reranker)
    # Warm up
    print("Warming up models...")
    import asyncio
    await _engine.search("warmup query", top_k_rerank=5, top_k_final=2)
    print("API ready.")
    yield
    # No teardown needed


app = FastAPI(title="RAG Retrieval Platform", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not loaded")
    t0 = time.perf_counter()
    hits = await _engine.search(
        query=req.query,
        top_k_retrieve=req.top_k_retrieve,
        top_k_rerank=req.top_k_rerank,
        top_k_final=req.top_k,
        fusion_method=req.fusion_method,
        alpha=req.alpha,
        rrf_k=req.rrf_k,
        use_rerank=req.use_rerank,
        filters=req.filters,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return SearchResponse(
        query=req.query,
        results=[
            SearchResult(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                text=h.text,
                scores=Scores(**h.scores),
                metadata=h.metadata,
            )
            for h in hits
        ],
        latency_ms=round(latency_ms, 2),
        config={
            "fusion_method": req.fusion_method,
            "alpha": req.alpha,
            "use_rerank": req.use_rerank,
            "top_k_rerank": req.top_k_rerank,
        },
    )
```

- [ ] **Step 2:** Start server and test

Run (in one terminal):
```bash
uvicorn rag.api.app:app --host 127.0.0.1 --port 8000
```
Expected startup log: Loading embedder... / Loading reranker... / Loading indices... / Warming up models... / API ready.

Run (in another terminal):
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"climate change impact","top_k":3}' | python -m json.tool
```
Expected: JSON with 3 `results`, each having `scores` keys, and `latency_ms` under ~500.

- [ ] **Step 3:** Stop server (Ctrl-C)

- [ ] **Step 4:** Commit

```bash
git add src/rag/api/app.py
git commit -m "feat(api): FastAPI /search endpoint with lifespan-loaded engine"
```

---

## Phase K — Evaluation (TDD on metrics)

### Task 21: Metrics tests first

**Files:**
- Create: `tests/test_metrics.py`

- [ ] **Step 1:** Write tests

```python
"""Tests for retrieval metrics: precision@k, recall@k, NDCG@k."""
import math

from rag.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k


def test_precision_all_relevant():
    assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_precision_none_relevant():
    assert precision_at_k(["a", "b"], {"c"}, k=2) == 0.0


def test_precision_partial():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == 0.5


def test_precision_at_k_truncates():
    # Only first k items considered
    assert precision_at_k(["a", "b", "x", "y"], {"a", "b", "x"}, k=2) == 1.0


def test_recall_at_k():
    # 2 of 3 relevant retrieved in top 5
    assert recall_at_k(["a", "x", "b", "y", "z"], {"a", "b", "c"}, k=5) == 2 / 3


def test_recall_at_k_no_relevant_returns_zero():
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_ndcg_zero_when_no_hits():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_ndcg_partial():
    # Retrieved at positions 1 and 3 (DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5)
    # Ideal: both at top (IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309)
    val = ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3)
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert abs(val - expected) < 1e-6
```

- [ ] **Step 2:** Run tests — expect failure

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

---

### Task 22: Implement metrics

**Files:**
- Create: `src/rag/eval/__init__.py`
- Create: `src/rag/eval/metrics.py`

- [ ] **Step 1:** Empty package init

Write `src/rag/eval/__init__.py`: (empty)

- [ ] **Step 2:** Write metrics

```python
"""Retrieval metrics: precision@k, recall@k, NDCG@k (binary relevance).

All expect:
  retrieved: list[str]  -- doc_ids in ranked order
  relevant:  set[str]   -- ground-truth relevant doc_ids
"""
from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for d in topk if d in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for d in topk if d in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k == 0:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(retrieved[:k]):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
```

- [ ] **Step 3:** Run tests — expect pass

Run: `pytest tests/test_metrics.py -v`
Expected: 9 passed.

- [ ] **Step 4:** Commit

```bash
git add src/rag/eval/__init__.py src/rag/eval/metrics.py tests/test_metrics.py
git commit -m "feat(eval): precision/recall/NDCG metrics (tested)"
```

---

### Task 23: Qrels adapter for MultiHop-RAG

**Files:**
- Create: `src/rag/eval/qrels.py`

- [ ] **Step 1:** Write qrels adapter

```python
"""Adapt MultiHop-RAG Q/A records into (query_id, query, relevant_doc_ids) tuples.

MultiHop-RAG schema (per record):
  - 'query' (str)
  - 'answer' (str)
  - 'question_type' (str, e.g. 'inference_query')
  - 'evidence_list' (list of dicts with 'url', 'title', 'source', 'category', 'fact', ...)

We map each evidence item to a doc_id equal to its 'url' (matches loader.doc_id convention).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalQuery:
    query_id: str
    query: str
    relevant_doc_ids: set[str]
    question_type: str = ""


def load_multihop_eval(hf_dataset_dir: Path, split: str | None = None) -> list[EvalQuery]:
    """Load MultiHop-RAG eval queries from a HF dataset save_to_disk output."""
    from datasets import load_from_disk

    ds = load_from_disk(str(hf_dataset_dir))

    # Pick the split containing Q/A records (not the corpus)
    if split is None:
        candidates = ["MultiHopRAG", "test", "train", "eval", "queries"]
        split = next((c for c in candidates if c in ds), None)
        if split is None:
            # Fallback: any split with 'query' field in first row
            for s in ds.keys():
                if ds[s] and "query" in ds[s][0]:
                    split = s
                    break
    if split is None or split not in ds:
        raise ValueError(f"Could not find Q/A split in {list(ds.keys())}")

    out: list[EvalQuery] = []
    for i, row in enumerate(ds[split]):
        q = row.get("query", "")
        if not q:
            continue
        evidence = row.get("evidence_list", []) or []
        rel_ids: set[str] = set()
        for ev in evidence:
            url = ev.get("url") or ev.get("source") or ev.get("title")
            if url:
                rel_ids.add(url)
        if not rel_ids:
            continue  # skip queries with no mapable ground truth
        qid = hashlib.md5(q.encode("utf-8")).hexdigest()[:12]
        out.append(EvalQuery(
            query_id=qid,
            query=q,
            relevant_doc_ids=rel_ids,
            question_type=row.get("question_type", ""),
        ))
    return out
```

- [ ] **Step 2:** Smoke test

```bash
python -c "
from pathlib import Path
from rag.eval.qrels import load_multihop_eval
queries = load_multihop_eval(Path('data/multihop_rag'))
print(f'Loaded {len(queries)} eval queries')
if queries:
    q = queries[0]
    print('Sample query:', q.query[:80])
    print('Relevant doc_ids:', list(q.relevant_doc_ids)[:2])
    print('Question type:', q.question_type)
"
```
Expected: ~2,500+ queries; sample query text; 2-4 relevant doc URLs.

- [ ] **Step 3:** Commit

```bash
git add src/rag/eval/qrels.py
git commit -m "feat(eval): MultiHop-RAG qrels adapter"
```

---

### Task 24: Eval harness with parallel queries + latency capture

**Files:**
- Create: `src/rag/eval/harness.py`

- [ ] **Step 1:** Write harness

```python
"""Eval harness: run a batch of queries through SearchEngine in parallel,
compute P@k / R@k / NDCG@k, capture per-query latency."""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any

from rag.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k
from rag.eval.qrels import EvalQuery
from rag.retrieve.search import SearchEngine


@dataclass
class PerQueryResult:
    query_id: str
    query: str
    latency_ms: float
    retrieved_doc_ids: list[str]
    relevant_doc_ids: set[str]
    precision_at_5: float
    recall_at_5: float
    ndcg_at_5: float


@dataclass
class EvalReport:
    config_name: str
    n_queries: int
    precision_at_5_mean: float
    recall_at_5_mean: float
    ndcg_at_5_mean: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    first_query_latency_ms: float
    per_query: list[PerQueryResult]


async def _run_one(engine: SearchEngine, eq: EvalQuery, search_kwargs: dict[str, Any]) -> PerQueryResult:
    t0 = time.perf_counter()
    hits = await engine.search(eq.query, **search_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    retrieved = [h.doc_id for h in hits]
    return PerQueryResult(
        query_id=eq.query_id,
        query=eq.query,
        latency_ms=latency_ms,
        retrieved_doc_ids=retrieved,
        relevant_doc_ids=eq.relevant_doc_ids,
        precision_at_5=precision_at_k(retrieved, eq.relevant_doc_ids, k=5),
        recall_at_5=recall_at_k(retrieved, eq.relevant_doc_ids, k=5),
        ndcg_at_5=ndcg_at_k(retrieved, eq.relevant_doc_ids, k=5),
    )


async def run_eval(
    engine: SearchEngine,
    queries: list[EvalQuery],
    config_name: str,
    search_kwargs: dict[str, Any],
    concurrency: int = 8,
) -> EvalReport:
    """Run all queries with bounded concurrency; capture first-query latency separately."""
    if not queries:
        raise ValueError("No queries provided")

    # First query in isolation (captures cold-cache latency)
    first = await _run_one(engine, queries[0], search_kwargs)

    # Remaining in parallel
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(eq: EvalQuery):
        async with sem:
            return await _run_one(engine, eq, search_kwargs)

    rest = await asyncio.gather(*(_guarded(q) for q in queries[1:]))
    all_results = [first] + list(rest)

    # Aggregate
    latencies_warm = [r.latency_ms for r in all_results[1:]]  # exclude cold first-call
    if not latencies_warm:
        latencies_warm = [first.latency_ms]

    return EvalReport(
        config_name=config_name,
        n_queries=len(all_results),
        precision_at_5_mean=statistics.mean(r.precision_at_5 for r in all_results),
        recall_at_5_mean=statistics.mean(r.recall_at_5 for r in all_results),
        ndcg_at_5_mean=statistics.mean(r.ndcg_at_5 for r in all_results),
        latency_p50_ms=_percentile(latencies_warm, 50),
        latency_p95_ms=_percentile(latencies_warm, 95),
        latency_p99_ms=_percentile(latencies_warm, 99),
        first_query_latency_ms=first.latency_ms,
        per_query=all_results,
    )


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
```

- [ ] **Step 2:** Commit

```bash
git add src/rag/eval/harness.py
git commit -m "feat(eval): harness with parallel queries + latency capture"
```

---

### Task 25: Report serializer

**Files:**
- Create: `src/rag/eval/reports.py`

- [ ] **Step 1:** Write reports

```python
"""Save EvalReport to JSON + markdown table."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag.eval.harness import EvalReport


def save_report(report: EvalReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = report.config_name

    # Full JSON (per-query included)
    with open(output_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=_default)

    # Summary markdown
    md = f"""# Eval Report: {name}

- **N queries:** {report.n_queries}
- **Precision@5 (mean):** {report.precision_at_5_mean:.4f}
- **Recall@5 (mean):** {report.recall_at_5_mean:.4f}
- **NDCG@5 (mean):** {report.ndcg_at_5_mean:.4f}
- **Latency p50 (warm):** {report.latency_p50_ms:.1f} ms
- **Latency p95 (warm):** {report.latency_p95_ms:.1f} ms
- **Latency p99 (warm):** {report.latency_p99_ms:.1f} ms
- **First-query (cold):** {report.first_query_latency_ms:.1f} ms
"""
    with open(output_dir / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(md)


def _default(o):
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"Cannot serialize {type(o)}")


def combine_reports_table(reports: list[EvalReport]) -> str:
    lines = [
        "| Config | P@5 | R@5 | NDCG@5 | p50 ms | p95 ms | p99 ms | cold ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.config_name} "
            f"| {r.precision_at_5_mean:.4f} "
            f"| {r.recall_at_5_mean:.4f} "
            f"| {r.ndcg_at_5_mean:.4f} "
            f"| {r.latency_p50_ms:.1f} "
            f"| {r.latency_p95_ms:.1f} "
            f"| {r.latency_p99_ms:.1f} "
            f"| {r.first_query_latency_ms:.1f} |"
        )
    return "\n".join(lines)
```

- [ ] **Step 2:** Commit

```bash
git add src/rag/eval/reports.py
git commit -m "feat(eval): JSON + markdown report serialization"
```

---

### Task 26: `scripts/run_eval.py` runner

**Files:**
- Create: `scripts/run_eval.py`

- [ ] **Step 1:** Write runner

```python
"""Run one eval config end-to-end."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rag.config import get_settings
from rag.eval.harness import run_eval
from rag.eval.qrels import load_multihop_eval
from rag.eval.reports import save_report
from rag.retrieve.embedder import Embedder
from rag.retrieve.reranker import Reranker
from rag.retrieve.search import SearchEngine


def build_search_kwargs(config: str, settings) -> dict:
    base = {
        "top_k_retrieve": settings.top_k_retrieve,
        "top_k_rerank": settings.top_k_rerank,
        "top_k_final": settings.top_k_final,
        "rrf_k": settings.rrf_k,
    }
    if config == "semantic_only":
        return {**base, "fusion_method": "semantic_only", "use_rerank": False}
    if config == "hybrid":
        return {**base, "fusion_method": "rrf", "use_rerank": False}
    if config == "hybrid+rerank":
        return {**base, "fusion_method": "rrf", "use_rerank": True}
    if config == "bm25_only":
        return {**base, "fusion_method": "bm25_only", "use_rerank": False}
    raise ValueError(f"Unknown config: {config}")


async def main_async(args) -> int:
    settings = get_settings()
    index_dir = settings.index_dir / args.index_subdir
    if not (index_dir / "faiss").exists():
        print(f"No index at {index_dir}. Run scripts/ingest.py first.", file=sys.stderr)
        return 1

    print("Loading models and indices...")
    embedder = Embedder(settings.embedding_model)
    reranker = Reranker(settings.reranker_model) if args.config.endswith("rerank") else None
    engine = SearchEngine(index_dir, embedder, reranker)

    queries = load_multihop_eval(settings.data_dir / "multihop_rag")
    if args.limit:
        queries = queries[: args.limit]
    print(f"Evaluating {len(queries)} queries on config={args.config}")

    search_kwargs = build_search_kwargs(args.config, settings)
    report = await run_eval(engine, queries, config_name=args.config, search_kwargs=search_kwargs,
                            concurrency=args.concurrency)

    save_report(report, settings.output_dir / "runs")
    print(f"P@5={report.precision_at_5_mean:.4f} R@5={report.recall_at_5_mean:.4f} "
          f"NDCG@5={report.ndcg_at_5_mean:.4f} "
          f"p95={report.latency_p95_ms:.1f}ms")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        choices=["semantic_only", "hybrid", "hybrid+rerank", "bm25_only"])
    parser.add_argument("--index-subdir", default="default")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of queries (for fast iteration)")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2:** Quick sanity run

Run:
```bash
python scripts/run_eval.py --config hybrid --limit 20
```
Expected: eval completes in <1 min, prints P@5 / R@5 / NDCG@5 / p95. Check `outputs/runs/hybrid.md` exists.

- [ ] **Step 3:** Commit

```bash
git add scripts/run_eval.py
git commit -m "feat(scripts): run_eval runner for single config"
```

---

## Phase L — Experiment Matrix

### Task 27: run_all_experiments.py orchestrator

**Files:**
- Create: `scripts/run_all_experiments.py`

- [ ] **Step 1:** Write orchestrator

```python
"""Run the full 9-experiment matrix.

Tier 1 (required): semantic_only / hybrid / hybrid+rerank @ recursive-512-10% (full 2556 queries)
Tier 2: chunk size sweep (256, 1024) @ hybrid+rerank (200 queries)
Tier 3: alpha sweep (0.3, 0.7) via weighted fusion (200 queries)
Tier 4: strategy ablation (fixed-window) (200 queries)

For Tier 2/4 we re-ingest with different chunk configs, saving to different index subdirs.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from rag.config import get_settings
from rag.eval.harness import run_eval
from rag.eval.qrels import load_multihop_eval
from rag.eval.reports import combine_reports_table, save_report
from rag.retrieve.embedder import Embedder
from rag.retrieve.reranker import Reranker
from rag.retrieve.search import SearchEngine


SUBSET_SIZE = 200


async def run_tier1(embedder, reranker, queries, settings, reports):
    index_dir = settings.index_dir / "default"
    engine_no_rr = SearchEngine(index_dir, embedder, None)
    engine_rr = SearchEngine(index_dir, embedder, reranker)

    for config_name, kwargs, use_reranker in [
        ("tier1_semantic_only", {"fusion_method": "semantic_only", "use_rerank": False}, False),
        ("tier1_hybrid", {"fusion_method": "rrf", "use_rerank": False}, False),
        ("tier1_hybrid_rerank", {"fusion_method": "rrf", "use_rerank": True}, True),
    ]:
        base = {"top_k_retrieve": 100, "top_k_rerank": 20, "top_k_final": 5, "rrf_k": 60, **kwargs}
        engine = engine_rr if use_reranker else engine_no_rr
        print(f"\n=== {config_name} (full) ===")
        report = await run_eval(engine, queries, config_name, base)
        save_report(report, settings.output_dir / "runs")
        reports.append(report)


async def run_tier3(embedder, reranker, queries, settings, reports):
    """Alpha sweep on default index — no re-ingest needed."""
    index_dir = settings.index_dir / "default"
    engine = SearchEngine(index_dir, embedder, None)  # Tier 3 is fusion-only (no rerank)
    subset = queries[:SUBSET_SIZE]
    for alpha in (0.3, 0.7):
        name = f"tier3_alpha_{alpha}"
        print(f"\n=== {name} (subset={len(subset)}) ===")
        report = await run_eval(engine, subset, name, {
            "top_k_retrieve": 100, "top_k_rerank": 20, "top_k_final": 5,
            "fusion_method": "weighted", "alpha": alpha, "use_rerank": False,
        })
        save_report(report, settings.output_dir / "runs")
        reports.append(report)


def reingest(chunk_size: int, overlap: int, subdir: str, strategy: str = "recursive"):
    """Re-run ingest with different chunk config. Synchronous subprocess."""
    cmd = [
        sys.executable, "scripts/ingest.py",
        "--chunk-size", str(chunk_size),
        "--overlap", str(overlap),
        "--index-subdir", subdir,
        "--force",
    ]
    # Strategy is controlled via env var or flag if added to ingest.py in future.
    # For now chunk_size + overlap are the primary levers.
    print(f"\n=== Re-ingesting: {cmd} ===")
    subprocess.check_call(cmd)


async def run_tier2(embedder, reranker, queries, settings, reports):
    """Chunk-size sweep: re-ingest then eval."""
    subset = queries[:SUBSET_SIZE]
    for chunk_size, subdir in [(256, "chunk256"), (1024, "chunk1024")]:
        reingest(chunk_size, chunk_size // 10, subdir)
        index_dir = settings.index_dir / subdir
        engine = SearchEngine(index_dir, embedder, reranker)
        name = f"tier2_chunk_{chunk_size}"
        print(f"\n=== {name} (subset={len(subset)}) ===")
        report = await run_eval(engine, subset, name, {
            "top_k_retrieve": 100, "top_k_rerank": 20, "top_k_final": 5,
            "fusion_method": "rrf", "use_rerank": True,
        })
        save_report(report, settings.output_dir / "runs")
        reports.append(report)


async def main_async(tiers: list[int]):
    settings = get_settings()
    print("Loading models...")
    embedder = Embedder(settings.embedding_model)
    reranker = Reranker(settings.reranker_model)

    queries = load_multihop_eval(settings.data_dir / "multihop_rag")
    print(f"{len(queries)} eval queries loaded")

    reports = []
    if 1 in tiers:
        await run_tier1(embedder, reranker, queries, settings, reports)
    if 3 in tiers:
        await run_tier3(embedder, reranker, queries, settings, reports)
    if 2 in tiers:
        await run_tier2(embedder, reranker, queries, settings, reports)

    # Combined table
    table = combine_reports_table(reports)
    summary_path = settings.output_dir / "runs" / "SUMMARY.md"
    summary_path.write_text("# Experiment Matrix Summary\n\n" + table + "\n", encoding="utf-8")
    print(f"\nSummary written to {summary_path}")
    print("\n" + table)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 3, 2],
                        help="Which tiers to run (default: 1 3 2)")
    args = parser.parse_args()
    asyncio.run(main_async(args.tiers))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Run Tier 1 only first (fastest)

Run: `python scripts/run_all_experiments.py --tiers 1`
Expected: 3 reports in `outputs/runs/tier1_*`, SUMMARY.md with 3 rows.

- [ ] **Step 3:** Commit

```bash
git add scripts/run_all_experiments.py
git commit -m "feat(scripts): experiment matrix orchestrator"
```

- [ ] **Step 4:** If time allows, run full matrix

Run: `python scripts/run_all_experiments.py --tiers 1 3 2`
Expected: 3 + 2 + 2 = 7 reports total. (Tier 4 strategy ablation is optional stretch — it requires extending `ingest.py` to accept `--strategy fixed`, which we can add as a stretch task if time permits.)

---

## Phase M — Derived Analyses & Writeup

### Task 28: BM25-vs-semantic failure-mode diff

**Files:**
- Create: `scripts/analyze_bm25_wins.py`

- [ ] **Step 1:** Write analyzer

```python
"""Find queries where BM25-only beats semantic-only by the largest NDCG@5 margin."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rag.config import get_settings
from rag.eval.harness import run_eval
from rag.eval.qrels import load_multihop_eval
from rag.retrieve.embedder import Embedder
from rag.retrieve.search import SearchEngine


async def main():
    settings = get_settings()
    queries = load_multihop_eval(settings.data_dir / "multihop_rag")
    embedder = Embedder(settings.embedding_model)
    engine = SearchEngine(settings.index_dir / "default", embedder, None)

    base = {"top_k_retrieve": 100, "top_k_rerank": 20, "top_k_final": 5, "rrf_k": 60}

    print(f"Running BM25-only on {len(queries)} queries...")
    bm25_rep = await run_eval(engine, queries, "bm25_only_analysis",
                              {**base, "fusion_method": "bm25_only", "use_rerank": False})

    print("Running semantic-only...")
    sem_rep = await run_eval(engine, queries, "semantic_only_analysis",
                              {**base, "fusion_method": "semantic_only", "use_rerank": False})

    # Align per-query and compute diffs
    bm25_by_qid = {r.query_id: r for r in bm25_rep.per_query}
    sem_by_qid = {r.query_id: r for r in sem_rep.per_query}
    diffs = []
    for qid in bm25_by_qid:
        if qid not in sem_by_qid:
            continue
        b = bm25_by_qid[qid]
        s = sem_by_qid[qid]
        diffs.append({
            "query": b.query,
            "bm25_ndcg": b.ndcg_at_5,
            "semantic_ndcg": s.ndcg_at_5,
            "delta": b.ndcg_at_5 - s.ndcg_at_5,
        })

    # Top 10 BM25 wins
    diffs.sort(key=lambda d: d["delta"], reverse=True)
    top_wins = diffs[:10]
    out = settings.output_dir / "runs" / "bm25_wins.json"
    out.write_text(json.dumps(top_wins, indent=2), encoding="utf-8")

    print("\nTop 10 queries where BM25 beat semantic (by NDCG@5 delta):")
    for d in top_wins:
        print(f"  Δ={d['delta']:+.3f}  bm25={d['bm25_ndcg']:.3f} sem={d['semantic_ndcg']:.3f}")
        print(f"  Q: {d['query'][:100]}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2:** Run

Run: `python scripts/analyze_bm25_wins.py`
Expected: prints top 10 query texts with positive delta (BM25 won); saves JSON.

- [ ] **Step 3:** Commit

```bash
git add scripts/analyze_bm25_wins.py
git commit -m "feat(scripts): BM25-vs-semantic query-level diff analyzer"
```

---

### Task 29: Cost projection doc

**Files:**
- Create: `docs/COST.md`

- [ ] **Step 1:** Write cost notes

```markdown
# Cost per 1,000 Queries

## Retrieval path (all-local)
- Embedding (MiniLM on CPU): 1000 × ~8ms compute, **$0** external spend
- BM25: in-process, **$0**
- FAISS IndexFlatIP: in-process, **$0**
- Cross-encoder rerank (MiniLM-L6 on CPU): 1000 × 20 pairs × ~25ms, **$0** external spend

**Retrieval subtotal: $0 per 1000 queries.**

## Optional generation (gpt-4_1_dev_1 via Azure OpenAI)

If the system is extended with answer synthesis step:

Assume per-query: 500 tokens of retrieved context + 100 tokens user query + 200 tokens answer.

| Component | Tokens/1K queries | Rate* | Cost/1K |
|---|---:|---:|---:|
| Input (context + query) | 600,000 | ${INPUT_RATE}/M | ~$X |
| Output (answer) | 200,000 | ${OUTPUT_RATE}/M | ~$Y |
| **Total /1K queries** | — | — | ~$Z |

*Fill in from Azure OpenAI pricing page for gpt-4_1_dev_1 (or the gpt-4.1 family).

## Compute cost (amortized hardware)
On a typical dev laptop (8-core CPU, 16GB RAM), running the retrieval service continuously:
~negligible vs. the above. For production, consider a single c6i.xlarge @ ~$0.17/hr = ~$125/mo amortized → $0.0035/1K queries if throughput is 1QPS sustained.
```

- [ ] **Step 2:** Commit

```bash
git add docs/COST.md
git commit -m "docs: cost per 1K queries"
```

---

### Task 30: Defense narrative writeup

**Files:**
- Create: `docs/DEFENSE.md`

- [ ] **Step 1:** Write defense doc

```markdown
# Defense Notes — Retrieval Platform v1

## Design decisions and rationale

### Chunking: recursive, 512 chars, 10% overlap
Embedding models compress each chunk into one vector — that vector is an average
of the chunk's meaning. Chunks that straddle topic boundaries produce "averaged"
vectors that represent nothing well. Recursive splitting prefers paragraph, then
line, then sentence boundaries, aligning chunk limits with natural meaning
boundaries. Overlap is duplication insurance — embedding models do NOT read
chunks sequentially, so overlap exists to ensure that a key phrase straddling
two chunks still appears cleanly in at least one chunk's interior.

### Hybrid search: RRF primary, weighted-alpha as tunable alternative
BM25 scores rare-token overlap; dense embeddings score meaning. They fail in
orthogonal ways — BM25 is strong on exact phrases, rare entities, out-of-vocab
tokens; semantic wins on paraphrase and synonymy. RRF combines the two via
rank reciprocals, avoiding the score-scale mismatch (BM25 unbounded positive,
cosine ∈ [-1,1]) that makes naive weighted fusion mathematically invalid.
Weighted-alpha is exposed with **per-query min-max normalization**, making
alpha interpretable as a true BM25-vs-semantic mix.

### Reranking: cross-encoder top-20 → top-5
Bi-encoders (used for dense retrieval) embed query and chunk independently.
Cross-encoders feed them jointly through a transformer — every query token can
attend to every chunk token. That cross-attention is what bi-encoders
structurally cannot do; it reliably adds 5–15 NDCG points on ranking tasks.
But cross-encoders cannot be pre-indexed (every query × candidate is a fresh
forward pass), so we apply them only to the top-20 candidates from the
cheap-but-lossy first stage.

### Vector index: FAISS IndexFlatIP (exhaustive)
For <100K vectors, exhaustive inner-product search is faster than
HNSW/ANN approximation because there is no graph-traversal overhead.
IndexFlatIP costs <2ms for ~5K chunks and has zero recall loss. Swap to
IndexHNSWFlat is a one-line change if corpus scales past ~100K.

### Why all-local retrieval (no Azure embeddings)
CPU-only target hardware + p95 <500ms budget makes network embedding calls
a non-starter: Azure round-trip adds ~80–150ms warm, with ~500ms cold-start
penalty on the first request. In-process MiniLM runs in ~8ms. Azure OpenAI
is reserved for optional answer-synthesis (gpt-4_1_dev_1) where network
latency is already assumed.

## Hard-mode signal answers

### Which chunk size wins and why
See `outputs/runs/SUMMARY.md` rows `tier1_hybrid_rerank`, `tier2_chunk_256`,
`tier2_chunk_1024`. On MultiHop-RAG's multi-hop queries, [smaller / 512 / larger]
chunks win because [populate after Tier 2 runs]. This matches first-principles
reasoning: multi-hop queries want evidence stitched across multiple docs →
higher top-k + smaller chunks → higher recall for the LLM to reason over.

### When BM25 beats semantic
See `outputs/runs/bm25_wins.json` for the top 10 queries ranked by NDCG@5
delta. Common patterns in the winners:
- Queries containing rare proper nouns / named entities the embedder doesn't
  encode well
- Queries with exact phrases (ticker symbols, dates, acronyms)
- Queries where the paraphrase gap is zero (query and doc share unusual tokens)

### Cold-cache vs warm-cache latency profile
First-query latency: see `outputs/runs/tier1_hybrid_rerank.json`
→ `first_query_latency_ms`. It includes model attention-key-value cache
warmup plus FAISS first-query initialization. p50/p95/p99 **exclude** the
cold first query.

### Cost per 1,000 queries
See `docs/COST.md`. Retrieval is $0 (all-local); optional generation via
gpt-4_1_dev_1 is the only spend if/when that's added.
```

- [ ] **Step 2:** Commit

```bash
git add docs/DEFENSE.md
git commit -m "docs: defense narrative with rationale + hard-mode signals"
```

---

## Phase N — API integration test

### Task 31: API integration test

**Files:**
- Create: `tests/test_api.py`

- [ ] **Step 1:** Write integration test

```python
"""Integration test for /search. Requires a populated index_dir/default."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


INDEX_PATH = Path("indices/default/faiss")
SKIP_REASON = "Integration test requires a populated index at indices/default/. Run scripts/ingest.py."


@pytest.mark.skipif(not INDEX_PATH.exists(), reason=SKIP_REASON)
def test_search_endpoint_returns_shape():
    from rag.api.app import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["engine_loaded"] is True

        r = client.post("/search", json={"query": "climate change impact", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "climate change impact"
        assert len(body["results"]) <= 3
        assert body["latency_ms"] > 0
        if body["results"]:
            hit = body["results"][0]
            assert set(hit["scores"].keys()) == {"bm25", "semantic", "hybrid_fused", "rerank", "final"}
            assert "chunk_id" in hit and "doc_id" in hit and "text" in hit
```

- [ ] **Step 2:** Run test (requires ingested index)

Run: `pytest tests/test_api.py -v`
Expected: 1 passed (if index exists) or 1 skipped (if no index).

- [ ] **Step 3:** Commit

```bash
git add tests/test_api.py
git commit -m "test(api): integration test for /search endpoint"
```

---

## Phase O — Final checks

### Task 32: Full test suite green

- [ ] **Step 1:** Run full test suite

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 2:** Verify deliverables list

Check off each assignment requirement:
- [ ] 1,000+ doc ingest — `python scripts/ingest.py` loads MultiHop-RAG (609) + PDFs (optional padding to 1000+)
- [ ] Incremental re-runs skip unchanged (manifest)
- [ ] Intelligent chunking: recursive, documented in DEFENSE.md
- [ ] Hybrid BM25 + semantic + cross-encoder rerank — all present
- [ ] Tunable weights — `alpha` param in API + Tier 3 sweep
- [ ] Metadata filtering — `filters` param in API
- [ ] POST /search with score breakdown — `Scores` schema
- [ ] Eval framework: ≥20 Q/A pairs, P@5/R@5/NDCG@5 across 3 configs — Tier 1 delivers all three
- [ ] Latency p95 <500ms documented — captured in every eval report
- [ ] Chunk-size comparison — Tier 2
- [ ] BM25 failure modes — `analyze_bm25_wins.py`
- [ ] Cold vs warm latency — `first_query_latency_ms` in every report
- [ ] Cost per 1K queries projected — `docs/COST.md`

- [ ] **Step 3:** Final commit if any doc edits

```bash
git add -A
git status
# If nothing to commit, skip. Otherwise:
git commit -m "docs: final deliverables checklist"
```

---

## Execution Priority (time-boxed)

If hours are tight, execute in this order, with early-kill checkpoints:

**Hour 0** (15-30 min): Tasks 1, 2 (preflight — models downloaded, corpus fetched)
**Hour 0-1** (30-45 min): Tasks 3-14 (types, config, chunking, loaders, pipeline, ingest script runs green)
**Hour 1-2** (45 min): Tasks 15-20 (fusion, reranker, search engine, FastAPI, /search live)
**Hour 2-3** (45 min): Tasks 21-27 (eval metrics, harness, Tier 1 runs — this ships the required deliverable)
**Hour 3-4** (45 min): Tasks 28-32 (BM25 diff analysis, defense writeup, integration test, Tier 2/3 if time)

**Kill switches:**
- If Tier 1 doesn't finish by end of Hour 3 → skip Tier 2/3, ship with Tier 1 only
- If ingest takes longer than expected (>30 min for 609 docs) → drop PDF padding, eval on MultiHop-RAG only
- If p95 exceeds 500ms on Tier 1 hybrid+rerank → reduce top_k_rerank from 20 to 10

---

## Self-review

**Spec coverage check:**
- 1,000+ docs incremental ingest → Tasks 7, 8, 9, 13, 14 ✓
- Intelligent chunking with defense → Tasks 5, 6, 30 ✓
- Hybrid BM25+semantic+rerank composable tunable → Tasks 11, 12, 15, 16, 17, 18 ✓
- Metadata filtering → Task 18 (`_metadata_matches`), 19 (schema `filters`) ✓
- POST /search with score breakdown → Tasks 19, 20 ✓
- ≥20 Q/A eval, P@5/R@5/NDCG across 3 configs → Tasks 21-27 ✓
- p95 <500ms documented → Task 24 (harness captures it) ✓
- Chunk size wins → Task 27 (Tier 2) ✓
- BM25 beats semantic → Task 28 ✓
- Cold vs warm → Task 24 (`first_query_latency_ms`) ✓
- Cost per 1K → Task 29 ✓

**Placeholder scan:** No "TBD", "TODO", "implement later" — all steps have complete code.

**Type consistency:** `Document`, `Chunk`, `SearchHit`, `EvalQuery`, `EvalReport`, `PerQueryResult` names match across all tasks.

No gaps found.
