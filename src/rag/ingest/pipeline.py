"""End-to-end ingest: docs -> chunks -> embeddings -> BM25 + FAISS -> save.

Incremental: skips docs whose content_sha256 matches the manifest. When any
doc changes we rebuild the full BM25 + FAISS indices (simple, correct, fast
enough for <100K chunks on MultiHop-RAG scale).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

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
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_batch_size: int = 64,
    force: bool = False,
) -> dict:
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = IngestManifest(index_dir / "ingest_manifest.db")

    all_docs = list(docs)
    print(f"Loaded {len(all_docs)} docs")

    if not force:
        changed = [d for d in all_docs if not manifest.is_unchanged(d.doc_id, d.content_sha256)]
        print(f"{len(changed)} docs changed since last ingest; "
              f"{len(all_docs) - len(changed)} skipped")
        if len(changed) == 0:
            print("No changes — indices still valid.")
            return {"docs_total": len(all_docs), "docs_changed": 0}

    all_chunks: list[Chunk] = []
    for d in tqdm(all_docs, desc="chunking"):
        all_chunks.extend(chunk_document(d, chunk_size, overlap))
    print(f"Produced {len(all_chunks)} chunks")

    embedder = Embedder(embedder_model)
    chunk_texts = [c.text for c in all_chunks]
    print("Embedding chunks...")
    vectors = embedder.encode_docs(chunk_texts, batch_size=embed_batch_size)

    chunk_ids = [c.chunk_id for c in all_chunks]
    bm25 = BM25Index()
    bm25.build(chunk_ids, chunk_texts)
    faiss_idx = FaissFlatIndex(dim=embedder.dim)
    faiss_idx.build(chunk_ids, vectors)

    bm25.save(index_dir / "bm25.pkl")
    faiss_idx.save(index_dir / "faiss")
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }) + "\n")

    per_doc_counts = Counter(c.doc_id for c in all_chunks)
    for d in all_docs:
        manifest.record(d.doc_id, d.content_sha256, per_doc_counts[d.doc_id])

    return {
        "docs_total": len(all_docs),
        "chunks_total": len(all_chunks),
        "embedding_dim": embedder.dim,
        "index_dir": str(index_dir),
    }
