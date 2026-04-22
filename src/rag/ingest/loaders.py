"""Document loaders for MultiHop-RAG (HuggingFace) and PDF files."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import fitz  # pymupdf

from rag.types import Document


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_multihop_from_hf(corpus_dir: Path) -> Iterator[Document]:
    """Yield Document per article from the MultiHop-RAG 'corpus' config saved to disk.

    Expects the dataset produced by preflight.py — a DatasetDict with a 'train' split
    whose rows have keys: category, author, published_at, body, title, url, source.
    """
    from datasets import load_from_disk

    ds = load_from_disk(str(corpus_dir))
    split = "train" if "train" in ds else list(ds.keys())[0]
    for row in ds[split]:
        text = row.get("body", "") or ""
        if not text.strip():
            continue
        doc_id = row.get("url") or f"article::{row.get('title', '')}"
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


def load_pdf(pdf_path: Path) -> Document | None:
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
