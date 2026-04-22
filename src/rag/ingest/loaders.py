"""Document loaders for MultiHop-RAG (HuggingFace), PDF files, and CSV files."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterator

import fitz  # pymupdf

from rag.types import Document


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pick(row: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        if row.get(k):
            return str(row[k])
    return default


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


def load_csv(csv_path: Path) -> Iterator[Document]:
    """Load documents from a CSV file.

    Flexible column mapping — tries common variants for each field:
      text:     text, body, content, description
      title:    title, headline, name
      id/url:   url, id, doc_id, link
      category: category, label, topic, type
      date:     published_at, date, pub_date, created_at
      source:   source, author, publication, outlet
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = _pick(row, "text", "body", "content", "description")
            if not text.strip():
                continue
            title = _pick(row, "title", "headline", "name")
            url = _pick(row, "url", "id", "doc_id", "link",
                        default=f"csv::{csv_path.stem}::{i}")
            doc_id = url if url.startswith(("http", "csv::")) else f"csv::{csv_path.stem}::{url}"
            yield Document(
                doc_id=doc_id,
                source=csv_path.name,
                title=title,
                text=text,
                metadata={
                    "category": _pick(row, "category", "label", "topic", "type"),
                    "published_at": _pick(row, "published_at", "date", "pub_date", "created_at"),
                    "author": _pick(row, "author", "byline"),
                    "source_name": _pick(row, "source", "publication", "outlet",
                                         default=csv_path.stem),
                    "format": "csv",
                },
                content_sha256=_sha256(text),
            )


def load_csv_directory(csv_dir: Path) -> Iterator[Document]:
    for csv_path in sorted(csv_dir.glob("*.csv")):
        yield from load_csv(csv_path)
