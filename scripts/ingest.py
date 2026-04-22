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
    parser.add_argument("--pdf-dir", type=str, default=None)
    parser.add_argument("--index-subdir", type=str, default="default")
    args = parser.parse_args()

    settings = get_settings()
    chunk_size = args.chunk_size if args.chunk_size is not None else settings.chunk_size
    overlap = args.overlap if args.overlap is not None else settings.chunk_overlap
    index_dir = settings.index_dir / args.index_subdir

    docs = []
    corpus_dir = settings.data_dir / "multihop_rag_corpus"
    if corpus_dir.exists():
        docs.extend(load_multihop_from_hf(corpus_dir))
    else:
        print(f"ERROR: {corpus_dir} not found. Run scripts/preflight.py first.", file=sys.stderr)
        return 1

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
