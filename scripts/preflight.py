"""Hour-0 ritual: pre-download models, pull MultiHop-RAG, kick off arXiv padding.

Run this ONCE before anything else. Prevents late-stage hangs on model downloads
and network weirdness during the actual build.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def ensure_models() -> None:
    print("[1/3] Pre-downloading sentence-transformers models...")
    t0 = time.time()
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SentenceTransformer("all-MiniLM-L6-v2")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"      models cached in {time.time() - t0:.1f}s")


def fetch_multihop_rag(data_dir: Path) -> None:
    print("[2/3] Downloading MultiHop-RAG benchmark...")
    t0 = time.time()
    from datasets import load_dataset

    ds = load_dataset("yixuantt/MultiHopRAG")
    out = data_dir / "multihop_rag"
    ds.save_to_disk(str(out))
    print(f"      saved to {out} in {time.time() - t0:.1f}s")
    # Inspect the first record to verify qrels format
    first_split = list(ds.keys())[0]
    print(f"      sample record keys: {list(ds[first_split][0].keys())}")


def inspect_qrels_format(data_dir: Path) -> None:
    print("[3/3] Inspecting qrels format...")
    from datasets import load_from_disk

    ds = load_from_disk(str(data_dir / "multihop_rag"))
    for split in ds.keys():
        rec = ds[split][0]
        print(f"      [{split}] keys: {list(rec.keys())}")
        if "evidence_list" in rec:
            print(f"      evidence_list type: {type(rec['evidence_list']).__name__}")
            if rec["evidence_list"]:
                print(f"      evidence item keys: {list(rec['evidence_list'][0].keys())}")
        print()


def main() -> int:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    try:
        ensure_models()
        fetch_multihop_rag(data_dir)
        inspect_qrels_format(data_dir)
    except Exception as e:
        print(f"\nPreflight FAILED: {type(e).__name__}: {e}")
        return 1
    print("\nPreflight OK. Safe to proceed with ingest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
