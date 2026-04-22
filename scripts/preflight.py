"""Hour-0 ritual: pre-download models, pull MultiHop-RAG, kick off arXiv padding.

Run this ONCE before anything else. Prevents late-stage hangs on model downloads
and network weirdness during the actual build.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def ensure_models() -> None:
    print("[1/3] Pre-downloading FastEmbed (ONNX) models...")
    t0 = time.time()
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    # Downloading triggers the model cache; one dummy embed/rerank confirms it works
    embed = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _ = list(embed.embed(["warmup"]))
    rerank = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    _ = list(rerank.rerank("warmup query", ["warmup doc"]))
    print(f"      models cached in {time.time() - t0:.1f}s")


def fetch_multihop_rag(data_dir: Path) -> None:
    print("[2/3] Downloading MultiHop-RAG benchmark (two configs: corpus + MultiHopRAG)...")
    t0 = time.time()
    from datasets import load_dataset

    # 'corpus' holds the 609 articles for indexing
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")
    corpus.save_to_disk(str(data_dir / "multihop_rag_corpus"))
    # 'MultiHopRAG' holds the Q/A records for eval
    queries = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
    queries.save_to_disk(str(data_dir / "multihop_rag_queries"))
    print(f"      saved both configs in {time.time() - t0:.1f}s")


def inspect_qrels_format(data_dir: Path) -> None:
    print("[3/3] Inspecting formats...")
    from datasets import load_from_disk

    corpus_ds = load_from_disk(str(data_dir / "multihop_rag_corpus"))
    for split in corpus_ds.keys():
        rec = corpus_ds[split][0]
        print(f"      corpus[{split}]: {len(corpus_ds[split])} rows, keys: {list(rec.keys())}")

    queries_ds = load_from_disk(str(data_dir / "multihop_rag_queries"))
    for split in queries_ds.keys():
        rec = queries_ds[split][0]
        print(f"      queries[{split}]: {len(queries_ds[split])} rows, keys: {list(rec.keys())}")
        if "evidence_list" in rec and rec["evidence_list"]:
            ev0 = rec["evidence_list"][0]
            print(f"      evidence item keys: {list(ev0.keys()) if isinstance(ev0, dict) else type(ev0).__name__}")


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
