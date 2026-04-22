"""Download AG News articles to supplement MultiHop-RAG corpus to 1,000+ docs.

AG News is a public benchmark dataset (4 categories: World, Sports, Business,
Sci/Tech) distributed via HuggingFace datasets. We pull 400 articles (100 per
category) and export them as a CSV so the ingest pipeline can treat them like
any other CSV source.

Usage:
    python scripts/fetch_supplementary.py           # saves to data/csvs/ag_news.csv
    python scripts/fetch_supplementary.py --count 500
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_LABEL_MAP = {0: "world", 1: "sports", 2: "business", 3: "technology"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=400,
                        help="Total articles to fetch (split evenly across 4 categories)")
    parser.add_argument("--out", type=str, default="data/csvs/ag_news.csv")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install -e \".[dev]\"", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_cat = max(1, args.count // 4)
    print(f"Fetching AG News: {per_cat} articles per category → {per_cat * 4} total")
    print("Downloading AG News (streaming)...")

    ds = load_dataset("ag_news", split="train", streaming=True, trust_remote_code=False)

    buckets: dict[int, list] = {0: [], 1: [], 2: [], 3: []}
    for row in ds:
        label = int(row["label"])
        if len(buckets[label]) < per_cat:
            buckets[label].append(row)
        if all(len(v) >= per_cat for v in buckets.values()):
            break

    rows_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "title", "text", "category", "source"])
        writer.writeheader()
        for label, articles in buckets.items():
            category = _LABEL_MAP[label]
            for i, art in enumerate(articles):
                # AG News text is "title. body" — split on first ". " if present
                raw = art["text"]
                if ". " in raw[:120]:
                    title, body = raw.split(". ", 1)
                else:
                    title, body = raw[:80], raw
                writer.writerow({
                    "url": f"ag_news::{category}::{i}",
                    "title": title.strip(),
                    "text": body.strip(),
                    "category": category,
                    "source": "AG News",
                })
                rows_written += 1

    print(f"Saved {rows_written} articles to {out_path}")
    print(f"\nTotal corpus after ingest: 609 (MultiHop-RAG) + {rows_written} (AG News) = {609 + rows_written} docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
