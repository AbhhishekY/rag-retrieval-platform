"""Adapt MultiHop-RAG Q/A records into (query_id, query, relevant_doc_ids) tuples.

Schema (verified via preflight):
  - query (str)
  - answer (str)
  - question_type (str)
  - evidence_list: list of dicts with keys [author, category, fact, published_at, source, title, url]

Evidence 'url' matches the article 'url' used as doc_id by the corpus loader, so
precision/recall/NDCG can be computed at the document level.
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
    category: str = ""  # most common category across evidence items


def load_multihop_eval(queries_dir: Path) -> list[EvalQuery]:
    from datasets import load_from_disk

    ds = load_from_disk(str(queries_dir))
    split = "train" if "train" in ds else list(ds.keys())[0]
    out: list[EvalQuery] = []
    for row in ds[split]:
        q = row.get("query", "")
        if not q:
            continue
        evidence = row.get("evidence_list", []) or []
        rel_ids: set[str] = set()
        cat_counts: dict[str, int] = {}
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            url = ev.get("url")
            if url:
                rel_ids.add(url)
            cat = ev.get("category", "")
            if cat:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if not rel_ids:
            continue
        category = max(cat_counts, key=cat_counts.get) if cat_counts else ""
        qid = hashlib.md5(q.encode("utf-8")).hexdigest()[:12]
        out.append(
            EvalQuery(
                query_id=qid,
                query=q,
                relevant_doc_ids=rel_ids,
                question_type=row.get("question_type", ""),
                category=category,
            )
        )
    return out
