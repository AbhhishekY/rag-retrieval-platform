"""BM25 index over chunk texts. Lowercase whitespace+punctuation tokenization,
no stopword removal or lemmatization (BM25 IDF handles common terms well, and
lemmatization breaks named entities in news text).
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
        if self.bm25 is None or not self.chunk_ids:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
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
