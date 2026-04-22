"""Core dataclasses for documents, chunks, and search results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_sha256: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    chunk_id: str
    doc_id: str
    text: str
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
