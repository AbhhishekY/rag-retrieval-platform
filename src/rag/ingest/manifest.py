"""SHA-256 content-hash manifest for incremental ingestion.

Stores per-doc hashes so re-runs skip unchanged docs. Single SQLite file.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class IngestManifest:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.db_path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _ensure_schema(self) -> None:
        with self._conn() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS ingest_manifest (
                    doc_id TEXT PRIMARY KEY,
                    content_sha256 TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                )"""
            )

    def is_unchanged(self, doc_id: str, content_sha256: str) -> bool:
        with self._conn() as con:
            row = con.execute(
                "SELECT content_sha256 FROM ingest_manifest WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        return row is not None and row[0] == content_sha256

    def record(self, doc_id: str, content_sha256: str, chunk_count: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as con:
            con.execute(
                """INSERT INTO ingest_manifest(doc_id, content_sha256, chunk_count, indexed_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     content_sha256=excluded.content_sha256,
                     chunk_count=excluded.chunk_count,
                     indexed_at=excluded.indexed_at""",
                (doc_id, content_sha256, chunk_count, now),
            )

    def known_doc_ids(self) -> set[str]:
        with self._conn() as con:
            rows = con.execute("SELECT doc_id FROM ingest_manifest").fetchall()
        return {r[0] for r in rows}
