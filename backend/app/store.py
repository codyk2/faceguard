"""SQLite persistence for enrolled identities and their per-image embeddings.

A new connection is opened per call, so a single Store instance is safe to share
across uvicorn's threadpool. The template is computed on read with
interfaces.make_template (cheap: a handful of 512-d vectors per identity).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from app import config
from app.ml.interfaces import make_template

EMBED_DIM = 512

_SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_identity ON embeddings(identity_id);
"""


class DuplicateNameError(Exception):
    """An identity with this name already exists."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_blob(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape != (EMBED_DIM,):
        raise ValueError(f"embedding must have shape ({EMBED_DIM},), got {arr.shape}")
    return arr.tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class Store:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else Path(config.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:  # commit on success, rollback on error
                yield conn
        finally:
            conn.close()

    # ---- writes -----------------------------------------------------------
    def create_identity(self, name: str, embeddings: list[np.ndarray]) -> dict:
        if not embeddings:
            raise ValueError("at least one embedding is required")
        blobs = [_to_blob(e) for e in embeddings]
        now = _now()
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO identities (name, created_at) VALUES (?, ?)", (name, now)
                )
                identity_id = cur.lastrowid
                conn.executemany(
                    "INSERT INTO embeddings (identity_id, vector, created_at) VALUES (?, ?, ?)",
                    [(identity_id, b, now) for b in blobs],
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateNameError(name) from exc
        return {
            "identity_id": identity_id,
            "name": name,
            "images_used": len(blobs),
            "created_at": datetime.fromisoformat(now),
        }

    def delete_identity(self, identity_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
            return cur.rowcount > 0

    # ---- reads ------------------------------------------------------------
    def name_exists(self, name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM identities WHERE name = ?", (name,)).fetchone()
            return row is not None

    def list_identities(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT i.id, i.name, i.created_at, COUNT(e.id) AS n
                FROM identities i LEFT JOIN embeddings e ON e.identity_id = i.id
                GROUP BY i.id ORDER BY i.id
                """
            ).fetchall()
        return [
            {
                "identity_id": r["id"],
                "name": r["name"],
                "images_used": r["n"],
                "created_at": datetime.fromisoformat(r["created_at"]),
            }
            for r in rows
        ]

    def get_identity(self, identity_id: int) -> dict | None:
        for ident in self.list_identities():
            if ident["identity_id"] == identity_id:
                return ident
        return None

    def get_embeddings(self, identity_id: int) -> list[np.ndarray]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT vector FROM embeddings WHERE identity_id = ? ORDER BY id",
                (identity_id,),
            ).fetchall()
        return [_from_blob(r["vector"]) for r in rows]

    def get_template(self, identity_id: int) -> np.ndarray | None:
        """Re-normalised mean embedding, or None if the identity is unknown."""
        embs = self.get_embeddings(identity_id)
        if not embs:
            return None
        return make_template(embs)
