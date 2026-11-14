"""SQLite persistence for accounts, sessions, encrypted secrets and job ownership (stored under STORAGE_DIR)."""
import os
import sqlite3
import threading
from typing import Any, List, Optional, Sequence

from app.config import storage_root

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at REAL NOT NULL,
  expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS secrets (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  ciphertext BLOB NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL,
  UNIQUE (user_id, kind, name)
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at REAL NOT NULL
);
"""

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = storage_root() / "demoday.db"
        fresh = not path.exists()
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        if fresh:
            os.chmod(path, 0o600)  # password hashes and ciphertexts: owner-only
        _conn = conn
    return _conn


def query(sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
    with _lock:
        return _connection().execute(sql, params).fetchall()


def one(sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Sequence[Any] = ()) -> int:
    with _lock:
        return _connection().execute(sql, params).rowcount
