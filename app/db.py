"""SQLite persistence for accounts, sessions, encrypted secrets, fixtures and job ownership.

The database lives at DB_PATH (default: STORAGE_DIR/demoday.db). On Cloud Run the storage directory is a mounted
Cloud Storage bucket, which SQLite can't lock, so DB_PATH points to local disk and DB_BACKUP_PATH keeps a
consistent copy on the bucket: restored on startup, refreshed shortly after every write.
"""
import logging
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence

from app.config import storage_root

logger = logging.getLogger("demoday.db")
BACKUP_DELAY_SEC = 2.0

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
CREATE TABLE IF NOT EXISTS fixtures (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  origin TEXT NOT NULL,
  entries TEXT NOT NULL,
  polished INTEGER NOT NULL DEFAULT 0,
  brief TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL,
  UNIQUE (user_id, origin)
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at REAL NOT NULL
);
"""

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_backup_timer: Optional[threading.Timer] = None


def db_path() -> Path:
    configured = os.getenv("DB_PATH", "").strip()
    return Path(configured) if configured else storage_root() / "demoday.db"


def backup_path() -> Optional[Path]:
    configured = os.getenv("DB_BACKUP_PATH", "").strip()
    return Path(configured) if configured else None


def _restore_if_needed(path: Path) -> None:
    """Copies the backup into place atomically (a partial copy must never become the live database)."""
    backup = backup_path()
    if path.exists() or backup is None or not backup.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".restore")
    try:
        shutil.copyfile(backup, tmp)
        if sqlite3.connect(tmp).execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("backup failed its integrity check")
        os.replace(tmp, path)
        logger.info("restored database from %s", backup)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"database restore from {backup} failed: {exc}") from exc


def _write_backup() -> None:
    """A consistent copy of the live database: snapshot to local disk under the lock (fast), then copy it to the
    backup location (a slow mounted bucket) outside the lock. A failed copy is retried shortly after."""
    backup = backup_path()
    if backup is None or _conn is None:
        return
    local_snapshot = db_path().with_name(db_path().name + ".snapshot")
    try:
        with _lock:
            dest = sqlite3.connect(local_snapshot)
            try:
                _conn.backup(dest)
            finally:
                dest.close()
        backup.parent.mkdir(parents=True, exist_ok=True)
        tmp = backup.with_name(backup.name + ".tmp")
        shutil.copyfile(local_snapshot, tmp)
        os.replace(tmp, backup)
    except Exception as exc:  # the live database is unaffected; retry after a pause
        logger.warning("database backup to %s failed: %s (retrying)", backup, exc)
        _schedule_backup(delay=10.0)


def flush_backup() -> None:
    """Writes the backup now (called at shutdown so the last seconds of writes survive a restart)."""
    global _backup_timer
    if _backup_timer is not None:
        _backup_timer.cancel()
        _backup_timer = None
    _write_backup()


def _schedule_backup(delay: float = BACKUP_DELAY_SEC) -> None:
    global _backup_timer
    if backup_path() is None:
        return
    if _backup_timer is not None:
        _backup_timer.cancel()
    _backup_timer = threading.Timer(delay, _write_backup)
    _backup_timer.daemon = True
    _backup_timer.start()


def _connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _restore_if_needed(path)
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
        count = _connection().execute(sql, params).rowcount
    _schedule_backup()
    return count
