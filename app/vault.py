"""Per-user secrets encrypted at rest with AES-256-GCM.

The key comes from DEMODAY_SECRET_KEY (32 random bytes, base64) and is never written to disk by this service.
Each ciphertext is bound to its owner, kind and name through the GCM associated data, so a record copied to
another user or slot fails to decrypt instead of leaking.
"""
import base64
import json
import os
import time
import uuid
from typing import Dict, List, Optional, Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app import db

KEY_ENV = "DEMODAY_SECRET_KEY"
VERSION = b"\x01"
GENERATE_HINT = (
    "python -c \"import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())\""
)


class VaultUnavailable(RuntimeError):
    pass


def _key() -> bytes:
    raw = os.getenv(KEY_ENV, "").strip()
    if not raw:
        raise VaultUnavailable(f"{KEY_ENV} is not set, so secrets can't be stored. Generate one with: {GENERATE_HINT}")
    try:
        key = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    except ValueError:
        key = b""
    if len(key) != 32:
        raise VaultUnavailable(f"{KEY_ENV} must be 32 random bytes, base64-encoded. Generate one with: {GENERATE_HINT}")
    return key


def available() -> bool:
    try:
        _key()
        return True
    except VaultUnavailable:
        return False


def _aad(user_id: str, kind: str, name: str) -> bytes:
    return f"demoday:v1:{user_id}:{kind}:{name}".encode()


def _public(row) -> Dict:
    return {"id": row["id"], "kind": row["kind"], "name": row["name"], "created_at": row["created_at"],
            **json.loads(row["meta"] or "{}")}


def put(user_id: str, kind: str, name: str, secret: Union[str, bytes], meta: Optional[Dict] = None) -> Dict:
    """Stores (or replaces) one secret; returns its metadata, never the secret."""
    data = secret.encode() if isinstance(secret, str) else secret
    nonce = os.urandom(12)
    blob = VERSION + nonce + AESGCM(_key()).encrypt(nonce, data, _aad(user_id, kind, name))
    db.execute(
        "INSERT INTO secrets (id, user_id, kind, name, ciphertext, meta, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (user_id, kind, name) DO UPDATE SET id = excluded.id, ciphertext = excluded.ciphertext, "
        "meta = excluded.meta, created_at = excluded.created_at",
        (uuid.uuid4().hex, user_id, kind, name, blob, json.dumps(meta or {}), time.time()),
    )
    return _public(db.one("SELECT * FROM secrets WHERE user_id = ? AND kind = ? AND name = ?", (user_id, kind, name)))


def get(user_id: str, kind: str, name: str) -> Optional[bytes]:
    row = db.one("SELECT ciphertext FROM secrets WHERE user_id = ? AND kind = ? AND name = ?", (user_id, kind, name))
    if row is None:
        return None
    blob = bytes(row["ciphertext"])
    if blob[:1] != VERSION:
        raise VaultUnavailable("Unsupported secret format")
    try:
        return AESGCM(_key()).decrypt(blob[1:13], blob[13:], _aad(user_id, kind, name))
    except InvalidTag as exc:
        raise VaultUnavailable(f"A stored secret can't be decrypted; was {KEY_ENV} changed?") from exc


def meta(user_id: str, kind: str, name: str) -> Optional[Dict]:
    row = db.one("SELECT * FROM secrets WHERE user_id = ? AND kind = ? AND name = ?", (user_id, kind, name))
    return _public(row) if row else None


def list_kind(user_id: str, kind: str) -> List[Dict]:
    rows = db.query("SELECT * FROM secrets WHERE user_id = ? AND kind = ? ORDER BY created_at DESC", (user_id, kind))
    return [_public(r) for r in rows]


def delete(user_id: str, kind: str, *, name: Optional[str] = None, secret_id: Optional[str] = None) -> bool:
    if secret_id:
        return db.execute("DELETE FROM secrets WHERE id = ? AND user_id = ? AND kind = ?",
                          (secret_id, user_id, kind)) > 0
    return db.execute("DELETE FROM secrets WHERE user_id = ? AND kind = ? AND name = ?", (user_id, kind, name)) > 0


def encode_key(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode()
