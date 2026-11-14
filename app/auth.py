"""Accounts and cookie sessions for the hosted dashboard.

Passwords are hashed with scrypt. The browser holds a random session token in an HttpOnly cookie and only its
SHA-256 is stored, so a leaked database contains no usable sessions.
"""
import asyncio
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app import db
from app.config import env_flag

COOKIE = "demoday_session"
SESSION_TTL_SEC = 14 * 24 * 3600
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ATTEMPT_WINDOW_SEC, MAX_ATTEMPTS = 600, 40
_attempts: Dict[str, Deque[float]] = defaultdict(deque)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., min_length=8, max_length=256)
    signup_code: Optional[str] = Field(default=None, max_length=200)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        expected = bytes.fromhex(digest)
        actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                                dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))  # keeps login timing equal for unknown emails


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _client_key(request: Request) -> str:
    """The caller's address. Behind Cloud Run's proxy request.client is the proxy; the proxy appends the real
    client to X-Forwarded-For, so the LAST entry is the trustworthy one (a client can't spoof it)."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip() or "unknown"
    return request.client.host if request.client else "unknown"


def _rate_limit(request: Request) -> None:
    key = _client_key(request)
    now, window = time.monotonic(), _attempts[key]
    while window and now - window[0] > ATTEMPT_WINDOW_SEC:
        window.popleft()
    if len(window) >= MAX_ATTEMPTS:
        raise HTTPException(429, "Too many attempts. Try again in a few minutes.")
    window.append(now)


def user_from_token(token: Optional[str]) -> Optional[Dict]:
    if not token:
        return None
    row = db.one(
        "SELECT users.id, users.email, sessions.expires_at FROM sessions JOIN users ON users.id = sessions.user_id "
        "WHERE sessions.token_hash = ?", (_token_hash(token),))
    if row is None or row["expires_at"] < time.time():
        return None
    return {"id": row["id"], "email": row["email"]}


async def current_user(request: Request) -> Dict:
    user = user_from_token(request.cookies.get(COOKIE))
    if user is None:
        raise HTTPException(401, "Sign in to continue")
    return user


def _start_session(response: Response, user_id: str) -> None:
    token, now = secrets.token_urlsafe(32), time.time()
    db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
    db.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
               (_token_hash(token), user_id, now, now + SESSION_TTL_SEC))
    response.set_cookie(COOKIE, token, max_age=SESSION_TTL_SEC, httponly=True, secure=env_flag("COOKIE_SECURE"),
                        samesite="lax", path="/")


@router.post("/signup")
async def signup(body: Credentials, request: Request, response: Response):
    _rate_limit(request)
    required = os.getenv("SIGNUP_CODE", "").strip()
    if required and not hmac.compare_digest((body.signup_code or "").strip().encode(), required.encode()):
        raise HTTPException(403, "That signup code isn't valid")
    email = body.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(422, "Enter a valid email address")
    password_hash = await asyncio.to_thread(hash_password, body.password)
    user_id = uuid.uuid4().hex
    try:
        db.execute("INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                   (user_id, email, password_hash, time.time()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists")
    _start_session(response, user_id)
    return {"id": user_id, "email": email}


@router.post("/login")
async def login(body: Credentials, request: Request, response: Response):
    _rate_limit(request)
    row = db.one("SELECT id, email, password_hash FROM users WHERE email = ?", (body.email.strip().lower(),))
    ok = await asyncio.to_thread(verify_password, body.password, row["password_hash"] if row else _DUMMY_HASH)
    if row is None or not ok:
        raise HTTPException(401, "Email or password is incorrect")
    _start_session(response, row["id"])
    return {"id": row["id"], "email": row["email"]}


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE)
    if token:
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
    response.delete_cookie(COOKIE, path="/", httponly=True, secure=env_flag("COOKIE_SECURE"), samesite="lax")
    return {"ok": True}


@router.get("/me")
async def me(user: Dict = Depends(current_user)):
    return user
