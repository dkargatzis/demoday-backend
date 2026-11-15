"""Unit checks for vault, auth hashing, netguard, request validation, and private/multi-repo ingest."""
import asyncio
import base64
import os
import secrets
import sys
import tempfile
import threading
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
FAKE = "http://127.0.0.1:8766"
os.environ.update({
    "STORAGE_DIR": tempfile.mkdtemp(prefix="demoday-unit-"),
    "DEMODAY_SECRET_KEY": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
    "GITHUB_API_URL": FAKE, "GITHUB_RAW_URL": FAKE + "/raw", "GITHUB_TOKEN": "", "ALLOW_PRIVATE_URLS": "",
})
sys.path.insert(0, BACKEND)

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app import auth, db, vault  # noqa: E402
from app.models.schemas import PipelineRequest  # noqa: E402
from app.services import netguard  # noqa: E402
from app.services.ingest import extract_repos_context  # noqa: E402

GOOD = "good-token-0123456789abcdef"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail else ''}")


# --------------------------------------------------------------------------- fake GitHub
gh = FastAPI()
PRIVATE = {"name": "secret-app", "full_name": "acme/secret-app", "private": True, "default_branch": "main",
           "description": "Private product", "language": "TypeScript"}
PUBLIC = {"name": "public-lib", "full_name": "acme/public-lib", "private": False, "default_branch": "main",
          "description": "Public library", "language": "Python"}


def auth_state(request: Request):
    header = request.headers.get("authorization", "")
    return "good" if header == f"Bearer {GOOD}" else ("bad" if header else "none")


@gh.get("/repos/acme/{repo}")
async def repo(repo: str, request: Request):
    state = auth_state(request)
    if state == "bad":
        return JSONResponse({"message": "Bad credentials"}, 401)
    if repo == "secret-app":
        return PRIVATE if state == "good" else JSONResponse({"message": "Not Found"}, 404)
    return PUBLIC


@gh.get("/repos/acme/{repo}/readme")
async def readme(repo: str, request: Request):
    if repo == "secret-app" and auth_state(request) != "good":
        return JSONResponse({"message": "Not Found"}, 404)
    return PlainTextResponse(f"# {repo}\nThe {repo} README.")


@gh.get("/repos/acme/{repo}/git/trees/main")
async def tree(repo: str, request: Request):
    if repo == "secret-app" and auth_state(request) != "good":
        return JSONResponse({"message": "Not Found"}, 404)
    paths = ["app/page.tsx", "app/dashboard/page.tsx", "package.json"] if repo == "secret-app" else ["pyproject.toml"]
    return {"tree": [{"path": p, "type": "blob"} for p in paths]}


@gh.get("/raw/acme/{repo}/main/{name}")
async def raw(repo: str, name: str, request: Request):
    if repo == "secret-app" and auth_state(request) != "good":
        return JSONResponse({"message": "Not Found"}, 404)
    if name == "package.json":
        return {"dependencies": {"next": "15", "react": "19"}}
    return PlainTextResponse('[project]\ndependencies = ["fastapi>=0.115", "httpx"]\n')


threading.Thread(target=uvicorn.run, args=(gh,), kwargs={"host": "127.0.0.1", "port": 8766, "log_level": "error"},
                 daemon=True).start()
time.sleep(1.5)


async def main():
    print("vault")
    for uid in ("user-a", "user-b"):
        db.execute("INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, 'x', 0)",
                   (uid, f"{uid}@example.com"))
    stored = vault.put("user-a", "github_token", "default", "ghp_secret_value", {"login": "octo"})
    check("metadata never contains the secret", "ghp_secret_value" not in str(stored), stored)
    check("round trip decrypts", vault.get("user-a", "github_token", "default") == b"ghp_secret_value")
    raw_blob = db.one("SELECT ciphertext FROM secrets WHERE user_id='user-a'")["ciphertext"]
    check("stored bytes are ciphertext", b"ghp_secret_value" not in bytes(raw_blob))
    db.execute("INSERT INTO secrets (id, user_id, kind, name, ciphertext, meta, created_at) "
               "SELECT 'copy', 'user-b', kind, name, ciphertext, meta, created_at FROM secrets WHERE user_id='user-a'")
    try:
        vault.get("user-b", "github_token", "default")
        check("ciphertext copied to another user is rejected", False)
    except vault.VaultUnavailable:
        check("ciphertext copied to another user is rejected", True)
    key = os.environ.pop("DEMODAY_SECRET_KEY")
    check("vault reports unavailable without a key", vault.available() is False)
    os.environ["DEMODAY_SECRET_KEY"] = key

    print("passwords")
    hashed = auth.hash_password("correct horse battery")
    check("scrypt hash verifies", auth.verify_password("correct horse battery", hashed))
    check("wrong password rejected", not auth.verify_password("correct horse battery!", hashed))
    check("malformed hash rejected", not auth.verify_password("x", "not-a-hash"))

    print("network guard")
    for host in ["127.0.0.1", "10.1.2.3", "192.168.1.10", "169.254.169.254", "100.64.0.1", "::1",
                 "::ffff:127.0.0.1", "localhost", "0.0.0.0"]:
        check(f"blocks {host}", await netguard.host_problem(host) is not None)
    for host in ["1.1.1.1", "github.com"]:
        check(f"allows {host}", await netguard.host_problem(host) is None)
    for url in ["file:///etc/passwd", "javascript:alert(1)", "http://169.254.169.254/latest/meta-data"]:
        try:
            await netguard.ensure_public_url(url)
            check(f"ensure_public_url rejects {url}", False)
        except netguard.BlockedURL as exc:
            check(f"ensure_public_url rejects {url}", True, exc)
    os.environ["ALLOW_PRIVATE_URLS"] = "true"
    check("ALLOW_PRIVATE_URLS=true permits localhost (local dev)", await netguard.host_problem("127.0.0.1") is None)
    os.environ["ALLOW_PRIVATE_URLS"] = ""

    print("request validation")
    legacy = PipelineRequest(repo_url="https://github.com/a/b", app_url="https://x.dev")
    check("legacy repo_url still accepted", legacy.repo_urls == ["https://github.com/a/b"], legacy.repo_urls)
    merged = PipelineRequest(repo_urls=["https://github.com/a/b", " https://github.com/a/b "],
                             repo_url="https://github.com/c/d", app_url="https://x.dev")
    check("repo lists merge and de-duplicate", merged.repo_urls == ["https://github.com/a/b", "https://github.com/c/d"],
          merged.repo_urls)
    cases = [
        ({"repo_urls": [f"https://github.com/a/r{i}" for i in range(6)]}, "six repos rejected"),
        ({"repo_urls": ["ftp://github.com/a/b"]}, "non-http repo rejected"),
    ]
    for bad, label in cases:
        try:
            PipelineRequest(app_url="https://x.dev", **bad)
            check(label, False)
        except ValidationError:
            check(label, True)

    print("private + multi-repo ingest (fake GitHub)")
    repos = ["https://github.com/acme/secret-app", "https://github.com/acme/public-lib"]
    anon = await extract_repos_context(repos)
    check("without a token the private repo is reported, not silently empty",
          any("isn't visible" in n and "Add one in Settings" in n for n in anon["notes"]), anon["notes"])
    check("without a token the public repo still contributes", "fastapi" in anon["dependencies"], anon["dependencies"])
    good = await extract_repos_context(repos, GOOD)
    check("with a token the private repo is read", good["repos"][0]["private"] is True and not good["notes"],
          good["notes"])
    check("routes come from the private repo tree", good["routes"] == ["/", "/dashboard"], good["routes"])
    check("dependencies merge across repos", {"next", "react", "fastapi", "httpx"} <= set(good["dependencies"]),
          good["dependencies"])
    check("READMEs are labelled per repo", "## acme/secret-app" in good["readme"] and "## acme/public-lib" in good["readme"])
    bad = await extract_repos_context(repos, "revoked-token-0123456789")
    check("a revoked token is reported and public data still loads",
          any("rejected the token" in n for n in bad["notes"]) and "fastapi" in bad["dependencies"], bad["notes"])

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
