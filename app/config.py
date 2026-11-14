"""Central configuration. Importing this module loads the backend's .env exactly once."""
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger("demoday.config")

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")

# gemini-2.0-flash was shut down on 2026-06-01; Google lists gemini-3.6-flash as its replacement.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# GitHub endpoints, overridable for GitHub Enterprise Server.
GITHUB_API_URL = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
GITHUB_RAW_URL = os.getenv("GITHUB_RAW_URL", "https://raw.githubusercontent.com").rstrip("/")


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def get_secret(name: str) -> str:
    """Returns a secret from the environment, treating unset or template placeholder values as missing."""
    value = os.getenv(name, "").strip()
    if not value or value == "..." or (value.startswith("your_") and value.endswith("_here")):
        return ""
    return value


DEFAULT_STORAGE = BACKEND_ROOT / "storage"
_storage_root: Optional[Path] = None


def storage_root() -> Path:
    """Directory for per-job media. Relative STORAGE_DIR values resolve against the backend root.

    An unwritable STORAGE_DIR (typically the Docker path /app/storage copied into a local .env) falls
    back to demoday-backend/storage with a one-time warning instead of failing every job.
    """
    global _storage_root
    if _storage_root is not None:
        return _storage_root
    configured = Path(os.getenv("STORAGE_DIR", "storage"))
    root = configured if configured.is_absolute() else BACKEND_ROOT / configured
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if root == DEFAULT_STORAGE:
            raise
        logger.warning("STORAGE_DIR=%s is not writable (%s); using %s instead", configured, exc.strerror,
                       DEFAULT_STORAGE)
        root = DEFAULT_STORAGE
        root.mkdir(parents=True, exist_ok=True)
    _storage_root = root
    return root


def work_root() -> Path:
    """Scratch space for a job's intermediates (recordings, clips, narration). Defaults to the storage root;
    on Cloud Run WORK_DIR points to local disk so only the finished demo.mp4 and job.json reach the bucket."""
    configured = os.getenv("WORK_DIR", "").strip()
    root = Path(configured) if configured else storage_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def capabilities() -> dict:
    from app import vault  # imported here to avoid a config <-> vault import cycle

    return {
        "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
        "gemini": bool(get_secret("GEMINI_API_KEY")),
        "fal": bool(get_secret("FAL_KEY")),
        "gemini_model": GEMINI_MODEL,
        "vault": vault.available(),
        "signup_code": bool(os.getenv("SIGNUP_CODE", "").strip()),
    }
