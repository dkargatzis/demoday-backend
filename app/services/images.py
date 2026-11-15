"""Small image helpers on top of FFmpeg (no PIL in the image): thumbnails for the agent trace, shrunk JPEGs for Gemini."""
import base64
import tempfile
from pathlib import Path
from typing import Union

from app.services.stitcher import run_ffmpeg

PathLike = Union[str, Path]


async def shrink_jpeg(path: PathLike, max_width: int = 1280, quality: int = 6) -> bytes:
    """`path` (PNG/JPEG) re-encoded as a JPEG no wider than `max_width` (ffmpeg -q:v quality: 2 best … 31 worst)."""
    return await _encode(path, f"scale='min(iw,{max_width})':-2", quality)


async def thumbnail_b64(path: PathLike, width: int = 320, quality: int = 6) -> str:
    """A `width`-pixel JPEG thumbnail of `path` as base64 (for a one-off step event, never for JobStatus)."""
    return base64.b64encode(await _encode(path, f"scale={width}:-2", quality)).decode("ascii")


async def _encode(path: PathLike, scale: str, quality: int) -> bytes:
    src = Path(path)
    with tempfile.TemporaryDirectory(prefix="demoday-img-", dir=src.parent) as tmp:
        out = Path(tmp) / "out.jpg"
        await run_ffmpeg(["-i", str(src), "-vf", scale, "-q:v", str(quality), "-frames:v", "1", "-update", "1", str(out)],
                         timeout=60)
        return out.read_bytes()
