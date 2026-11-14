"""End-to-end pipeline test: runs a full job in-process, then validates demo.mp4 for web playback.

    python test_e2e.py [--repo URL] [--url URL] [--seconds 30] [--voice energetic_tech] [--timeout 900]

Exits non-zero if the pipeline fails or any output check fails.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Tuple

from app.config import GEMINI_MODEL, capabilities, storage_root  # loads .env
from app.models.schemas import JobStatus, PipelineRequest
from app.services.orchestrator import JOBS, create_pipeline_job
from app.services.stitcher import describe_media


def mp4_box_order(path: Path) -> List[str]:
    """Top-level MP4 box types in file order (used to verify moov precedes mdat, i.e. fast start)."""
    order = []
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            size, kind = int.from_bytes(header[:4], "big"), header[4:8].decode("latin-1")
            order.append(kind)
            if size == 1:
                size = int.from_bytes(f.read(8), "big")
                f.seek(size - 16, 1)
            elif size == 0:
                break
            else:
                f.seek(size - 8, 1)
    return order


async def validate(job: JobStatus) -> List[Tuple[str, bool, str]]:
    path = storage_root() / job.job_id / "demo.mp4"
    if not path.is_file():
        return [("demo.mp4 exists", False, str(path))]
    media = await describe_media(path)
    boxes = mp4_box_order(path)
    scenes = job.storyboard.scenes if job.storyboard else []
    storyboard = job.storyboard
    # per-scene clip lengths (narration + tail), plus the AI intro/outro when present
    expected = sum(s.duration_sec for s in scenes) + (storyboard.intro_sec + storyboard.outro_sec if storyboard else 0)
    av_gap = abs(media["video_duration"] - media["audio_duration"])
    kit = job.launch_kit
    return [
        ("demo.mp4 exists", True, f"{path} ({media['size_bytes'] / 1e6:.1f} MB)"),
        ("video codec is H.264", media["video_codec"] == "h264", str(media["video_codec"])),
        ("pixel format yuv420p (browser-safe)", media["pix_fmt"] == "yuv420p", str(media["pix_fmt"])),
        ("resolution is 1920x1080", (media["width"], media["height"]) == (1920, 1080),
         f"{media['width']}x{media['height']} @ {media['fps']} fps"),
        ("audio codec is AAC 48 kHz", media["audio_codec"] == "aac" and media["audio_rate"] == 48000,
         f"{media['audio_codec']} {media['audio_rate']} Hz, {media['audio_channels']} ch"),
        ("audio and video durations match (<=100 ms)", av_gap <= 0.1,
         f"video {media['video_duration']:.3f}s, audio {media['audio_duration']:.3f}s"),
        ("length matches the scene clips (+-0.5 s)", abs(media["duration"] - expected) <= 0.5,
         f"{media['duration']:.2f}s vs expected {expected:.2f}s over {len(scenes)} scenes"
         + (f" + {storyboard.intro_sec:.1f}s intro + {storyboard.outro_sec:.1f}s outro" if storyboard and storyboard.intro_sec else "")),
        ("moov atom before mdat (streams before full download)",
         "moov" in boxes and "mdat" in boxes and boxes.index("moov") < boxes.index("mdat"), " > ".join(boxes)),
        ("launch kit populated",
         bool(kit and kit.readme_markdown and kit.devpost_pitch and kit.x_thread and kit.linkedin_post),
         f"{len(kit.x_thread)} tweets" if kit else "missing"),
    ]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Test the DemoDay pipeline end to end")
    parser.add_argument("--repo", default="https://github.com/fastapi/fastapi", help="GitHub repository URL")
    parser.add_argument("--url", default="https://fastapi.tiangolo.com", help="Live deployed application URL")
    parser.add_argument("--seconds", type=int, default=30, help="Target demo length in seconds")
    parser.add_argument("--voice", default="energetic_tech", choices=["energetic_tech", "calm_expert", "fast_founder"])
    parser.add_argument("--timeout", type=float, default=900, help="Fail if the job takes longer (seconds)")
    parser.add_argument("--ai-clips", action="store_true", help="Add the AI intro/outro (fal.ai video credits)")
    args = parser.parse_args()

    caps = capabilities()
    print("=" * 72)
    print("DemoDay Agent - End-to-End Pipeline Test")
    print("=" * 72)
    print(f"Target repo:  {args.repo}")
    print(f"Target URL:   {args.url}")
    print(f"Target time:  {args.seconds}s  ({args.voice})")
    print(f"FFmpeg:       {'[OK]' if caps['ffmpeg'] else '[MISSING - install ffmpeg]'}")
    print(f"GEMINI_KEY:   {f'[CONFIGURED] {GEMINI_MODEL}' if caps['gemini'] else '[UNSET - template storyboard]'}")
    print(f"FAL_KEY:      {'[CONFIGURED] Kokoro TTS' if caps['fal'] else '[UNSET - local TTS fallback]'}")
    print("=" * 72)
    if not caps["ffmpeg"]:
        return 1

    job = create_pipeline_job(PipelineRequest(
        repo_url=args.repo, app_url=args.url, target_seconds=args.seconds, voice_style=args.voice,
        ai_clips=args.ai_clips,
    ))
    print(f"Started pipeline job {job.job_id}")

    printed, started = 0, time.monotonic()
    while True:
        await asyncio.sleep(0.5)
        job = JOBS[job.job_id]
        for line in job.logs[printed:]:
            print(f"[{job.progress:3d}%] {line}")
        printed = len(job.logs)
        if job.status in ("completed", "failed"):
            break
        if time.monotonic() - started > args.timeout:
            print(f"TIMEOUT after {args.timeout:.0f}s (stage: {job.stage})")
            return 1

    print("=" * 72)
    if job.status == "failed":
        print(f"FAILED: {job.error}")
        return 1

    checks = await validate(job)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if job.launch_kit:
        print("\n--- Devpost pitch preview ---")
        print(job.launch_kit.devpost_pitch[:300] + "...")
    passed = all(ok for _, ok, _ in checks)
    print("=" * 72)
    print(f"{'SUCCESS' if passed else 'FAILED'}: {sum(ok for _, ok, _ in checks)}/{len(checks)} checks passed "
          f"in {time.monotonic() - started:.0f}s")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
