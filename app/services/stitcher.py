"""FFmpeg/ffprobe helpers: per-scene muxing, final concatenation, and media inspection."""
import asyncio
import json
import math
from pathlib import Path
from typing import List, Sequence, Union

PathLike = Union[str, Path]

FPS = 30
WIDTH, HEIGHT = 1920, 1080
AUDIO_RATE = 48000


class MediaToolError(RuntimeError):
    pass


async def _exec(cmd: Sequence[str], timeout: float) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise MediaToolError(
            f"{cmd[0]} not found on PATH. Install FFmpeg (macOS: brew install ffmpeg, Debian: apt-get install ffmpeg)."
        ) from exc
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise MediaToolError(f"{cmd[0]} timed out after {timeout:.0f}s")
    if proc.returncode != 0:
        tail = " | ".join(stderr.decode(errors="replace").strip().splitlines()[-5:])
        raise MediaToolError(f"{cmd[0]} exited with code {proc.returncode}: {tail}")
    return stdout.decode(errors="replace")


async def run_ffmpeg(args: Sequence[str], timeout: float = 600) -> None:
    await _exec(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args], timeout)


async def probe_duration(path: PathLike) -> float:
    """Exact container duration in seconds, as reported by ffprobe."""
    out = await _exec(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        timeout=30,
    )
    try:
        return float(out.strip())
    except ValueError as exc:
        raise MediaToolError(f"ffprobe returned no duration for {path}") from exc


async def describe_media(path: PathLike) -> dict:
    """Codec, geometry, and per-stream durations of a rendered MP4."""
    info = json.loads(await _exec(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        timeout=30,
    ))
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    num, _, den = (video.get("avg_frame_rate") or "0/1").partition("/")
    return {
        "duration": float(info.get("format", {}).get("duration", 0)),
        "size_bytes": int(info.get("format", {}).get("size", 0)),
        "video_codec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": round(float(num) / float(den or 1), 2) if den and float(den) else 0.0,
        "video_duration": float(video.get("duration", 0)),
        "audio_codec": audio.get("codec_name"),
        "audio_rate": int(audio.get("sample_rate", 0) or 0),
        "audio_channels": audio.get("channels"),
        "audio_duration": float(audio.get("duration", 0)),
    }


def frame_aligned(seconds: float) -> float:
    """Rounds a duration up to a whole number of video frames so audio and video can end together."""
    return math.ceil(round(seconds * FPS, 6)) / FPS


async def mux_scene_clip(
    video_path: PathLike, audio_path: PathLike, output_path: PathLike,
    start_sec: float, duration_sec: float,
) -> None:
    """Cuts [start, start+duration] out of the session recording and muxes it with the scene narration.

    Both streams are forced to exactly `duration_sec`: video holds its last frame if the recording runs
    short, and narration is padded with silence. Every clip shares identical codec parameters
    (H.264 high/yuv420p 1080p30, AAC 48 kHz stereo), so clips can be concatenated losslessly.
    """
    d = f"{duration_sec:.4f}"
    video_chain = (
        f"[0:v]setpts=PTS-STARTPTS,fps={FPS},"
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={d},trim=duration={d},setpts=PTS-STARTPTS,format=yuv420p[v]"
    )
    audio_chain = (
        f"[1:a]aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"apad,atrim=duration={d},asetpts=PTS-STARTPTS[a]"
    )
    await run_ffmpeg([
        "-ss", f"{max(0.0, start_sec):.4f}", "-i", str(video_path),
        "-i", str(audio_path),
        "-filter_complex", f"{video_chain};{audio_chain}",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "160k", "-ar", str(AUDIO_RATE), "-ac", "2",
        "-t", d, "-movflags", "+faststart",
        str(output_path),
    ])


async def concat_scenes(clip_paths: List[PathLike], final_output_path: PathLike) -> None:
    """Joins the per-scene clips into the final MP4 (stream copy, moov atom up front for web playback)."""
    if not clip_paths:
        raise MediaToolError("No scene clips to concatenate")
    final_output_path = Path(final_output_path)
    list_file = final_output_path.with_name("concat_list.txt")
    lines = []
    for clip in clip_paths:
        escaped = str(Path(clip).resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_file.write_text("\n".join(lines) + "\n")
    await run_ffmpeg([
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", "-movflags", "+faststart",
        str(final_output_path),
    ])
