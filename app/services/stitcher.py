import os
import asyncio
from typing import List

async def mux_scene_clip(video_path: str, audio_path: str, output_path: str):
    """Muxes scene video clip with scene voiceover audio using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()

async def concat_scenes(clip_paths: List[str], final_output_path: str):
    """Concatenates all muxed scene clips into a final demo MP4."""
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    list_file = os.path.join(os.path.dirname(final_output_path), "concat_list.txt")

    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        final_output_path
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
