"""Google Cloud integrations, all optional: public share links (Cloud Storage) and Chirp 3 HD narration (Text-to-Speech).

Credentials come from the runtime (the Cloud Run service account via Application Default Credentials); nothing is
configured when GCS_BUCKET / GOOGLE_TTS are unset, and every caller falls back cleanly.
"""
import asyncio
import logging
import os
import secrets
from pathlib import Path
from typing import Optional, Tuple

from app.config import env_flag

logger = logging.getLogger("demoday.cloud")

# voice_style -> (Chirp 3 HD voice, speaking rate)
GOOGLE_VOICES = {
    "energetic_tech": ("en-US-Chirp3-HD-Aoede", 1.08),
    "calm_expert": ("en-US-Chirp3-HD-Charon", 0.96),
    "fast_founder": ("en-US-Chirp3-HD-Leda", 1.15),
}


class CloudUnavailable(RuntimeError):
    pass


def public_bucket_name() -> str:
    return os.getenv("GCS_PUBLIC_BUCKET", "").strip()


def storage_bucket_name() -> str:
    return os.getenv("GCS_BUCKET", "").strip()


def google_tts_enabled() -> bool:
    return env_flag("GOOGLE_TTS")


async def share_video_url(job_id: str, local_path: Path) -> str:
    """A link anyone can open: the video copied into the public bucket under an unguessable token.

    The runtime service account only needs object access on the buckets (no URL-signing permission), and a
    link stays valid until the object is deleted. The job's own storage bucket, when mounted, is copied
    server-side; otherwise the local file is uploaded.
    """
    public_bucket = public_bucket_name()
    if not public_bucket:
        raise CloudUnavailable("Share links need a public Cloud Storage bucket (GCS_PUBLIC_BUCKET) on this server")
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise CloudUnavailable("google-cloud-storage is not installed") from exc

    def copy() -> str:
        client = storage.Client()
        token = secrets.token_urlsafe(18)
        target = client.bucket(public_bucket).blob(f"{token}/demo.mp4")
        source_bucket = storage_bucket_name()
        if source_bucket:
            source = client.bucket(source_bucket).blob(f"{job_id}/demo.mp4")
            if source.exists():
                client.bucket(source_bucket).copy_blob(source, client.bucket(public_bucket), target.name)
                target.reload()
            else:
                target.upload_from_filename(str(local_path), content_type="video/mp4")
        else:
            target.upload_from_filename(str(local_path), content_type="video/mp4")
        if target.content_type != "video/mp4":
            target.content_type = "video/mp4"
            target.patch()
        return f"https://storage.googleapis.com/{public_bucket}/{target.name}"

    try:
        return await asyncio.to_thread(copy)
    except Exception as exc:
        logger.warning("could not publish a share link: %s", exc)
        raise CloudUnavailable("Couldn't create a share link right now") from exc


async def google_tts(text: str, dest: Path, voice_style: str, credentials=None) -> Path:
    """Synthesizes `text` with Google Cloud Text-to-Speech (Chirp 3 HD) into a 24 kHz WAV."""
    try:
        from google.cloud import texttospeech
    except ImportError as exc:
        raise CloudUnavailable("google-cloud-texttospeech is not installed") from exc
    voice_name, rate = GOOGLE_VOICES.get(voice_style, GOOGLE_VOICES["energetic_tech"])

    def synthesize() -> Path:
        # REST, not gRPC: gRPC's background threads make the later ffmpeg fork() stall the whole process.
        client = texttospeech.TextToSpeechClient(credentials=credentials, transport="rest")
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=24000, speaking_rate=rate),
        )
        dest.write_bytes(response.audio_content)
        return dest

    return await asyncio.to_thread(synthesize)
