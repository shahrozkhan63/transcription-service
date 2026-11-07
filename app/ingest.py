"""Step 1 — ingest & validate.

Key decision: trust the file's CONTENT (via ffprobe), never its extension. A ".wav" can be
anything; ffprobe tells us what the bytes actually are, whether there's a real audio stream,
and the true duration — so we can reject corrupt / empty / no-audio / oversized files up front
with a clear error instead of failing deep inside the model.
"""
import json
import os
import subprocess
from dataclasses import dataclass


class IngestError(ValueError):
    """Anything we refuse to process. The API maps this to a 4xx with the message."""


@dataclass
class AudioInfo:
    format_name: str
    duration_sec: float
    sample_rate: int | None
    channels: int | None


def probe_and_validate(path: str, *, max_upload_mb: int, max_duration_sec: int) -> AudioInfo:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_upload_mb:
        raise IngestError(f"File is {size_mb:.1f} MB; limit is {max_upload_mb} MB.")

    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        raise IngestError("ffprobe not found — install ffmpeg and ensure it is on PATH.")

    if proc.returncode != 0:
        raise IngestError("Could not read the file — it may be corrupt or not a media file.")

    info = json.loads(proc.stdout or "{}")

    # A video file WITH an audio track is fine (we extract the audio in normalize).
    # A file with NO audio stream cannot be transcribed.
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise IngestError("No audio stream found in the file.")

    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or audio.get("duration") or 0.0)
    if duration <= 0:
        raise IngestError("File reports zero / unknown duration — likely empty or corrupt.")
    if duration > max_duration_sec:
        raise IngestError(f"Audio is {duration:.0f}s; limit is {max_duration_sec}s.")

    return AudioInfo(
        format_name=fmt.get("format_name", "unknown"),
        duration_sec=duration,
        sample_rate=int(audio["sample_rate"]) if audio.get("sample_rate") else None,
        channels=audio.get("channels"),
    )
