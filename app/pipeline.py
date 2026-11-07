"""Orchestrator — the single entry point both the API and the CLI call.

Flow: ingest (validate) → normalize (one canonical WAV) → chunk if long → transcribe each chunk
with its time offset → merge/dedup → return the JSON contract. Also measures the real-time factor
(processing time ÷ audio length), which is the key observability metric for a transcription service.
"""
import os
import tempfile
import time
import uuid

from . import chunking, ingest, normalize, postprocess, transcriber
from .config import settings
from .models import TranscriptionResult


def run(src_path: str, *, job_id: str | None = None, language: str | None = None) -> TranscriptionResult:
    job_id = job_id or uuid.uuid4().hex[:12]

    info = ingest.probe_and_validate(
        src_path, max_upload_mb=settings.max_upload_mb, max_duration_sec=settings.max_duration_sec
    )

    with tempfile.TemporaryDirectory() as work:
        wav = normalize.to_wav_16k_mono(
            src_path, os.path.join(work, "norm.wav"), sample_rate=settings.target_sample_rate
        )

        t0 = time.monotonic()
        windows = chunking.plan_chunks(
            wav, info.duration_sec,
            target=settings.chunk_target_sec,
            overlap=settings.chunk_overlap_sec,
            threshold=settings.long_audio_threshold_sec,
        )

        language_detected = language
        per_chunk = []
        for idx, (start, end) in enumerate(windows):
            # Single-window (short) audio skips the extra ffmpeg cut and transcribes the whole WAV.
            chunk_wav = wav if len(windows) == 1 else normalize.cut_wav(
                wav, os.path.join(work, f"chunk_{idx}.wav"), start, end,
                sample_rate=settings.target_sample_rate,
            )
            lang, segs = transcriber.transcribe(chunk_wav, offset=start, language=language_detected)
            language_detected = language_detected or lang
            per_chunk.append(segs)

        merged = postprocess.merge_segments(per_chunk, overlap=settings.chunk_overlap_sec)
        elapsed = time.monotonic() - t0

    return TranscriptionResult(
        job_id=job_id,
        language=language_detected or "unknown",
        duration_sec=round(info.duration_sec, 2),
        model=settings.model_size,
        rtf=round(elapsed / info.duration_sec, 3) if info.duration_sec else None,
        segments=merged,
    )
