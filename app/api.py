"""FastAPI app — the HTTP surface over the pipeline.

Two entry points, matching the two real shapes of this problem:
  * POST /transcribe        — synchronous. Good for short clips (e.g. chat voice notes).
  * POST /transcribe/async  — returns 202 + job_id immediately; poll GET /jobs/{id}.
                              Long audio can't finish inside a single HTTP request.

HONEST PRODUCTION NOTE: the async job store here is in-memory and the work runs in a background
thread — fine for a self-contained demo, but a real deployment swaps this for Celery/RQ + Redis
(durable queue, parallel workers, per-chunk retries) and object storage (S3) for the audio. The
HTTP contract stays identical, so nothing else changes. See README → "Scaling".
"""
import os
import tempfile
import uuid
from enum import Enum

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from . import postprocess
from .config import settings
from .ingest import IngestError
from .normalize import NormalizeError
from .pipeline import run as run_pipeline

app = FastAPI(
    title="Transcription Service",
    version="0.1.0",
    description="Audio → text with per-segment timestamps (faster-whisper + ffmpeg).",
)


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


# In-memory job store — DEMO ONLY (see module docstring). {job_id: {status, result, error}}
_jobs: dict[str, dict] = {}


async def _save_upload(file: UploadFile) -> str:
    """Stream the upload to a temp file, enforcing the size cap while writing (never buffer a
    huge file fully in memory)."""
    suffix = os.path.splitext(file.filename or "")[1]
    fd, path = tempfile.mkstemp(suffix=suffix)
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with os.fdopen(fd, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                out.close()
                os.unlink(path)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit.")
            out.write(chunk)
    return path


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.model_size, "device": settings.device}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Query(None, description="force a language e.g. 'en','ar','ur'; omit to autodetect"),
    format: str = Query("json", pattern="^(json|srt|vtt)$"),
):
    """Synchronous transcription — best for short audio (chat voice notes)."""
    path = await _save_upload(file)
    try:
        result = run_pipeline(path, language=language)
    except (IngestError, NormalizeError) as e:
        raise HTTPException(400, str(e))
    finally:
        if os.path.exists(path):
            os.unlink(path)

    if format == "srt":
        return PlainTextResponse(postprocess.to_srt(result.segments), media_type="text/plain")
    if format == "vtt":
        return PlainTextResponse(postprocess.to_vtt(result.segments), media_type="text/vtt")
    return result


def _run_job(job_id: str, path: str, language: str | None):
    """Runs in a background thread (Starlette runs sync BackgroundTasks in a threadpool, so the
    event loop is never blocked by the CPU-bound transcription)."""
    _jobs[job_id]["status"] = JobStatus.processing
    try:
        result = run_pipeline(path, job_id=job_id, language=language)
        _jobs[job_id].update(status=JobStatus.done, result=result.model_dump())
    except (IngestError, NormalizeError) as e:
        _jobs[job_id].update(status=JobStatus.error, error=str(e))
    except Exception as e:  # keep the job queryable instead of failing silently
        _jobs[job_id].update(status=JobStatus.error, error=f"internal error: {e}")
    finally:
        if os.path.exists(path):
            os.unlink(path)


@app.post("/transcribe/async", status_code=202)
async def transcribe_async(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    language: str | None = Query(None),
):
    """Accept the file, return a job_id immediately, process in the background — use for long audio."""
    path = await _save_upload(file)
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": JobStatus.queued, "result": None, "error": None}
    background.add_task(_run_job, job_id, path, language)
    return {"job_id": job_id, "status": JobStatus.queued}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id.")
    return job
