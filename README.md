# Transcription Service

Audio → text with **per-segment timestamps**. An open-source speech-to-text pipeline built around
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and `ffmpeg`, exposed as a FastAPI
service and a CLI. Audio stays on your own servers — no third-party API required.

> This service powers **voice messages in [Dirayah AI](https://dirayahai.com)** (an AI chatbot for the
> UAE/GCC): a visitor records a voice note → it's transcribed here → the text flows into the existing
> chat/RAG pipeline. Demo video: _`<add link>`_.

The focus is **engineering decisions**, not training a model: format handling, long-audio strategy,
timestamp correctness, failure handling, observability, and a clean path to scale.

---

## Architecture

```
          audio/video file (wav, mp3, m4a, webm/opus, mp4/aac, …)
                               │  upload
                               ▼
        ┌──────────────────────────────────────────────────────────┐
        │  FastAPI  (app/api.py)                                     │
        │    POST /transcribe        → sync   (short clips)          │
        │    POST /transcribe/async  → 202 + job_id (long audio)     │
        │    GET  /jobs/{id}         → progress / result            │
        └──────────────────────────────────────────────────────────┘
                               │  pipeline.run()  (app/pipeline.py)
                               ▼
   [1] ingest.py     validate by CONTENT via ffprobe (not extension); reject corrupt/empty/no-audio/oversized
   [2] normalize.py  ffmpeg → 16 kHz mono 16-bit PCM WAV (one canonical format; extracts audio from video)
   [3] chunking.py   long audio → split at silences near ~30s, with overlap (ffmpeg silencedetect)
   [4] transcriber.py faster-whisper (int8, vad_filter, condition_on_previous_text=False); per-chunk offset
   [5] postprocess.py merge chunks, drop overlap duplicates, re-id; export SRT / VTT
                               │
                               ▼
        JSON { job_id, language, duration_sec, model, rtf, segments:[{id,start,end,text,avg_logprob}] }
```

## Quickstart

```bash
# 1. system dependency (ffmpeg + ffprobe)
#    macOS: brew install ffmpeg   |   Ubuntu: apt-get install -y ffmpeg   |   Windows: choco install ffmpeg
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2a. CLI
python cli.py sample.mp3                       # JSON to stdout
python cli.py meeting.m4a --format srt --out meeting.srt

# 2b. API
uvicorn app.api:app --reload                   # docs at http://127.0.0.1:8000/docs
curl -F "file=@sample.mp3" "http://127.0.0.1:8000/transcribe"
curl -F "file=@long.mp3"   "http://127.0.0.1:8000/transcribe/async"   # → {"job_id": "...", ...}
curl "http://127.0.0.1:8000/jobs/<job_id>"
```

## Output contract

```json
{
  "job_id": "a1b2c3d4e5f6",
  "language": "en",
  "duration_sec": 3621.4,
  "model": "small",
  "rtf": 0.42,
  "segments": [
    { "id": 0, "start": 0.00, "end": 4.32, "text": "Welcome to the meeting.", "avg_logprob": -0.21 }
  ]
}
```

## Key engineering decisions

| Decision | Why | Trade-off |
|---|---|---|
| **faster-whisper** (Whisper on CTranslate2) | ~4× faster than openai-whisper, int8, CPU/GPU, segment timestamps, **audio stays on our servers** | A hosted API (Deepgram/AssemblyAI/OpenAI) ships faster but costs per minute and sends audio to a third party |
| **Normalize everything with ffmpeg first** | The model + pipeline only ever see one format (16 kHz mono PCM) → no per-format code paths | One extra decode step (cheap) |
| **Detect format by content (ffprobe)** | A `.wav` can be anything; validate the real bytes and fail fast with a clear error | — |
| **Silence-aware chunking via ffmpeg** | Fixed 30s cuts split words; cutting at silence near 30s + overlap keeps words whole — and avoids a torch/Silero dependency just for boundaries | Silence detection is heuristic; faster-whisper's own `vad_filter` (Silero) still runs inside transcription |
| **Per-chunk timestamp offset** | Merged output is absolute to the original audio, not relative to a chunk — the classic long-audio bug, handled explicitly | — |
| **Async `202 + job_id` for long audio** | An HTTP request can't stay open for 20+ minutes | Caller must poll (or add a webhook) |
| `condition_on_previous_text=False` | Reduces Whisper inventing text to match prior context | Slightly less fluent continuity across segments |

## How it answers the brief

- **Different audio formats** → not special-cased. `ffmpeg` decodes *everything* to 16 kHz mono PCM; video containers have their audio track extracted (`-vn`); stereo is mixed to mono; VBR-MP3 timestamp drift disappears because we work on raw samples. Format is detected by content, not extension.
- **Long audio** → (a) silence-aware chunking at ~30s with overlap + timestamp offsets + overlap-dedup; (b) an async job API so the request returns immediately; (c) per-chunk results so only a failed chunk is retried, not the whole file.

## Observability

Every result carries **`rtf` (real-time factor = processing time ÷ audio length)** — the single most
useful health metric: `rtf < 1` means faster than real time; a rising `rtf` is the first sign of an
overloaded worker or a too-large model for the hardware. Per-chunk timing makes it easy to find a
slow chunk.

## Scaling (what production looks like)

- **Stateless workers** behind the queue — the HTTP contract above is already async, so swapping the
  in-memory store for **Celery/RQ + Redis** needs no client change.
- **Parallel chunks**: chunks are independent, so a long file fans out across workers and merges.
- **GPU pool** for throughput (`device=cuda`, `compute_type=float16`); autoscale / scale-to-zero so
  the model isn't held in RAM 24/7 when idle (cost).
- **Object storage (S3)** for the audio and results instead of local temp files.

## Cost

Driven almost entirely by the model size and whether the model is held in memory:
`tiny → base → small → medium → large-v3` trade accuracy for speed, RAM and $. `int8` quantization
roughly halves memory. Self-hosting is fixed cost (a box that must stay warm); a hosted API is
pay-per-minute. The right answer depends on volume — low/spiky volume often favours an API or
scale-to-zero GPU; steady high volume favours a warm self-hosted pool.

## Testing

`tests/` covers the deterministic logic directly (chunk planning, timestamp offsetting, overlap
dedup, SRT/VTT formatting) and includes an **accuracy check using Word Error Rate (`jiwer`)** against
a small known-transcript clip, plus a ~30-minute long-file test to exercise chunking end to end.
Run with `pytest`.

## Privacy

Audio is written to a temp file, processed, and **deleted immediately** after (success or failure —
see the `finally` blocks). Nothing is sent to a third party. A production deployment adds an explicit
retention policy and auto-delete on stored audio/results.

## Limitations (honest)

- The async store is **in-memory** here (demo); production = Celery/RQ + Redis (durable, parallel).
- No speaker diarization yet (who-said-what).
- Silence-based chunk boundaries are heuristic; very noisy audio with no clear silences falls back to
  fixed 30s cuts.

## Next steps

Speaker diarization (`pyannote`) — and for call audio, keeping stereo channels separate gives speaker
separation for free · custom vocabulary / initial-prompt biasing for domain terms · streaming /
real-time transcription · text-to-speech voice replies to close the loop in Dirayah AI.
