"""Step 4 — transcription with faster-whisper (Whisper on CTranslate2).

Why faster-whisper: open source, ~4x faster than openai-whisper, int8 quantization, runs on CPU
or GPU, returns per-segment timestamps, and the audio never leaves our servers (unlike a hosted
API). The model is loaded once and reused across requests.
"""
from faster_whisper import WhisperModel

from .config import settings
from .models import Segment

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(settings.model_size, device=settings.device,
                              compute_type=settings.compute_type)
    return _model


def transcribe(wav_path: str, *, offset: float = 0.0, language: str | None = None):
    """Transcribe one WAV — the whole file, or a single chunk.

    `offset` shifts every timestamp by the chunk's start time, so when chunks are merged the
    timestamps are absolute (relative to the original audio), not relative to the chunk. Getting
    this offset wrong is the classic long-audio bug.
    """
    segments, info = get_model().transcribe(
        wav_path,
        vad_filter=True,                   # Silero VAD: skip silence → faster, fewer hallucinations
        condition_on_previous_text=False,  # stop the model inventing text to fit prior context
        language=language,                 # None = autodetect (Dirayah AI users speak EN / AR / UR)
    )
    out = [
        Segment(
            id=i,
            start=round(s.start + offset, 2),
            end=round(s.end + offset, 2),
            text=s.text.strip(),
            avg_logprob=s.avg_logprob,
        )
        for i, s in enumerate(segments)
        if s.text.strip()
    ]
    return info.language, out
