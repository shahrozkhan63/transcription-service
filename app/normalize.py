"""Step 2 — normalize everything to 16 kHz mono 16-bit PCM WAV (and cut chunks).

This is the answer to "how do you handle different audio formats?": we don't special-case
formats at all. Every input is decoded by ffmpeg to ONE canonical format, so the model and the
rest of the pipeline only ever see 16 kHz mono PCM. Decoding to PCM here also:
  - extracts the audio track out of video containers (-vn drops video),
  - mixes stereo down to mono (-ac 1),
  - resamples to 16 kHz (-ar 16000) which is what Whisper expects,
  - removes variable-bitrate MP3 timestamp drift (we work on raw samples, not the compressed stream).
"""
import subprocess


class NormalizeError(RuntimeError):
    pass


def to_wav_16k_mono(src: str, dst: str, *, sample_rate: int = 16000) -> str:
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vn",                     # drop video → pull the audio track out of containers
        "-ac", "1",                # mix down to mono
        "-ar", str(sample_rate),   # resample to 16 kHz
        "-c:a", "pcm_s16le",       # plain 16-bit PCM WAV
        dst,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise NormalizeError(f"ffmpeg normalize failed: {proc.stderr[-300:]}")
    return dst


def cut_wav(src_wav: str, dst_wav: str, start: float, end: float, *, sample_rate: int = 16000) -> str:
    """Cut a [start, end] window out of an already-normalized WAV for one chunk.
    -ss before -i = fast input seeking; we re-encode to PCM so the cut is sample-accurate."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src_wav,
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
        dst_wav,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise NormalizeError(f"ffmpeg cut failed: {proc.stderr[-300:]}")
    return dst_wav
