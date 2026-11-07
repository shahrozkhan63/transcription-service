"""Step 3 — split long audio at natural silences.

Part of the answer to "how do you deal with long audio files?". Whisper's attention window is
~30s, so long audio must be split. Fixed 30s cuts would slice words in half; instead we cut at a
silence NEAR each ~30s mark, with a small overlap so a word straddling a boundary still lands
whole in one chunk. Silence is detected with ffmpeg (no extra torch/VAD dependency) — note that
faster-whisper's own vad_filter (Silero) still runs inside transcription, so we get Silero VAD
for the actual speech detection and use ffmpeg only to pick split points.
"""
import re
import subprocess


def _detect_silences(wav_path: str, *, noise_db: int = -30, min_silence: float = 0.4) -> list[float]:
    """Return the midpoints (seconds) of detected silences, using ffmpeg's silencedetect filter."""
    cmd = ["ffmpeg", "-i", wav_path, "-af",
           f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    return sorted((s + e) / 2 for s, e in zip(starts, ends))


def plan_chunks(wav_path: str, duration: float, *, target: float, overlap: float,
                threshold: float) -> list[tuple[float, float]]:
    """Return [(start, end), ...] windows covering the whole audio, overlapping by `overlap`s.
    Short audio returns a single window so we skip cutting entirely."""
    if duration <= threshold:
        return [(0.0, duration)]

    silences = _detect_silences(wav_path)
    chunks: list[tuple[float, float]] = []
    cur = 0.0
    while cur < duration:
        ideal = cur + target
        if ideal >= duration:
            chunks.append((cur, duration))
            break
        # Prefer a real silence within a window around the ideal cut; else hard-cut at `ideal`.
        window = [s for s in silences if ideal - 5 <= s <= ideal + 10]
        cut = min(window, key=lambda s: abs(s - ideal)) if window else ideal
        chunks.append((cur, cut))
        cur = max(cut - overlap, cur + 1.0)   # next chunk starts `overlap`s earlier; +1 guard against a stall
    return chunks
