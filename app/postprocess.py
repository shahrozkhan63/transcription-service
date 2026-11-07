"""Step 5 — merge chunk results into one clean transcript, and export SRT / VTT.

When long audio is chunked with overlap, consecutive chunks transcribe the same overlap region
twice. Here we drop those duplicates (keeping absolute timestamps), re-id the segments, and offer
subtitle exports built from the same segment list.
"""
from .models import Segment


def merge_segments(chunk_segments: list[list[Segment]], *, overlap: float) -> list[Segment]:
    """Flatten per-chunk segments (each already carrying absolute timestamps) and drop the
    duplicates that fall inside the overlap between consecutive chunks."""
    merged: list[Segment] = []
    last_end = 0.0
    for segs in chunk_segments:
        for s in segs:
            # Anything that starts well inside already-covered time is an overlap duplicate.
            if s.start < last_end - 0.5:
                continue
            merged.append(s)
            last_end = max(last_end, s.end)
    for i, s in enumerate(merged):  # re-id sequentially after dedup
        s.id = i
    return merged


def _fmt_ts(t: float, sep: str) -> str:
    h, rem = divmod(t, 3600)
    m, secs = divmod(rem, 60)
    ms = min(int(round((secs - int(secs)) * 1000)), 999)
    return f"{int(h):02d}:{int(m):02d}:{int(secs):02d}{sep}{ms:03d}"


def to_srt(segments: list[Segment]) -> str:
    return "\n".join(
        f"{i + 1}\n{_fmt_ts(s.start, ',')} --> {_fmt_ts(s.end, ',')}\n{s.text}\n"
        for i, s in enumerate(segments)
    )


def to_vtt(segments: list[Segment]) -> str:
    body = "\n".join(
        f"{_fmt_ts(s.start, '.')} --> {_fmt_ts(s.end, '.')}\n{s.text}\n"
        for s in segments
    )
    return "WEBVTT\n\n" + body
