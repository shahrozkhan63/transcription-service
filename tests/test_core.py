"""Deterministic unit tests — the logic that must be correct regardless of the model or audio.

These run with NO ffmpeg / Whisper / audio file: chunk planning is tested with silence detection
monkeypatched, and merge/SRT/VTT are pure functions. (End-to-end accuracy is in test_wer.py.)
"""
from app import chunking, postprocess
from app.models import Segment


def test_short_audio_is_a_single_window():
    # <= threshold → one window, and no ffmpeg call is made
    windows = chunking.plan_chunks("unused.wav", duration=40.0, target=30, overlap=1.5, threshold=90)
    assert windows == [(0.0, 40.0)]


def test_long_audio_chunks_cover_everything_with_overlap(monkeypatch):
    # force "no silences found" so we exercise the fixed-cut fallback deterministically
    monkeypatch.setattr(chunking, "_detect_silences", lambda *a, **k: [])
    windows = chunking.plan_chunks("unused.wav", duration=120.0, target=30, overlap=1.5, threshold=90)

    assert windows[0][0] == 0.0                     # starts at the beginning
    assert windows[-1][1] == 120.0                  # ends at the end (full coverage)
    for (s0, e0), (s1, e1) in zip(windows, windows[1:]):
        assert s1 < e0                              # consecutive chunks overlap (no gaps, no split words)


def test_chunks_prefer_a_nearby_silence(monkeypatch):
    # a silence at 31.0 is within the window around the ideal 30s cut → chunk should end there
    monkeypatch.setattr(chunking, "_detect_silences", lambda *a, **k: [31.0])
    windows = chunking.plan_chunks("unused.wav", duration=120.0, target=30, overlap=1.5, threshold=90)
    assert windows[0] == (0.0, 31.0)


def test_merge_drops_overlap_duplicates_and_reids():
    chunk_a = [Segment(id=0, start=0.0, end=5.0, text="hello"),
               Segment(id=1, start=5.0, end=10.0, text="world")]
    # chunk B began inside the overlap, so its first segment repeats "world"
    chunk_b = [Segment(id=0, start=8.6, end=10.0, text="world"),
               Segment(id=1, start=10.0, end=14.0, text="again")]

    merged = postprocess.merge_segments([chunk_a, chunk_b], overlap=1.5)

    assert [s.text for s in merged] == ["hello", "world", "again"]   # duplicate removed
    assert [s.id for s in merged] == [0, 1, 2]                        # re-id sequential


def test_srt_and_vtt_timestamp_format():
    segs = [Segment(id=0, start=0.0, end=4.32, text="Welcome to the meeting.")]
    srt = postprocess.to_srt(segs)
    vtt = postprocess.to_vtt(segs)
    assert "00:00:00,000 --> 00:00:04,320" in srt
    assert srt.strip().startswith("1\n")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:04.320" in vtt
