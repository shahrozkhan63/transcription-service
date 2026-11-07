"""End-to-end accuracy test — Word Error Rate against a known transcript.

Self-skips unless you drop a clip + its reference transcript in tests/data/:
    tests/data/sample.wav   (any short clip, ~10-30s)
    tests/data/sample.txt   (the correct transcript, plain text)

Then `pytest` transcribes the clip and asserts WER is below a sane threshold. This is how you
prove the pipeline actually works, with a number — not just "it ran". Requires ffmpeg + the model.
"""
import os

import pytest

DATA = os.path.join(os.path.dirname(__file__), "data")
AUDIO = os.path.join(DATA, "sample.wav")
REF = os.path.join(DATA, "sample.txt")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(AUDIO) and os.path.exists(REF)),
    reason="drop tests/data/sample.wav + sample.txt to enable the WER accuracy test",
)


def test_word_error_rate_within_threshold():
    import jiwer

    from app.pipeline import run

    reference = open(REF, encoding="utf-8").read().strip()
    result = run(AUDIO)
    hypothesis = " ".join(s.text for s in result.segments).strip()

    # normalise casing/punctuation so WER measures words, not formatting
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    wer = jiwer.wer(reference, hypothesis, truth_transform=transform, hypothesis_transform=transform)

    print(f"\nWER={wer:.3f}  rtf={result.rtf}  lang={result.language}")
    assert wer < 0.25, f"WER too high ({wer:.3f}); check model size / audio quality"
