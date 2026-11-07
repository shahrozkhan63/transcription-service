"""CLI — transcribe a local file without running the server.

Examples:
    python cli.py sample.mp3
    python cli.py meeting.m4a --language en --format srt --out meeting.srt
"""
import argparse
import json
import sys

from app import postprocess
from app.ingest import IngestError
from app.normalize import NormalizeError
from app.pipeline import run


def main() -> None:
    p = argparse.ArgumentParser(description="Transcribe an audio/video file to text with timestamps.")
    p.add_argument("path", help="path to an audio or video file")
    p.add_argument("--language", default=None, help="force a language (e.g. en, ar, ur); omit to autodetect")
    p.add_argument("--format", choices=["json", "srt", "vtt"], default="json")
    p.add_argument("--out", default=None, help="write to this file instead of stdout")
    args = p.parse_args()

    try:
        result = run(args.path, language=args.language)
    except (IngestError, NormalizeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        text = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    elif args.format == "srt":
        text = postprocess.to_srt(result.segments)
    else:
        text = postprocess.to_vtt(result.segments)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}  (rtf={result.rtf}, lang={result.language}, segments={len(result.segments)})")
    else:
        print(text)


if __name__ == "__main__":
    main()
