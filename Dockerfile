FROM python:3.12-slim

# ffmpeg + ffprobe are required at runtime (ingest validation, normalization, chunking).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Which Whisper model to run. Baked below MUST match this.
ENV TRANSCRIBE_MODEL_SIZE=small

# Bake the model into the image at build time. Critical for scale-to-zero: without this, every
# cold start would re-download ~460 MB and take minutes. With it, a cold start just loads the model
# from the image's disk cache (~10-20s).
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY . .

# Railway injects $PORT; default to 8000 for local `docker run`.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
