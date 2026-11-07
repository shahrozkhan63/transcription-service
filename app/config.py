from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables live here so behaviour is configuration, not code.
    Override any value with an env var prefixed TRANSCRIBE_ (e.g. TRANSCRIBE_MODEL_SIZE=medium)."""

    # protected_namespaces=() silences pydantic's warning about the "model_size" field name
    # colliding with its reserved "model_" prefix — harmless here, but we keep output clean.
    model_config = SettingsConfigDict(env_prefix="TRANSCRIBE_", env_file=".env", protected_namespaces=())

    # Whisper model: tiny < base < small < medium < large-v3 — accuracy ↑, but speed/RAM cost ↑.
    # "small" + int8 is a good CPU default; bump to medium/large-v3 on a GPU.
    model_size: str = "small"
    device: str = "cpu"             # "cuda" when a GPU is available
    compute_type: str = "int8"      # int8 on CPU; "float16" on GPU

    target_sample_rate: int = 16000  # Whisper is trained on 16 kHz mono

    # Ingest guardrails — reject early rather than load a huge or garbage file into the model.
    max_upload_mb: int = 50
    max_duration_sec: int = 2 * 60 * 60   # 2-hour hard cap

    # Long-audio chunking
    long_audio_threshold_sec: float = 90.0   # at/below this → single pass; above → chunk
    chunk_target_sec: float = 30.0           # Whisper's native attention window is ~30s
    chunk_overlap_sec: float = 1.5           # overlap so a word split at a boundary still lands whole in one chunk


settings = Settings()
