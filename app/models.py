from pydantic import BaseModel, Field


class Segment(BaseModel):
    """One timestamped span of transcript. This is the unit the form explicitly asks for
    ("transcription with timestamps per segment")."""
    id: int
    start: float = Field(..., description="seconds from the start of the audio")
    end: float = Field(..., description="seconds from the start of the audio")
    text: str
    avg_logprob: float | None = Field(
        None, description="model confidence proxy; closer to 0 = more confident, very negative = suspect"
    )


class TranscriptionResult(BaseModel):
    """The output contract — stable JSON that downstream consumers (the Dirayah AI chatbot,
    SRT/VTT export, etc.) can rely on."""
    job_id: str
    language: str
    duration_sec: float
    model: str
    rtf: float | None = Field(
        None, description="real-time factor = processing_time / audio_duration; < 1 is faster than real time"
    )
    segments: list[Segment]
