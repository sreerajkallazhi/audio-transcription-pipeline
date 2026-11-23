import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pydantic import BaseModel

app = FastAPI(title="Audio Transcription Pipeline")

MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    filename: str
    language: str
    language_probability: float
    duration: float
    full_text: str
    segments: List[TranscriptSegment]


def normalize_audio(input_path: str, output_path: str) -> None:
    """Convert supported audio formats to 16kHz mono 16-bit PCM WAV."""
    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.decode()}")

def load_wav_as_float32(wav_path: str) -> np.ndarray:
    """Read 16kHz mono 16-bit PCM WAV into a float32 NumPy array in [-1.0, 1.0]."""
    with wave.open(wav_path, "rb") as wf:
        pcm_data = wf.readframes(wf.getnframes())
    return np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {sorted(ALLOWED_EXTENSIONS)}",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = os.path.join(tmp_dir, f"input{ext}")
        wav_path = os.path.join(tmp_dir, "normalized.wav")

        # Stream upload in 1MB chunks to avoid memory spikes on large files
        with open(raw_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        try:
            normalize_audio(raw_path, wav_path)
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))

        audio_array = load_wav_as_float32(wav_path)

        segments_gen, info = model.transcribe(
            audio_array,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments = []
        full_text_parts = []
        for idx, seg in enumerate(segments_gen):
            clean_text = seg.text.strip()
            segments.append(
                TranscriptSegment(
                    id=idx,
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=clean_text,
                )
            )
            full_text_parts.append(clean_text)

        return TranscriptionResponse(
            filename=file.filename,
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration=round(info.duration, 2),
            full_text=" ".join(full_text_parts),
            segments=segments,
        )