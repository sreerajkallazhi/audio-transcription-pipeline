# Audio Transcription Pipeline

A lightweight FastAPI service that accepts audio files, normalizes them to 16 kHz mono WAV via FFmpeg, and returns structured, timestamped transcriptions using `faster-whisper` (CTranslate2).

## Engineering Choices
- **FastAPI + Pydantic:** Provides async streaming uploads, strict response schemas for downstream consumers, and automatic OpenAPI docs.
- **FFmpeg Normalization:** Decouples ingestion from inference by converting all supported codecs (MP3, WAV, M4A, FLAC, OGG, WebM) into a single canonical format (`16kHz`, mono, `pcm_s16le`).
- **faster-whisper + VAD:** Runs up to 4x faster with lower memory usage than standard PyTorch Whisper, while built-in Voice Activity Detection (`vad_filter=True`) skips silence and prevents hallucinations.
- **Memory Safety:** Streams uploads in 1MB chunks into auto-cleaned temporary directories.

## Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) installed and available in `PATH` (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Ubuntu)

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload