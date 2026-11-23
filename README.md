# Audio Transcription Pipeline

A lightweight, production-minded FastAPI service that accepts audio files in multiple formats, normalizes them into a canonical format, and returns structured, timestamped transcriptions ready for downstream use.

---

## 1. Overall Approach

The pipeline is designed around **separation of concerns** so each stage is independent, predictable, and easy to scale:

1. **Ingestion & Validation:** Accept audio uploads asynchronously, validate file types, and stream files to disk in chunks to protect memory.
2. **Format Normalization:** Decouple codec handling from the ML model by converting all inputs into a single canonical audio format via FFmpeg.
3. **Transcription & VAD:** Run efficient speech-to-text inference with Voice Activity Detection to filter silence and reduce hallucinations.
4. **Structured Output:** Return strictly typed JSON containing metadata, `full_text`, and segment-level timestamps for downstream consumers (e.g., RAG chunking, search indexing, or subtitles).

---

## 2. Key Design & Technical Decisions

- **FastAPI + Pydantic:** Chosen for async file handling, automatic OpenAPI/Swagger documentation (`/docs`), and strict response schema validation so downstream systems receive a reliable contract.
- **`faster-whisper` (CTranslate2) over Standard Whisper:** Delivers identical accuracy to OpenAI's PyTorch Whisper while running up to 4x faster with significantly lower RAM/VRAM usage. Configured with `int8` quantization for CPU portability (easily switched to `device="cuda"`, `compute_type="float16"` on GPU).
- **Voice Activity Detection (`vad_filter=True`):** Whisper models can hallucinate repetitive phrases during long stretches of silence. Enabling Silero VAD skips non-speech regions, improving both accuracy and latency.
- **FFmpeg Canonical Normalization:** Rather than relying on the STT library to decode dozens of codecs, all inputs (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`) are transcoded to **16 kHz, mono, 16-bit PCM WAV**—matching Whisper's native training distribution.
- **Direct PCM Loading (`wave` + `numpy`):** Once FFmpeg outputs a clean PCM WAV, reading it directly into a `float32` NumPy array avoids fragile third-party container-decoding issues (such as PyAV version mismatches on newer Python runtimes).
- **Streaming Uploads & Auto-Cleanup:** Files are written in 1 MB chunks inside a `tempfile.TemporaryDirectory()`, preventing OOM spikes on large uploads and guaranteeing temporary files are cleaned up even if an exception occurs.

---

## 3. Code Structure & Implementation

```text
audio-transcription-pipeline/
├── main.py            # FastAPI app, FFmpeg normalization, PCM loader, and faster-whisper inference
├── requirements.txt   # Pinned Python dependencies
├── .gitignore         # Excludes virtual environments and cache files
└── README.md          # Architecture, design decisions, and usage guide
