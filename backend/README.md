# Mixora Backend

FastAPI service implementing the Mixora analysis contract, DSP measurements, evidence-based observations, optional reference analysis, saved analysis retrieval, revision comparison, and text-only grounded questions.

## Requirements

- Python 3.11 or newer
- FFmpeg and ffprobe on `PATH` for MP3 input; PCM/float WAV works without them
- Windows PowerShell, macOS, or Linux

From the workspace root:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements-dev.txt
python -m uvicorn backend.app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive docs are at `/docs`. Run tests from the workspace root with `python -m unittest discover -s backend/tests -v`.

## Endpoints

- `GET /health`
- `POST /analyze` with `file`, optional `reference`, and `genre` multipart fields
- `GET /analysis/{id}` to retrieve an analysis while it remains within the retention window
- `POST /compare` with `previous_id`, revised `file`, and optional `genre`
- `POST /voice/ask` with `analysis_id` and a text `question`

`/compare` reports numerical changes only. `/voice/ask` is deterministic and text-only; it does not call an LLM, perform speech recognition, or return audio. The prototype's browser speech input can send recognized text to the endpoint.

## Audio and feedback limitations

Only mono and stereo WAV/MP3 are accepted. Uploads are limited to 100 MB each and 15 minutes. MP3 support requires FFmpeg plus ffprobe. Uploaded audio stays in memory and is not retained; analysis JSON is stored in SQLite and expires after 30 days by default.

No licensed or consented genre measurement corpus exists in this workspace. Therefore every response sets `meta.profile` to `generic-v0`; potential genre-related observations have low confidence and explicitly avoid claiming genre-calibrated ranges. This backend does not produce validated Dubstep or Bass House profiles yet. Collect and validate suitable data before changing that label.

The EQ curve is a deterministic fit against a generic neutral curve. It is descriptive, not corrective advice. Sample clipping and mono compatibility checks are objective signals, but listening remains necessary to judge audibility. True peak is not currently emitted.

Configure `ALLOWED_ORIGINS`, `MAX_UPLOAD_MB`, `MAX_DURATION_S`, `DB_URL`, `STORAGE_DIR`, `RETENTION_DAYS`, `RATE_LIMIT_PER_MINUTE`, `MAX_CONCURRENT_ANALYSES`, and `FFMPEG_BIN` through environment variables. SQLite is the supported database for this MVP. Do not expose the default development CORS origins as a production deployment policy.