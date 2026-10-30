from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.app.audio import AudioInputError
from backend.app.comparison import compare_analyses
from backend.app.config import Settings, load_settings
from backend.app.service import GENRES, analyze_bytes
from backend.app.store import AnalysisStore
from backend.app.voice import answer_question

logger = logging.getLogger("mixora.api")


class VoiceTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(min_length=1, max_length=2_000)


class VoiceAskRequest(BaseModel):
    analysis_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=1_000)
    history: list[VoiceTurn] = Field(default_factory=list, max_length=20)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise AudioInputError("too_large", "File exceeds the configured upload limit.", 413)
    if not data:
        raise AudioInputError("decode_failed", "The uploaded file is empty.", 422)
    return data


def _clean_filename(upload: UploadFile) -> str:
    return Path(upload.filename or "audio").name[:255]


def create_app(settings: Settings | None = None, store: AnalysisStore | None = None) -> FastAPI:
    configuration = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or AnalysisStore(configuration.db_path)
        app.state.store.purge_expired(configuration.retention_days)
        app.state.analysis_semaphore = asyncio.Semaphore(configuration.max_concurrent_analyses)
        app.state.rate_records = {}
        yield

    application = FastAPI(title="Mixora Analysis API", version="1.0.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(configuration.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    # Compress large JSON responses — spectrum + EQ arrays compress ~70%
    application.add_middleware(GZipMiddleware, minimum_size=1024)

    @application.middleware("http")
    async def rate_limit(request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path == "/health" or request.url.path.startswith("/docs"):
            return await call_next(request)
        now = time.monotonic()
        client = request.client.host if request.client else "unknown"
        records: dict[str, tuple[float, int]] = request.app.state.rate_records
        if len(records) > 10_000:
            records = {ip: record for ip, record in records.items() if now - record[0] < 60.0}
            request.app.state.rate_records = records
        start, count = records.get(client, (now, 0))
        if now - start >= 60.0:
            start, count = now, 0
        if count >= configuration.rate_limit_per_minute:
            return _error(429, "rate_limited", "Too many requests. Please try again shortly.")
        records[client] = (start, count + 1)
        return await call_next(request)

    @application.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        details = exc.errors()
        genre_error = any("genre" in error.get("loc", ()) for error in details)
        if genre_error:
            return _error(400, "invalid_genre", "Choose a supported genre.")
        return _error(400, "invalid_request", "The request is missing a required field or has invalid data.")

    @application.exception_handler(AudioInputError)
    async def audio_error(_: Request, exc: AudioInputError):
        return _error(exc.status_code, exc.code, exc.message)

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/analyze")
    async def analyze(
        request: Request,
        file: UploadFile = File(...),
        reference: UploadFile | None = File(default=None),
        genre: str = Form(...),
    ):
        if genre not in GENRES:
            return _error(400, "invalid_genre", "Choose a supported genre.")
        try:
            mix_bytes = await _read_limited(file, configuration.max_upload_bytes)
            reference_bytes = (
                await _read_limited(reference, configuration.max_upload_bytes) if reference else None
            )
            async with request.app.state.analysis_semaphore:
                response = await run_in_threadpool(
                    analyze_bytes,
                    mix_bytes,
                    _clean_filename(file),
                    genre,
                    configuration,
                    reference_bytes,
                    _clean_filename(reference) if reference else None,
                )
            await run_in_threadpool(request.app.state.store.save_analysis, response)
            return response
        except AudioInputError:
            raise
        except Exception as exc:
            logger.exception("Audio analysis failed")
            return _error(500, "analysis_failed", "The audio could not be analyzed.")
        finally:
            await file.close()
            if reference:
                await reference.close()

    @application.post("/compare")
    async def compare(
        request: Request,
        previous_id: str = Form(...),
        file: UploadFile = File(...),
        genre: str | None = Form(default=None),
    ):
        previous = await run_in_threadpool(request.app.state.store.get_analysis, previous_id)
        if previous is None:
            return _error(404, "not_found", "The previous analysis was not found or has expired.")
        selected_genre = genre or previous["meta"]["genre"]
        if selected_genre not in GENRES:
            return _error(400, "invalid_genre", "Choose a supported genre.")
        try:
            revised_bytes = await _read_limited(file, configuration.max_upload_bytes)
            async with request.app.state.analysis_semaphore:
                revised = await run_in_threadpool(
                    analyze_bytes,
                    revised_bytes,
                    _clean_filename(file),
                    selected_genre,
                    configuration,
                )
            await run_in_threadpool(request.app.state.store.save_analysis, revised)
            comparison = compare_analyses(previous, revised)
            await run_in_threadpool(request.app.state.store.save_comparison, comparison)
            return comparison
        except AudioInputError:
            raise
        except Exception:
            logger.exception("Revision comparison failed")
            return _error(500, "analysis_failed", "The revised audio could not be analyzed.")
        finally:
            await file.close()

    @application.get("/analysis/{analysis_id}")
    async def get_analysis(request: Request, analysis_id: str):
        response = await run_in_threadpool(request.app.state.store.get_analysis, analysis_id)
        if response is None:
            return _error(404, "not_found", "Analysis not found or expired.")
        # Analysis JSON is immutable once saved — safe to cache at CDN/browser edge
        return JSONResponse(
            content=response,
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    @application.post("/voice/ask")
    async def voice_ask(request: Request, payload: VoiceAskRequest):
        analysis = await run_in_threadpool(request.app.state.store.get_analysis, payload.analysis_id)
        if analysis is None:
            return _error(404, "not_found", "Analysis not found or expired.")
        answer = answer_question(analysis, payload.question.strip())
        return answer

    return application


app = create_app()
