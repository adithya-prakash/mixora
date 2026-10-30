from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.audio import decode_audio
from backend.app.config import Settings
from backend.app.dsp import (
    ANALYSIS_SAMPLE_RATE,
    FFT_SIZE,
    SPECTRUM_CEILING_DB,
    SPECTRUM_FLOOR_DB,
    analyze_audio,
)
from backend.app.observations import build_observations

GENRES = (
    "Dubstep", "Bass House", "Pop", "Rock", "Hip Hop", "EDM", "Techno",
    "Drum & Bass", "Trap", "Lo-fi", "R&B", "Indie",
)


def analyze_bytes(
    data: bytes,
    filename: str,
    genre: str,
    settings: Settings,
    reference_data: bytes | None = None,
    reference_filename: str | None = None,
) -> dict[str, Any]:
    if genre not in GENRES:
        raise ValueError("Unsupported genre")

    decoded = decode_audio(
        data,
        filename,
        max_upload_bytes=settings.max_upload_bytes,
        max_duration_s=settings.max_duration_s,
        ffmpeg_bin=settings.ffmpeg_bin,
    )
    metrics = analyze_audio(decoded.samples)
    response: dict[str, Any] = {
        "id": f"an_{uuid4().hex}",
        "meta": {
            "filename": Path(filename).name[:255],
            "duration_s": round(decoded.duration_s, 3),
            "sample_rate": decoded.original_sample_rate,
            "channels": decoded.channels,
            "genre": genre,
            "profile": "generic-v0",
        },
        "peak": metrics["peak"],
        "rms": metrics["rms"],
        "lufs": metrics["lufs"],
        "lra": metrics["lra"],
        "crest": metrics["crest"],
        "clipping": metrics["clipping"],
        "clipping_detail": metrics["clipping_detail"],
        "bands": metrics["bands"],
        "bands_full": metrics["bands_full"],
        "spectrum": {"A": metrics["spectrum"]},
        "eq": {"A": metrics["eq"]},
        "stereo": metrics["stereo"],
        "settings": {
            "analysis_sample_rate": ANALYSIS_SAMPLE_RATE,
            "fft_size": FFT_SIZE,
            "window": "hann",
            "overlap": 0.5,
            "spectrum_db_floor": SPECTRUM_FLOOR_DB,
            "spectrum_db_ceiling": SPECTRUM_CEILING_DB,
        },
    }
    response["obs"] = build_observations(response)

    if reference_data is not None:
        if not reference_filename:
            reference_filename = "reference"
        reference = decode_audio(
            reference_data,
            reference_filename,
            max_upload_bytes=settings.max_upload_bytes,
            max_duration_s=settings.max_duration_s,
            ffmpeg_bin=settings.ffmpeg_bin,
        )
        reference_metrics = analyze_audio(reference.samples)
        response["spectrum"]["B"] = reference_metrics["spectrum"]
        response["eq"]["B"] = reference_metrics["eq"]
        response["ref"] = {
            "peak": reference_metrics["peak"],
            "rms": reference_metrics["rms"],
            "lufs": reference_metrics["lufs"],
            "lra": reference_metrics["lra"],
            "crest": reference_metrics["crest"],
            "corr": reference_metrics["stereo"]["correlation"],
            "width": reference_metrics["stereo"]["width"],
            "bands": [pair[1] for pair in reference_metrics["bands"]],
        }
    return response