from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int
    max_duration_s: int
    allowed_origins: tuple[str, ...]
    storage_dir: Path
    db_path: Path
    retention_days: int
    rate_limit_per_minute: int
    max_concurrent_analyses: int
    ffmpeg_bin: str

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "100"))
    max_duration_s = int(os.getenv("MAX_DURATION_S", "900"))
    retention_days = int(os.getenv("RETENTION_DAYS", "30"))
    if max_upload_mb < 1 or max_duration_s < 1 or retention_days < 1:
        raise ValueError("Upload limits and retention days must be positive integers")

    origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    )
    storage_dir = Path(os.getenv("STORAGE_DIR", str(BACKEND_DIR / "data"))).resolve()
    db_url = os.getenv("DB_URL", "")
    if db_url and not db_url.startswith("sqlite:///"):
        raise ValueError("This MVP backend supports SQLite DB_URL values only")
    if db_url:
        db_location = db_url.removeprefix("sqlite:///")
        db_path = Path(db_location)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
    else:
        db_path = storage_dir / "mixora.sqlite3"

    return Settings(
        max_upload_mb=max_upload_mb,
        max_duration_s=max_duration_s,
        allowed_origins=origins,
        storage_dir=storage_dir,
        db_path=db_path.resolve(),
        retention_days=retention_days,
        rate_limit_per_minute=max(1, int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))),
        max_concurrent_analyses=max(1, int(os.getenv("MAX_CONCURRENT_ANALYSES", "2"))),
        ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
    )