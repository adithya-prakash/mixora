from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from scipy.io import wavfile

from backend.app.config import Settings
from backend.app.main import create_app


def wav_bytes(frequency: float = 440.0, duration_s: float = 4.0) -> bytes:
    sample_rate = 44_100
    time_axis = np.arange(int(sample_rate * duration_s)) / sample_rate
    samples = (0.25 * np.sin(2 * np.pi * frequency * time_axis) * 32767).astype(np.int16)
    stream = io.BytesIO()
    wavfile.write(stream, sample_rate, samples)
    return stream.getvalue()


class AnalysisApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        settings = Settings(
            max_upload_mb=100,
            max_duration_s=900,
            allowed_origins=("http://localhost:5173",),
            storage_dir=root,
            db_path=root / "test.sqlite3",
            retention_days=30,
            rate_limit_per_minute=100,
            max_concurrent_analyses=2,
            ffmpeg_bin="ffmpeg",
        )
        self.client = TestClient(create_app(settings))
        self.client.__enter__()
        self.schema = json.loads(
            (Path(__file__).resolve().parents[2] / "api-contract-v1.schema.json").read_text(encoding="utf-8")
        )
        self.validator = Draft202012Validator(self.schema)

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self.tempdir.cleanup()

    def analyze(self, data: bytes, reference: bytes | None = None):
        files = {"file": ("tone.wav", data, "audio/wav")}
        if reference is not None:
            files["reference"] = ("ref.wav", reference, "audio/wav")
        return self.client.post("/analyze", data={"genre": "Dubstep"}, files=files)

    def test_analyze_without_reference_matches_contract(self) -> None:
        response = self.analyze(wav_bytes())
        self.assertEqual(response.status_code, 200, response.text)
        self.validator.validate(response.json())
        self.assertNotIn("ref", response.json())
        self.assertNotIn("B", response.json()["spectrum"])
        self.assertEqual(response.json()["meta"]["profile"], "generic-v0")

    def test_analyze_with_reference_matches_contract(self) -> None:
        response = self.analyze(wav_bytes(), wav_bytes(880.0))
        self.assertEqual(response.status_code, 200, response.text)
        self.validator.validate(response.json())
        self.assertIn("ref", response.json())
        self.assertEqual(len(response.json()["spectrum"]["B"]), 16)

    def test_reload_compare_and_grounded_voice(self) -> None:
        original_response = self.analyze(wav_bytes())
        analysis_id = original_response.json()["id"]
        reload_response = self.client.get(f"/analysis/{analysis_id}")
        self.assertEqual(reload_response.status_code, 200)
        voice_response = self.client.post(
            "/voice/ask",
            json={"analysis_id": analysis_id, "question": "What is the loudness?"},
        )
        self.assertEqual(voice_response.status_code, 200)
        self.assertIn("LUFS", voice_response.json()["text"])
        compare_response = self.client.post(
            "/compare",
            data={"previous_id": analysis_id},
            files={"file": ("revision.wav", wav_bytes(220.0), "audio/wav")},
        )
        self.assertEqual(compare_response.status_code, 200, compare_response.text)
        self.assertIn("numerical only", compare_response.json()["note"])

    def test_invalid_upload_uses_error_envelope(self) -> None:
        response = self.analyze(b"invalid audio")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "unsupported_format")
        self.assertIn("message", response.json()["error"])

    def test_missing_analysis_returns_not_found_envelope(self) -> None:
        response = self.client.get("/analysis/not-a-real-id")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()