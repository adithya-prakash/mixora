from __future__ import annotations

import io
import json
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from backend.app.audio import AudioInputError, decode_audio


def wav_bytes(sample_rate: int = 22_050, channels: int = 1, duration_s: float = 1.0) -> bytes:
    frames = int(sample_rate * duration_s)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    signal = (0.25 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    if channels == 2:
        signal = np.column_stack((signal, signal))
    stream = io.BytesIO()
    wavfile.write(stream, sample_rate, signal)
    return stream.getvalue()


class AudioDecodeTests(unittest.TestCase):
    def test_decodes_and_resamples_wav_without_trusting_filename(self) -> None:
        result = decode_audio(wav_bytes(), "not-an-audio.extension")
        self.assertEqual(result.original_sample_rate, 22_050)
        self.assertEqual(result.channels, 1)
        self.assertEqual(result.samples.shape, (44_100, 1))
        self.assertEqual(result.samples.dtype, np.float32)
        self.assertAlmostEqual(result.duration_s, 1.0)

    def test_preserves_stereo_shape(self) -> None:
        result = decode_audio(wav_bytes(44_100, channels=2), "mix.wav")
        self.assertEqual(result.samples.shape, (44_100, 2))

    def test_rejects_upload_over_limit(self) -> None:
        with self.assertRaises(AudioInputError) as raised:
            decode_audio(wav_bytes(), "mix.wav", max_upload_bytes=10)
        self.assertEqual(raised.exception.code, "too_large")
        self.assertEqual(raised.exception.status_code, 413)

    def test_rejects_unsupported_magic_bytes(self) -> None:
        with self.assertRaises(AudioInputError) as raised:
            decode_audio(b"not audio", "mix.wav")
        self.assertEqual(raised.exception.code, "unsupported_format")

    def test_reports_missing_mp3_decoder_clearly(self) -> None:
        with patch("backend.app.audio.shutil.which", return_value=None):
            with self.assertRaises(AudioInputError) as raised:
                decode_audio(b"ID3" + bytes(32), "mix.mp3")
        self.assertEqual(raised.exception.code, "decode_failed")
        self.assertIn("FFmpeg", raised.exception.message)

    def test_mp3_uses_decoded_duration_when_probe_omits_duration(self) -> None:
        probe = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"streams": [{"sample_rate": "44100", "channels": 2}], "format": {}}).encode(),
            stderr=b"",
        )
        decoded = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=np.zeros((44_100, 2), dtype="<f4").tobytes(),
            stderr=b"",
        )
        with (
            patch("backend.app.audio.shutil.which", side_effect=lambda name: name),
            patch("backend.app.audio._run_process", side_effect=[probe, decoded]),
        ):
            result = decode_audio(b"ID3" + bytes(32), "mix.mp3")

        self.assertAlmostEqual(result.duration_s, 1.0)
        self.assertEqual(result.samples.shape, (44_100, 2))


if __name__ == "__main__":
    unittest.main()
