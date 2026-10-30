from __future__ import annotations

import unittest

import numpy as np

from backend.app.audio import ANALYSIS_SAMPLE_RATE
from backend.app.dsp import analyze_audio


class DspMeasurementTests(unittest.TestCase):
    def test_sine_levels_and_contract_shapes(self) -> None:
        time_axis = np.arange(ANALYSIS_SAMPLE_RATE * 4) / ANALYSIS_SAMPLE_RATE
        tone = 0.5 * np.sin(2 * np.pi * 440 * time_axis).astype(np.float32)
        result = analyze_audio(tone[:, None])
        self.assertAlmostEqual(result["peak"], -6.02, delta=0.1)
        self.assertAlmostEqual(result["rms"], -9.03, delta=0.1)
        self.assertEqual(len(result["bands"]), 4)
        self.assertEqual(len(result["bands_full"]), 5)
        self.assertAlmostEqual(sum(pair[1] for pair in result["bands_full"]), 100.0, delta=0.05)
        self.assertEqual(len(result["spectrum"]), 16)
        self.assertEqual(len(result["eq"]), 8)
        self.assertEqual(result["stereo"], {"balance": 0.0, "correlation": 1.0, "width": 0.0, "mono": "OK"})

    def test_hard_limited_sine_is_flagged(self) -> None:
        time_axis = np.arange(ANALYSIS_SAMPLE_RATE * 4) / ANALYSIS_SAMPLE_RATE
        tone = np.sin(2 * np.pi * 220 * time_axis).astype(np.float32)
        result = analyze_audio(tone[:, None])
        self.assertTrue(result["clipping"])
        self.assertGreaterEqual(result["clipping_detail"]["max_run"], 3)

    def test_antiphase_stereo_reports_mono_problem(self) -> None:
        time_axis = np.arange(ANALYSIS_SAMPLE_RATE * 2) / ANALYSIS_SAMPLE_RATE
        left = 0.5 * np.sin(2 * np.pi * 440 * time_axis)
        stereo = np.column_stack((left, -left)).astype(np.float32)
        result = analyze_audio(stereo)
        self.assertAlmostEqual(result["stereo"]["correlation"], -1.0, delta=0.001)
        self.assertEqual(result["stereo"]["mono"], "PROBLEM")


if __name__ == "__main__":
    unittest.main()