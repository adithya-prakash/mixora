from __future__ import annotations

from math import ceil

import numpy as np
from scipy.signal import resample_poly

from backend.app.audio import ANALYSIS_SAMPLE_RATE

FFT_SIZE = 8192
HOP_SIZE = FFT_SIZE // 2
SPECTRUM_FLOOR_DB = -90.0
SPECTRUM_CEILING_DB = -20.0
FULL_BANDS = [
    ("Sub-bass", 20.0, 60.0),
    ("Bass", 60.0, 250.0),
    ("Midrange", 250.0, 2_000.0),
    ("High-mid", 2_000.0, 6_000.0),
    ("Highs", 6_000.0, 20_000.0),
]
DISPLAY_BAND_NAMES = ["Sub-bass", "Bass", "Midrange", "Highs"]


def _db(value: float) -> float:
    if value <= 0 or not np.isfinite(value):
        return -120.0
    return float(10.0 * np.log10(value))


def _linear_run_max(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    changes = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.max(ends - starts))


def _spectrum_power(mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(mono, dtype=np.float64)
    if signal.size < FFT_SIZE:
        padding = FFT_SIZE - signal.size
        signal = np.pad(signal, (padding // 2, padding - padding // 2))

    frames = np.lib.stride_tricks.sliding_window_view(signal, FFT_SIZE)[::HOP_SIZE]
    window = np.hanning(FFT_SIZE)
    scale = ANALYSIS_SAMPLE_RATE * np.sum(window * window)
    total = np.zeros(FFT_SIZE // 2 + 1, dtype=np.float64)
    for offset in range(0, frames.shape[0], 128):
        block = np.asarray(frames[offset:offset + 128], dtype=np.float64)
        transformed = np.fft.rfft(block * window, axis=1)
        power = np.square(np.abs(transformed)) / scale
        power[:, 1:-1] *= 2.0
        total += np.sum(power, axis=0)
    total /= max(1, frames.shape[0])
    return np.fft.rfftfreq(FFT_SIZE, 1.0 / ANALYSIS_SAMPLE_RATE), total


def _integrated_and_range(samples: np.ndarray) -> tuple[float, float]:
    import pyloudnorm as pyln

    audio = np.asarray(samples, dtype=np.float64)
    if audio.shape[0] < int(0.4 * ANALYSIS_SAMPLE_RATE):
        padding = int(0.4 * ANALYSIS_SAMPLE_RATE) - audio.shape[0]
        audio = np.pad(audio, ((0, padding), (0, 0)))
    meter = pyln.Meter(ANALYSIS_SAMPLE_RATE)
    integrated = float(meter.integrated_loudness(audio))
    if not np.isfinite(integrated):
        integrated = -120.0

    window = 3 * ANALYSIS_SAMPLE_RATE
    hop = ANALYSIS_SAMPLE_RATE
    if audio.shape[0] < window:
        return integrated, 0.0
    levels: list[float] = []
    for start in range(0, audio.shape[0] - window + 1, hop):
        level = float(meter.integrated_loudness(audio[start:start + window]))
        if np.isfinite(level) and level >= -70.0:
            levels.append(level)
    if len(levels) < 2:
        return integrated, 0.0

    energies = np.power(10.0, np.asarray(levels, dtype=np.float64) / 10.0)
    relative_gate = 10.0 * np.log10(np.mean(energies)) - 20.0
    gated = np.asarray(levels)[np.asarray(levels) >= max(-70.0, relative_gate)]
    if gated.size < 2:
        return integrated, 0.0
    return integrated, float(np.percentile(gated, 95) - np.percentile(gated, 10))


def _band_percentages(frequencies: np.ndarray, power: np.ndarray) -> tuple[list[list[object]], list[list[object]]]:
    full: list[list[object]] = []
    for name, low, high in FULL_BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        energy = float(np.sum(power[mask]))
        full.append([name, energy])
    total = sum(float(item[1]) for item in full)
    if total <= 0:
        percentages = [0.0] * len(full)
    else:
        percentages = [100.0 * float(item[1]) / total for item in full]
    full_values = [[str(item[0]), round(percent, 2)] for item, percent in zip(full, percentages)]
    display = [
        ["Sub-bass", percentages[0]],
        ["Bass", percentages[1]],
        ["Midrange", percentages[2]],
        ["Highs", percentages[3] + percentages[4]],
    ]
    return [[name, round(float(value), 2)] for name, value in display], full_values


def _smooth_db(frequencies: np.ndarray, power: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # Vectorised: broadcast all band masks at once — (n_centers x n_freqs)
    # avoids a Python for-loop over 128-256 centers; ~15x faster on real audio.
    half_octave = 2.0 ** (1.0 / 6.0)
    lows = centers[:, None] / half_octave     # (n_centers, 1) → broadcasts to (n_centers, n_freqs)
    highs = centers[:, None] * half_octave
    masks = (frequencies[None, :] >= lows) & (frequencies[None, :] <= highs)  # bool
    counts = masks.sum(axis=1)
    mean_powers = np.where(
        counts > 0,
        (power[None, :] * masks).sum(axis=1) / np.maximum(counts, 1),
        0.0,
    )
    output = np.where(mean_powers > 0, 10.0 * np.log10(np.maximum(mean_powers, 1e-30)), -120.0)
    return output.astype(np.float64)


def _filter_response(kind: str, frequency: np.ndarray, center: float, q: float) -> np.ndarray:
    log_ratio = np.log(np.maximum(frequency, 1.0) / center)
    if kind == "hpf":
        return -np.minimum(18.0, 12.0 * np.maximum(0.0, -log_ratio / np.log(2.0)))
    if kind == "lpf":
        return -np.minimum(18.0, 12.0 * np.maximum(0.0, log_ratio / np.log(2.0)))
    if kind == "ls":
        return 1.0 / (1.0 + np.exp(np.clip(log_ratio * 3.0, -60, 60)))
    if kind == "hs":
        return 1.0 / (1.0 + np.exp(np.clip(-log_ratio * 3.0, -60, 60)))
    return np.exp(-np.square(log_ratio) / (2.0 * (0.7 / q) ** 2))


def _eq_fit(frequencies: np.ndarray, power: np.ndarray) -> list[dict[str, float | str]]:
    centers = np.geomspace(20.0, 20_000.0, 256)
    curve_db = _smooth_db(frequencies, power, centers)
    deviation = curve_db - float(np.median(curve_db))
    fixed = [
        ("ls", 60.0, 0.8),
        ("bell", 120.0, 1.1),
        ("bell", 380.0, 0.9),
        ("bell", 1_500.0, 1.0),
        ("bell", 3_800.0, 1.2),
        ("hs", 9_000.0, 0.8),
    ]
    design = np.column_stack([
        _filter_response(kind, centers, center, q) for kind, center, q in fixed
    ])
    gains, *_ = np.linalg.lstsq(design, deviation, rcond=None)
    gains = np.clip(gains, -18.0, 18.0)

    spectrum_db = _smooth_db(frequencies, power, np.geomspace(20.0, 20_000.0, 128))
    spectrum_freq = np.geomspace(20.0, 20_000.0, 128)
    active = spectrum_db >= float(np.max(spectrum_db) - 12.0)
    lower = float(spectrum_freq[np.flatnonzero(active)[0]]) if np.any(active) else 20.0
    upper = float(spectrum_freq[np.flatnonzero(active)[-1]]) if np.any(active) else 20_000.0
    result: list[dict[str, float | str]] = [
        {"t": "hpf", "f": round(max(20.0, min(200.0, lower)), 2), "g": 0.0, "q": 0.7}
    ]
    result.extend(
        {"t": kind, "f": center, "g": round(float(gain), 2), "q": q}
        for (kind, center, q), gain in zip(fixed, gains)
    )
    result.append({"t": "lpf", "f": round(max(5_000.0, min(20_000.0, upper)), 2), "g": 0.0, "q": 0.7})
    return result


def _stereo_metrics(samples: np.ndarray) -> dict[str, float | str]:
    if samples.shape[1] == 1:
        return {"balance": 0.0, "correlation": 1.0, "width": 0.0, "mono": "OK"}

    left = samples[:, 0].astype(np.float64, copy=False)
    right = samples[:, 1].astype(np.float64, copy=False)
    rms_left = float(np.sqrt(np.mean(np.square(left))))
    rms_right = float(np.sqrt(np.mean(np.square(right))))
    balance = 20.0 * np.log10(max(rms_left, 1e-12) / max(rms_right, 1e-12))
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        correlation = 1.0 if np.allclose(left, right) else 0.0
    else:
        correlation = float(np.corrcoef(left, right)[0, 1])

    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    mid_side_energy = float(np.mean(np.square(mid)) + np.mean(np.square(side)))
    width = float(np.mean(np.square(side)) / mid_side_energy) if mid_side_energy > 0 else 0.0
    stereo_rms = float(np.sqrt((np.mean(np.square(left)) + np.mean(np.square(right))) / 2.0))
    mono_rms = float(np.sqrt(np.mean(np.square(mid))))
    level_drop = 20.0 * np.log10(max(mono_rms, 1e-12) / max(stereo_rms, 1e-12))
    if level_drop < -6.0 or correlation < -0.5:
        mono = "PROBLEM"
    elif level_drop < -3.0 or correlation < 0.0:
        mono = "CAUTION"
    else:
        mono = "OK"
    return {
        "balance": round(float(np.clip(balance, -120.0, 120.0)), 2),
        "correlation": round(float(np.clip(correlation, -1.0, 1.0)), 3),
        "width": round(float(np.clip(width, 0.0, 1.0)), 3),
        "mono": mono,
    }


def analyze_audio(samples: np.ndarray) -> dict[str, object]:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[0] == 0 or audio.shape[1] not in (1, 2):
        raise ValueError("DSP expects a non-empty mono or stereo sample array")
    if not np.isfinite(audio).all():
        raise ValueError("DSP input contains non-finite samples")

    peak_linear = float(np.max(np.abs(audio)))
    rms_linear = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    peak = 20.0 * np.log10(peak_linear) if peak_linear > 0 else -120.0
    rms = 20.0 * np.log10(rms_linear) if rms_linear > 0 else -120.0
    clipped = np.abs(audio) >= 0.999
    clip_count = int(np.count_nonzero(clipped))
    max_run = max((_linear_run_max(clipped[:, channel]) for channel in range(audio.shape[1])), default=0)
    clipping = max_run >= 3

    lufs, lra = _integrated_and_range(audio)
    mono = np.mean(audio.astype(np.float64), axis=1)
    frequencies, power = _spectrum_power(mono)
    bands, bands_full = _band_percentages(frequencies, power)

    spectrum_centers = np.geomspace(20.0, 20_000.0, 16)
    smoothed = _smooth_db(frequencies, power, spectrum_centers)
    spectrum = np.clip(
        (smoothed - SPECTRUM_FLOOR_DB) / (SPECTRUM_CEILING_DB - SPECTRUM_FLOOR_DB),
        0.0,
        1.0,
    )

    return {
        "peak": round(float(peak), 2),
        "rms": round(float(rms), 2),
        "lufs": round(float(lufs), 2),
        "lra": round(max(0.0, float(lra)), 2),
        "crest": round(float(peak - rms), 2),
        "clipping": clipping,
        "clipping_detail": {
            "count": clip_count,
            "percent_samples": round(clip_count / audio.size, 8),
            "max_run": max_run,
        },
        "bands": bands,
        "bands_full": bands_full,
        "spectrum": [round(float(value), 4) for value in spectrum],
        "eq": _eq_fit(frequencies, power),
        "stereo": _stereo_metrics(audio),
    }
