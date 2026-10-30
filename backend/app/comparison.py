from __future__ import annotations

from typing import Any


def compare_analyses(previous: dict[str, Any], revised: dict[str, Any]) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for name, unit in (
        ("peak", "dBFS"),
        ("rms", "dBFS"),
        ("lufs", "LUFS"),
        ("lra", "LU"),
        ("crest", "dB"),
    ):
        before = float(previous[name])
        after = float(revised[name])
        metrics.append({
            "name": name,
            "previous": before,
            "revised": after,
            "delta": round(after - before, 2),
            "unit": unit,
        })

    bands = []
    for before, after in zip(previous["bands_full"], revised["bands_full"]):
        bands.append({
            "name": before[0],
            "previous": before[1],
            "revised": after[1],
            "delta": round(float(after[1]) - float(before[1]), 2),
        })

    previous_observations = {item["id"]: item for item in previous["obs"]}
    revised_observations = {item["id"]: item for item in revised["obs"]}
    changes = []
    for observation_id in sorted(previous_observations.keys() | revised_observations.keys()):
        if observation_id not in revised_observations:
            status = "resolved"
        elif observation_id not in previous_observations:
            status = "new"
        else:
            status = "persisted"
        changes.append({"id": observation_id, "status": status})

    return {
        "previous_id": previous["id"],
        "revised_id": revised["id"],
        "metrics": metrics,
        "bands": bands,
        "stereo": {
            "correlation_delta": round(
                revised["stereo"]["correlation"] - previous["stereo"]["correlation"], 3
            ),
            "width_delta": round(revised["stereo"]["width"] - previous["stereo"]["width"], 3),
        },
        "clipping": {
            "previous": previous["clipping"],
            "revised": revised["clipping"],
        },
        "observation_changes": changes,
        "note": "Differences are numerical only. They do not show which version sounds better.",
    }