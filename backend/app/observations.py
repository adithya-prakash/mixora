from __future__ import annotations

from typing import Any


def build_observations(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []

    if metrics["clipping"]:
        detail = metrics["clipping_detail"]
        evidence = f"Peak {metrics['peak']:.2f} dBFS; {detail['count']} samples at or above 0.999 full scale"
        candidates.append((10.0, {
            "id": "clipping",
            "t": "Clipping indicators detected",
            "p": "High",
            "c": "confirmed",
            "conf": "High",
            "ev": evidence,
            "ex": "The sample-peak detector found consecutive samples at the digital limit. This indicates a measurable clipping risk, though listening is still needed to judge audibility.",
            "a": "Check the loudest section and inspect gain staging or the final limiter ceiling.",
            "evidence_fields": ["peak", "clipping_detail.count", "clipping_detail.max_run"],
        }))

    stereo = metrics["stereo"]
    if stereo["mono"] == "PROBLEM":
        candidates.append((9.0, {
            "id": "mono_compat",
            "t": "Potential mono compatibility loss",
            "p": "High",
            "c": "confirmed",
            "conf": "High",
            "ev": f"Correlation {stereo['correlation']:.3f}; mono status {stereo['mono']}",
            "ex": "The measured stereo channels show strong anti-correlation or a substantial level loss when summed to mono.",
            "a": "Audition the mix in mono and inspect polarity or unusually wide elements, especially in the low end.",
            "evidence_fields": ["stereo.correlation", "stereo.mono"],
        }))
    elif stereo["mono"] == "CAUTION":
        candidates.append((4.0, {
            "id": "mono_compat",
            "t": "Check mono compatibility",
            "p": "Medium",
            "c": "potential",
            "conf": "Low",
            "ev": f"Correlation {stereo['correlation']:.3f}; mono status {stereo['mono']}",
            "ex": "The mono-sum check raised a caution. Its audibility depends on the arrangement and playback context.",
            "a": "Compare the mix in stereo and mono and listen for elements that lose level or definition.",
            "evidence_fields": ["stereo.correlation", "stereo.mono"],
        }))

    low_share = sum(float(pair[1]) for pair in metrics["bands_full"][:2])
    if low_share >= 70.0:
        candidates.append((3.0, {
            "id": "low_end_concentration",
            "t": "Low-frequency energy concentration",
            "p": "Medium",
            "c": "potential",
            "conf": "Low",
            "ev": f"Sub-bass {metrics['bands_full'][0][1]:.2f}% · Bass {metrics['bands_full'][1][1]:.2f}% · combined {low_share:.2f}%",
            "ex": "The generic-v0 screening heuristic found a high share of measured energy below 250 Hz. This is not a calibrated genre target and may be intentional.",
            "a": "Listen to kick and bass together and check whether either masks important low-mid detail.",
            "evidence_fields": ["bands_full.0", "bands_full.1"],
        }))

    if metrics["lufs"] > -5.0 or metrics["lufs"] < -18.0:
        candidates.append((2.0, {
            "id": "loudness_screen",
            "t": "Review integrated loudness",
            "p": "Low",
            "c": "potential",
            "conf": "Low",
            "ev": f"Integrated loudness {metrics['lufs']:.2f} LUFS",
            "ex": "This falls outside a broad generic-v0 screening range. Loudness expectations depend on delivery context and genre; this is not a target level.",
            "a": "Check the intended delivery specification and audition the track at a comfortable monitoring level.",
            "evidence_fields": ["lufs"],
        }))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [observation for _, observation in candidates[:3]]