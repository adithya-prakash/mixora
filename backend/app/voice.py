from __future__ import annotations

from typing import Any


def answer_question(analysis: dict[str, Any], question: str) -> dict[str, Any]:
    normalized = question.casefold()
    observations = {item["id"]: item for item in analysis.get("obs", [])}
    used_fields: list[str] = []

    if any(term in normalized for term in ("clip", "distort", "peak")):
        used_fields.extend(["peak", "clipping", "clipping_detail"])
        if "clipping" in observations:
            observation = observations["clipping"]
            text = (
                f"The clipping check is flagged. The measured sample peak is {analysis['peak']:.2f} dBFS, "
                f"with {analysis['clipping_detail']['count']} samples at or above 0.999 full scale. "
                "Inspect the loudest section and gain staging; the measurement alone does not establish how audible it is."
            )
        else:
            text = (
                f"The sample peak is {analysis['peak']:.2f} dBFS and the clipping detector did not flag consecutive samples at the digital limit. "
                "This check does not include a true-peak measurement."
            )
    elif any(term in normalized for term in ("sub", "bass", "low end", "low-end")):
        used_fields.append("bands_full")
        sub = analysis["bands_full"][0][1]
        bass = analysis["bands_full"][1][1]
        observation = observations.get("low_end_concentration")
        if observation:
            used_fields.extend(observation["evidence_fields"])
            text = (
                f"The measured shares are {sub:.2f}% sub-bass and {bass:.2f}% bass. "
                "The generic-v0 screen marks this as a potential check, not a genre-calibrated target. "
                "Listen to kick and bass together and check whether either masks low-mid detail."
            )
        else:
            text = (
                f"The measured shares are {sub:.2f}% sub-bass and {bass:.2f}% bass. "
                "No low-end observation was triggered by the generic-v0 screen; that screen is not a calibrated genre profile."
            )
    elif any(term in normalized for term in ("stereo", "mono", "width", "correlation", "phase")):
        stereo = analysis["stereo"]
        used_fields.append("stereo")
        text = (
            f"Stereo correlation is {stereo['correlation']:.3f}, width is {stereo['width']:.3f}, "
            f"and the mono check is {stereo['mono']}. {stereo['mono'].capitalize()} is a measurement status; "
            "audition mono playback to judge the musical effect."
        )
    elif any(term in normalized for term in ("loud", "lufs", "dynamic", "crest", "range")):
        used_fields.extend(["lufs", "lra", "crest"])
        text = (
            f"Integrated loudness is {analysis['lufs']:.2f} LUFS, loudness range is {analysis['lra']:.2f} LU, "
            f"and crest factor is {analysis['crest']:.2f} dB. Compare these with the intended delivery context; "
            "the analysis does not rate artistic quality."
        )
    else:
        used_fields.extend(["meta.genre", "peak", "lufs", "stereo.mono"])
        text = (
            f"This {analysis['meta']['genre']} analysis reports a sample peak of {analysis['peak']:.2f} dBFS, "
            f"integrated loudness of {analysis['lufs']:.2f} LUFS, and mono status {analysis['stereo']['mono']}. "
            "Genre interpretation uses a generic-v0 profile, so treat potential findings as prompts to listen rather than standards."
        )
    return {"text": text, "used_fields": sorted(set(used_fields))}