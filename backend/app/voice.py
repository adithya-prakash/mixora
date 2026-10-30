from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("mixora.voice")


def _ask_gemini(analysis: dict[str, Any], question: str, api_key: str) -> str | None:
    """Query Google Gemini 2.0 / 1.5 Flash for audio mixing insights using standard library urllib."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    genre = analysis.get("meta", {}).get("genre", "Unknown")
    lufs = analysis.get("lufs", "N/A")
    peak = analysis.get("peak", "N/A")
    crest = analysis.get("crest", "N/A")
    bands = analysis.get("bands", [])
    stereo = analysis.get("stereo", {})
    clipping = analysis.get("clipping", False)
    observations = [o.get("title", "") for o in analysis.get("obs", [])]

    context = (
        f"Track Genre: {genre}\n"
        f"Measured Peak: {peak} dBFS\n"
        f"Integrated Loudness: {lufs} LUFS\n"
        f"Crest Factor (Dynamics): {crest} dB\n"
        f"Frequency Bands: {bands}\n"
        f"Stereo Correlation: {stereo.get('correlation')} (Mono status: {stereo.get('mono')}, Width: {stereo.get('width')})\n"
        f"Clipping Flagged: {clipping}\n"
        f"Key Observations: {', '.join(observations) if observations else 'None'}\n"
    )

    system_instruction = (
        "You are Mixora, a professional mixing and mastering engineer AI. "
        "You are in a live spoken conversation with a music producer about their track. "
        "Use the provided measurements to answer their question with specific, actionable mixing advice. "
        "Keep your response strictly to 2-4 conversational sentences so it sounds natural when spoken aloud. "
        "Do not use markdown formatting (no asterisks, bullet points, or bold text) because your answer will be read by text-to-speech."
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"System Context:\n{system_instruction}\n\nTrack Analysis:\n{context}\n\nProducer's Question: {question}"}
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 200,
        },
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
    except Exception as exc:
        logger.warning("Gemini voice query failed: %s", exc)

    return None


def answer_question(analysis: dict[str, Any], question: str) -> dict[str, Any]:
    # 1. If GEMINI_API_KEY is configured in the environment, use Gemini AI
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        ai_response = _ask_gemini(analysis, question, gemini_key)
        if ai_response:
            return {"text": ai_response, "used_fields": ["ai_gemini", "lufs", "peak", "bands", "stereo"]}

    # 2. Deterministic smart fallback when no API key is provided
    normalized = question.casefold()
    observations = {item["id"]: item for item in analysis.get("obs", [])}
    used_fields: list[str] = []

    if any(term in normalized for term in ("clip", "distort", "peak")):
        used_fields.extend(["peak", "clipping", "clipping_detail"])
        if "clipping" in observations:
            text = (
                f"The clipping check is flagged with a sample peak of {analysis['peak']:.2f} dBFS and "
                f"{analysis['clipping_detail']['count']} clipped samples. Inspect your gain staging and pull back the limiter ceiling."
            )
        else:
            text = (
                f"Your peak is at {analysis['peak']:.2f} dBFS with no consecutive digital clipping detected. "
                "Be sure to check inter-sample true peaks before export."
            )
    elif any(term in normalized for term in ("sub", "bass", "low end", "low-end")):
        used_fields.append("bands_full")
        sub = analysis["bands_full"][0][1]
        bass = analysis["bands_full"][1][1]
        text = (
            f"Your low end consists of {sub:.1f}% sub-bass and {bass:.1f}% bass. "
            "If your sub feels muddy or overwhelming, try sidechaining your kick and bass or cleaning below thirty Hertz."
        )
    elif any(term in normalized for term in ("stereo", "mono", "width", "correlation", "phase")):
        stereo = analysis["stereo"]
        used_fields.append("stereo")
        text = (
            f"Your stereo correlation is {stereo['correlation']:.2f} with a width rating of {stereo['width']:.2f}. "
            f"Mono compatibility status is {stereo['mono']}. Check your side channels to ensure elements do not cancel out in mono."
        )
    elif any(term in normalized for term in ("loud", "lufs", "dynamic", "crest", "range")):
        used_fields.extend(["lufs", "lra", "crest"])
        text = (
            f"Integrated loudness is sitting at {analysis['lufs']:.1f} LUFS with a crest factor of {analysis['crest']:.1f} dB. "
            "This gives you a good balance of impact and dynamic range for streaming distribution."
        )
    else:
        genre = analysis.get("meta", {}).get("genre", "your track")
        text = (
            f"For this {genre} mix, loudness is {analysis['lufs']:.1f} LUFS, peak is {analysis['peak']:.2f} dBFS, "
            f"and mono status is {analysis['stereo']['mono']}. Ask me specifically about low end, loudness, or stereo width for targeted advice."
        )

    return {"text": text, "used_fields": sorted(set(used_fields))}
