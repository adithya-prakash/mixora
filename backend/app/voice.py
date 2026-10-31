from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("mixora.voice")

# ---------------------------------------------------------------------------
# Build a rich context string from the analysis data so LLM has full facts
# ---------------------------------------------------------------------------

def _build_context(analysis: dict[str, Any]) -> str:
    meta = analysis.get("meta", {})
    stereo = analysis.get("stereo", {})
    bands = analysis.get("bands", [])
    bands_full = analysis.get("bands_full", [])
    clipping = analysis.get("clipping", False)
    clip_detail = analysis.get("clipping_detail", {})
    observations = analysis.get("obs", [])

    obs_lines = "\n".join(
        f"  - [{o.get('confidence','?').upper()}] {o.get('title','')}: {o.get('detail','')}"
        for o in observations
    ) or "  None flagged (mix is technically within standard ranges)."

    bands_str = ", ".join(f"{b[0]}: {b[1]:.1f}%" for b in bands) if bands else "N/A"
    bands_full_str = ", ".join(f"{b[0]}: {b[1]:.1f}%" for b in bands_full) if bands_full else "N/A"

    return f"""=== TRACK TECHNICAL ANALYSIS ===
Target Genre: {meta.get('genre', 'Unknown')}
Filename: {meta.get('filename', 'Unknown')}
Duration: {meta.get('duration_s', '?')} seconds
Channels: {meta.get('channels', '?')}

LOUDNESS & DYNAMICS:
- Integrated Loudness: {analysis.get('lufs', 'N/A')} LUFS
- Peak: {analysis.get('peak', 'N/A')} dBFS
- RMS: {analysis.get('rms', 'N/A')} dBFS
- Dynamic Range (LRA): {analysis.get('lra', 'N/A')} LU
- Crest Factor: {analysis.get('crest', 'N/A')} dB

CLIPPING:
- Clipping detected: {clipping}
- Clipped sample count: {clip_detail.get('count', 0)}
- Longest consecutive clip: {clip_detail.get('max_run', 0)} samples

FREQUENCY SPECTRUM ENERGY:
- Main Bands: {bands_str}
- Detailed Breakdown: {bands_full_str}

STEREO FIELD:
- Correlation: {stereo.get('correlation', 'N/A')} (-1.0 = out of phase, 0.0 = wide stereo, +1.0 = pure mono)
- Stereo Width: {stereo.get('width', 'N/A')} (0.0 = mono, 1.0 = wide)
- Balance L/R: {stereo.get('balance', 'N/A')} dB
- Mono Compatibility: {stereo.get('mono', 'N/A')}

FLAGGED OBSERVATIONS:
{obs_lines}
"""


# ---------------------------------------------------------------------------
# Call Google Gemini API
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Check common environment variable names for Google/Gemini key."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY", "GOOGLE_AI_KEY"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    return ""


def _ask_gemini(
    context: str,
    question: str,
    history: list[dict[str, str]],
    api_key: str,
) -> tuple[str | None, str | None]:
    """
    Sends conversation turn to Gemini.
    Returns (response_text, error_message).
    """
    models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    last_err = None

    for model in models:
        text, err = _try_gemini_model(model, context, question, history, api_key)
        if text:
            return text, None
        last_err = err

    return None, last_err


def _try_gemini_model(
    model: str,
    context: str,
    question: str,
    history: list[dict[str, str]],
    api_key: str,
) -> tuple[str | None, str | None]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    system_instruction = (
        "You are Mixora, a top-tier music mixing and mastering engineer speaking directly with the producer. "
        "You have the full technical audio analysis of their track below. "
        "Talk like a real human engineer in a recording studio — conversational, reactive, insightful, and natural. "
        "Answer ANY question they ask: emotional vibe, genre comparison, frequency issues, how to fix things, "
        "arrangement, loudness for Spotify/Apple Music, or creative production ideas. "
        "Keep each answer to 2-4 spoken sentences so it sounds great over audio. "
        "Do NOT use markdown, bold, asterisks, bullet points, or lists because your answer is spoken out loud. "
        "Be specific to their actual numbers and observations.\n\n"
        f"{context}"
    )

    contents: list[dict] = [
        {"role": "user", "parts": [{"text": system_instruction}]},
        {"role": "model", "parts": [{"text": "Understood! I'm loaded with the full mix details and ready to chat about the track."}]}
    ]

    for turn in history:
        role = "model" if turn.get("role") == "assistant" else "user"
        contents.append({
            "role": role,
            "parts": [{"text": turn.get("text", "")}]
        })

    contents.append({
        "role": "user",
        "parts": [{"text": question}]
    })

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.85,
            "maxOutputTokens": 300,
        }
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "").strip()
                    if text:
                        return text, None
            return None, "Empty response from Gemini"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        logger.warning("Gemini %s HTTP %s: %s", model, exc.code, body)
        return None, f"HTTP {exc.code}: {body}"
    except Exception as exc:
        logger.warning("Gemini %s error: %s", model, exc)
        return None, str(exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question(
    analysis: dict[str, Any],
    question: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    api_key = _get_api_key()
    context = _build_context(analysis)

    if not api_key:
        return {
            "text": (
                "To talk with me like a real AI assistant, add your GEMINI_API_KEY in your Render Dashboard. "
                "Go to mixora-api, click Environment, and add GEMINI_API_KEY with your key from Google AI Studio. "
                "Once saved, I will chat with you naturally in real-time."
            ),
            "source": "no_key",
            "used_fields": ["auth"],
        }

    ai_text, error = _ask_gemini(context, question, history or [], api_key)
    if ai_text:
        return {
            "text": ai_text,
            "source": "gemini",
            "used_fields": ["ai_gemini"],
        }

    return {
        "text": (
            f"AI connection error ({error or 'Request timed out'}). "
            "Please check that your GEMINI_API_KEY in Render is valid and has active quota at aistudio.google.com."
        ),
        "source": "error",
        "used_fields": ["error"],
    }
