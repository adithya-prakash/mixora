from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("mixora.voice")

# Supported models in Google Gemini API (high-availability Flash models)
PRIMARY_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
)


# ---------------------------------------------------------------------------
# Build rich context string for the AI
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
# API Key discovery
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Check common environment variable names for Google/Gemini key."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY", "GOOGLE_AI_KEY"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    return ""


# ---------------------------------------------------------------------------
# Call Gemini via official REST structure
# ---------------------------------------------------------------------------

def _ask_gemini(
    context: str,
    question: str,
    history: list[dict[str, str]],
    api_key: str,
) -> tuple[str | None, str | None]:
    system_instruction = (
        "You are Mixora, a friendly, top-tier audio mixing and mastering engineer speaking directly with the producer. "
        "You have access to the full technical analysis of their audio file below. "
        "Talk like a real human engineer in a recording studio — conversational, reactive, insightful, and natural. "
        "Answer ANY question they ask: emotional vibe, genre comparison, frequency issues, how to fix things, "
        "arrangement, loudness for Spotify/Apple Music, or creative production ideas. "
        "Keep each answer to 2-4 spoken sentences so it sounds great over text-to-speech. "
        "Do NOT use markdown formatting, bullet points, asterisks, or bold text because your answer is spoken out loud. "
        "Be specific to their actual numbers and observations.\n\n"
        f"{context}"
    )

    # Build properly alternating conversation turns
    contents: list[dict] = []
    for turn in history[-6:]:
        role = "model" if turn.get("role") == "assistant" else "user"
        text = turn.get("text", "").strip()
        if not text:
            continue
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n" + text
        else:
            contents.append({"role": role, "parts": [{"text": text}]})

    if contents and contents[-1]["role"] == "user":
        contents[-1]["parts"][0]["text"] += "\n" + question
    else:
        contents.append({"role": "user", "parts": [{"text": question}]})

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 300,
        }
    }
    encoded_payload = json.dumps(payload).encode("utf-8")

    errors: list[str] = []

    for model in PRIMARY_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        try:
            req = urllib.request.Request(
                url,
                data=encoded_payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    # Extract spoken text pieces (ignoring any thought reasoning blocks)
                    text_pieces = []
                    for p in parts:
                        if not p.get("thought", False):
                            t = p.get("text", "").strip()
                            if t:
                                text_pieces.append(t)
                    if text_pieces:
                        full_reply = " ".join(text_pieces)
                        logger.info("Generated reply using %s (%s chars)", model, len(full_reply))
                        return full_reply, None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:200]
            err_msg = f"{model} (HTTP {exc.code}): {body}"
            errors.append(err_msg)
            logger.warning("Gemini model %s failed: %s", model, err_msg)
        except Exception as exc:
            err_msg = f"{model} error: {exc}"
            errors.append(err_msg)
            logger.warning("Gemini error on %s: %s", model, exc)

    return None, " | ".join(errors)


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
                "Please add GEMINI_API_KEY to your Render Dashboard Environment Variables. "
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
            f"AI connection note: {error}. "
            "Please verify your key in Google AI Studio."
        ),
        "source": "error",
        "used_fields": ["error"],
    }
