from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("mixora.voice")

# Cached discovered endpoint: "https://generativelanguage.googleapis.com/{version}/{name}:generateContent"
_DISCOVERED_ENDPOINT: str | None = None


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
# Dynamic Model Discovery: Ask Google which models are active for this key
# ---------------------------------------------------------------------------

def _get_generate_url(api_key: str) -> tuple[str | None, str | None]:
    """
    Dynamically asks Google API for available models supporting generateContent.
    Picks the best available model (prefers Flash models).
    Returns (endpoint_url_with_key, error_message).
    """
    global _DISCOVERED_ENDPOINT
    if _DISCOVERED_ENDPOINT:
        return f"{_DISCOVERED_ENDPOINT}?key={api_key}", None

    last_error = None
    for api_version in ("v1beta", "v1"):
        list_url = f"https://generativelanguage.googleapis.com/{api_version}/models?key={api_key}"
        try:
            req = urllib.request.Request(list_url, method="GET")
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                candidates = []
                for m in models:
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        candidates.append((m.get("name", ""), api_version))

                # Preference order:
                # 1. gemini-2.0-flash
                # 2. gemini-1.5-flash
                # 3. any other flash model
                # 4. gemini-1.5-pro or any gemini
                # 5. any model with generateContent
                for pref in ("gemini-2.0-flash", "gemini-1.5-flash", "flash", "gemini-1.5-pro", "gemini"):
                    for name, ver in candidates:
                        if pref in name.lower():
                            _DISCOVERED_ENDPOINT = f"https://generativelanguage.googleapis.com/{ver}/{name}:generateContent"
                            logger.info("Auto-selected Gemini model: %s via %s", name, ver)
                            return f"{_DISCOVERED_ENDPOINT}?key={api_key}", None

                if candidates:
                    name, ver = candidates[0]
                    _DISCOVERED_ENDPOINT = f"https://generativelanguage.googleapis.com/{ver}/{name}:generateContent"
                    logger.info("Auto-selected fallback model: %s via %s", name, ver)
                    return f"{_DISCOVERED_ENDPOINT}?key={api_key}", None

                last_error = f"No models found supporting generateContent in {api_version}"
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:250]
            last_error = f"HTTP {exc.code} listing models: {body}"
            logger.warning("Listing models failed on %s: %s", api_version, last_error)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Listing models error on %s: %s", api_version, exc)

    return None, last_error


# ---------------------------------------------------------------------------
# Call Gemini with conversational multi-turn
# ---------------------------------------------------------------------------

def _ask_gemini(
    context: str,
    question: str,
    history: list[dict[str, str]],
    api_key: str,
) -> tuple[str | None, str | None]:
    url, err = _get_generate_url(api_key)
    if not url:
        return None, err or "Could not discover any active Gemini model for this API key"

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

    contents: list[dict] = [
        {"role": "user", "parts": [{"text": system_instruction}]},
        {"role": "model", "parts": [{"text": "Understood! I'm loaded with the full mix details and ready to chat naturally about the track."}]}
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
        with urllib.request.urlopen(req, timeout=18) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "").strip()
                    if text:
                        return text, None
            return None, "Empty response candidate from Gemini"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        # Reset cached endpoint in case model was decommissioned
        global _DISCOVERED_ENDPOINT
        _DISCOVERED_ENDPOINT = None
        return None, f"HTTP {exc.code}: {body}"
    except Exception as exc:
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
            "Please verify your key in Google AI Studio has access to Gemini Flash."
        ),
        "source": "error",
        "used_fields": ["error"],
    }
