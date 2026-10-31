from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("mixora.voice")

# ---------------------------------------------------------------------------
# Build a rich context string from the analysis data so Gemini has full facts
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
    ) or "  None flagged."

    bands_str = ", ".join(f"{b[0]}: {b[1]:.1f}%" for b in bands) if bands else "N/A"
    bands_full_str = ", ".join(f"{b[0]}: {b[1]:.1f}%" for b in bands_full) if bands_full else "N/A"

    return f"""=== TRACK ANALYSIS ===
Genre: {meta.get('genre', 'Unknown')}
Filename: {meta.get('filename', 'Unknown')}
Duration: {meta.get('duration_s', '?')}s
Sample Rate: {meta.get('sample_rate', '?')} Hz
Channels: {meta.get('channels', '?')}

=== LOUDNESS & DYNAMICS ===
Peak: {analysis.get('peak', 'N/A')} dBFS
RMS: {analysis.get('rms', 'N/A')} dBFS
Integrated Loudness: {analysis.get('lufs', 'N/A')} LUFS
Loudness Range (LRA): {analysis.get('lra', 'N/A')} LU
Crest Factor: {analysis.get('crest', 'N/A')} dB

=== CLIPPING ===
Clipping Detected: {clipping}
Clipped Samples: {clip_detail.get('count', 0)}
Longest Clip Run: {clip_detail.get('max_run', 0)} samples

=== FREQUENCY BANDS (Display) ===
{bands_str}

=== FREQUENCY BANDS (Full Detail) ===
{bands_full_str}

=== STEREO ===
Correlation: {stereo.get('correlation', 'N/A')} (-1=antiphase, 0=unrelated, 1=mono/perfect)
Width: {stereo.get('width', 'N/A')} (0=mono, 1=max width)
Balance (L vs R): {stereo.get('balance', 'N/A')} dB
Mono Compatibility: {stereo.get('mono', 'N/A')} (OK / CAUTION / PROBLEM)

=== OBSERVATIONS (Flagged Issues) ===
{obs_lines}
"""


# ---------------------------------------------------------------------------
# Call Google Gemini with full conversation history
# ---------------------------------------------------------------------------

def _ask_gemini(
    context: str,
    question: str,
    history: list[dict[str, str]],
    api_key: str,
) -> str | None:
    """
    Send a conversation turn to Gemini Flash with full history.
    history = list of {"role": "user"|"assistant", "text": "..."}
    Tries gemini-1.5-flash first, falls back to gemini-1.0-pro on failure.
    """
    for model in ("gemini-1.5-flash", "gemini-1.0-pro"):
        result = _try_gemini_model(model, context, question, history, api_key)
        if result:
            return result
    return None


def _try_gemini_model(
    model: str,
    context: str,
    question: str,
    history: list[dict[str, str]],
    api_key: str,
) -> str | None:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    system_instruction = (
        "You are Mixora, an expert AI mixing and mastering engineer having a natural spoken conversation "
        "with a music producer about their track. You have the full technical analysis below. "
        "Answer every question directly and naturally, like a knowledgeable studio engineer talking in person. "
        "Keep answers to 2-4 sentences — short enough to be spoken aloud. "
        "Never use markdown, bullet points, asterisks or bold text — your response will be read by text-to-speech. "
        "Answer ANY question: genre fit, emotional feel, what to fix, how it compares to other songs, "
        "mixing advice, mastering, distribution loudness targets, or anything else the producer asks. "
        "Always refer to specific numbers from the analysis when relevant. "
        "Each question must get a UNIQUE, specific answer — never repeat the same response.\n\n"
        f"{context}"
    )

    # Build Gemini multi-turn contents array
    contents: list[dict] = []

    # System context as first turn pair
    contents.append({
        "role": "user",
        "parts": [{"text": system_instruction}],
    })
    contents.append({
        "role": "model",
        "parts": [{"text": "Got it, I have the full analysis loaded and I'm ready to give specific advice about this track."}],
    })

    # Replay conversation history
    for turn in history:
        role = "model" if turn.get("role") == "assistant" else "user"
        contents.append({
            "role": role,
            "parts": [{"text": turn.get("text", "")}],
        })

    # Add the current question
    contents.append({
        "role": "user",
        "parts": [{"text": question}],
    })

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 300,
            "stopSequences": [],
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        ],
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                finish_reason = candidates[0].get("finishReason", "")
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "").strip()
                    if text:
                        logger.info("Gemini %s answered (%s chars, finish=%s)", model, len(text), finish_reason)
                        return text
            logger.warning("Gemini %s: empty candidates. Full response: %s", model, json.dumps(res_data)[:400])
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        logger.warning("Gemini %s HTTP %s: %s", model, exc.code, body)
    except Exception as exc:
        logger.warning("Gemini %s error: %s", model, exc)

    return None


# ---------------------------------------------------------------------------
# Fallback: smart keyword-based deterministic response (no AI key needed)
# ---------------------------------------------------------------------------

def _fallback_answer(analysis: dict[str, Any], question: str) -> str:
    n = question.casefold()

    if any(t in n for t in ("clip", "distort", "peak", "limit")):
        c = analysis.get("clipping_detail", {})
        if analysis.get("clipping"):
            return (
                f"Your track is clipping with {c.get('count', 0)} samples hitting digital zero, "
                f"and the longest run is {c.get('max_run', 0)} consecutive samples. "
                "Pull back your master fader or limiter ceiling to fix this before export."
            )
        return (
            f"Your sample peak is {analysis.get('peak', 'N/A')} dBFS with no hard clipping detected. "
            "Keep watching for inter-sample true peaks when exporting for streaming."
        )

    if any(t in n for t in ("sub", "bass", "low end", "low-end", "low freq", "frequency", "eq", "equaliz")):
        b = analysis.get("bands_full", [])
        sub = b[0][1] if b else 0
        bass = b[1][1] if len(b) > 1 else 0
        return (
            f"Your sub-bass sits at {sub:.1f}% and bass at {bass:.1f}% of total spectral energy. "
            "Check kick and bass together in mono to avoid masking in the low end."
        )

    if any(t in n for t in ("stereo", "mono", "width", "correlation", "phase", "wide", "narrow")):
        s = analysis.get("stereo", {})
        return (
            f"Stereo correlation is {s.get('correlation', 'N/A'):.2f} and width is {s.get('width', 'N/A'):.2f}. "
            f"Mono compatibility check: {s.get('mono', 'N/A')}. "
            "Use a correlation meter and check how it sounds in mono on a phone speaker."
        )

    if any(t in n for t in ("loud", "lufs", "volume", "dynamic", "crest", "range", "quiet", "spotify", "streaming", "apple", "youtube")):
        return (
            f"Integrated loudness is {analysis.get('lufs', 'N/A')} LUFS with a crest factor of {analysis.get('crest', 'N/A')} dB. "
            "Spotify targets -14 LUFS, Apple Music -16 LUFS, and YouTube -13 LUFS. "
            "If you are louder than the target, streaming platforms will turn you down automatically."
        )

    if any(t in n for t in ("genre", "style", "sound like", "good for", "suitable", "work for")):
        g = analysis.get("meta", {}).get("genre", "this genre")
        lufs = analysis.get("lufs", "N/A")
        return (
            f"Your track is labelled {g} with loudness at {lufs} LUFS. "
            f"Genre fit depends on how the energy distribution and stereo width compare to reference tracks in the same genre. "
            "Try uploading a reference track in the Reference tab for a direct comparison."
        )

    if any(t in n for t in ("wrong", "problem", "issue", "fix", "bad", "improve", "advice")):
        obs_list = analysis.get("obs", [])
        if obs_list:
            flagged = [o.get("title", "") for o in obs_list[:3]]
            return (
                f"The analysis flagged these potential issues: {', '.join(flagged)}. "
                "These are starting points for your ears to investigate, not definitive problems."
            )
        return "No significant issues were flagged automatically. The mix looks technically clean across loudness, clipping, and stereo measurements."

    if any(t in n for t in ("differ", "compare", "other song", "different song", "vs", "versus", "better", "worse")):
        g = analysis.get("meta", {}).get("genre", "this genre")
        lufs = analysis.get("lufs", "N/A")
        peak = analysis.get("peak", "N/A")
        s = analysis.get("stereo", {})
        return (
            f"Compared to typical {g} releases, your track hits {lufs} LUFS integrated loudness and {peak} dBFS peak. "
            f"Most commercial {g} tracks sit between -8 and -6 LUFS with wide stereo fields. "
            f"Your stereo width is {s.get('width', 'N/A'):.2f} — upload a reference track in the Reference tab to see exact differences."
        )

    if any(t in n for t in ("how", "why", "what", "explain", "tell", "describe")):
        obs_list = analysis.get("obs", [])
        g = analysis.get("meta", {}).get("genre", "your genre")
        lufs = analysis.get("lufs", "N/A")
        lra = analysis.get("lra", "N/A")
        if obs_list:
            top = obs_list[0]
            return (
                f"{top.get('title', 'The top observation')}: {top.get('detail', '')} "
                f"Overall the track is {lufs} LUFS with {lra} LU of dynamic range, which is typical context for {g}."
            )
        return (
            f"This {g} mix measures {lufs} LUFS integrated loudness and {lra} LU dynamic range. "
            "The frequency balance, stereo width, and dynamics all look within expected ranges. "
            "Ask me about a specific area — bass, loudness, stereo, or clipping — for targeted advice."
        )

    # Final fallback — still give unique info rather than generic
    g = analysis.get("meta", {}).get("genre", "your track")
    lufs = analysis.get("lufs", "N/A")
    lra = analysis.get("lra", "N/A")
    s = analysis.get("stereo", {})
    obs_list = analysis.get("obs", [])
    if obs_list:
        top_issue = obs_list[0].get("title", "")
        return (
            f"For this {g} mix: loudness is {lufs} LUFS, dynamic range is {lra} LU, "
            f"and the top flagged issue is '{top_issue}'. "
            "Ask me specifically about loudness, clipping, bass, stereo width, genre fit, or what to fix."
        )
    return (
        f"This {g} mix has {lufs} LUFS integrated loudness, {lra} LU dynamic range, "
        f"and stereo width of {s.get('width', 'N/A'):.2f}. No major issues were flagged. "
        "Ask me about loudness, clipping, bass, stereo, or genre fit for specific advice."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question(
    analysis: dict[str, Any],
    question: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    context = _build_context(analysis)

    if gemini_key:
        ai_text = _ask_gemini(context, question, history or [], gemini_key)
        if ai_text:
            return {
                "text": ai_text,
                "source": "gemini",
                "used_fields": ["ai_gemini"],
            }
        logger.warning("Gemini failed for question: %r — using fallback", question)

    # Deterministic fallback when no key or API failure
    return {
        "text": _fallback_answer(analysis, question),
        "source": "fallback",
        "used_fields": ["analysis"],
    }
