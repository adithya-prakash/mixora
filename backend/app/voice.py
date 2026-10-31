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
    history = list of {"role": "user"|"model", "text": "..."}
    """
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )

    system_prompt = (
        "You are Mixora, an expert AI mixing and mastering engineer having a real spoken conversation "
        "with a music producer about their track. You have access to the full technical analysis of their "
        "audio file shown below. Answer every question directly and naturally, like a knowledgeable "
        "studio engineer talking in person. Keep answers to 2-4 natural conversational sentences — "
        "short enough to be spoken aloud clearly. Never use markdown formatting, bullet points, "
        "asterisks or bold text because your response will be read by text-to-speech. "
        "Answer ANY question the producer asks — about genre fit, emotional feel, what to fix, "
        "comparisons, mixing advice, mastering, distribution loudness targets, etc. "
        "Always refer back to the specific numbers in the analysis when relevant.\n\n"
        f"{context}"
    )

    # Build conversation turns for Gemini multi-turn format
    # First turn carries the system context as a preamble
    contents: list[dict] = []

    # Inject system context as the very first user turn if no history yet
    if not history:
        contents.append({
            "role": "user",
            "parts": [{"text": system_prompt}],
        })
        contents.append({
            "role": "model",
            "parts": [{"text": "Got it, I have the full analysis loaded. Ask me anything about this track!"}],
        })
    else:
        # For continued conversations, prepend context as a system-style preamble
        contents.append({
            "role": "user",
            "parts": [{"text": f"[Track context for this conversation]\n{context}\n[End of context]"}],
        })
        contents.append({
            "role": "model",
            "parts": [{"text": "Understood, I have the analysis context. I am ready to continue our conversation."}],
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
            "temperature": 0.75,
            "maxOutputTokens": 250,
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
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "").strip()
                    if text:
                        return text
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:400]
        logger.warning("Gemini HTTP error %s: %s", exc.code, body)
    except Exception as exc:
        logger.warning("Gemini voice query failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Fallback: smart keyword-based deterministic response (no AI key needed)
# ---------------------------------------------------------------------------

def _fallback_answer(analysis: dict[str, Any], question: str) -> str:
    n = question.casefold()
    obs = {item["id"]: item for item in analysis.get("obs", [])}

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
    if any(t in n for t in ("sub", "bass", "low end", "low-end", "low freq", "frequency")):
        b = analysis.get("bands_full", [])
        sub = b[0][1] if b else 0
        bass = b[1][1] if len(b) > 1 else 0
        return (
            f"Your sub-bass sits at {sub:.1f}% and bass at {bass:.1f}% of total spectral energy. "
            "Check kick and bass together in mono to avoid masking in the low end."
        )
    if any(t in n for t in ("stereo", "mono", "width", "correlation", "phase", "wide")):
        s = analysis.get("stereo", {})
        return (
            f"Stereo correlation is {s.get('correlation', 'N/A'):.2f} and width is {s.get('width', 'N/A'):.2f}. "
            f"Mono compatibility check: {s.get('mono', 'N/A')}. "
            "Use a correlation meter and check how it sounds in mono on a phone speaker."
        )
    if any(t in n for t in ("loud", "lufs", "volume", "dynamic", "crest", "range", "quiet", "spotify", "streaming", "apple")):
        return (
            f"Integrated loudness is {analysis.get('lufs', 'N/A')} LUFS with a crest factor of {analysis.get('crest', 'N/A')} dB. "
            "Spotify targets -14 LUFS, Apple Music targets -16 LUFS, and YouTube targets -13 LUFS. "
            "If you are louder than the target, streaming platforms will turn you down automatically."
        )
    if any(t in n for t in ("genre", "style", "fit", "sound like", "good for")):
        g = analysis.get("meta", {}).get("genre", "this genre")
        return (
            f"Your track is labelled as {g}. The measurements look at frequency balance, loudness, and stereo field — "
            "but genre fit is ultimately subjective. Compare the spectrum and loudness readings against a reference track from the same genre."
        )
    if any(t in n for t in ("wrong", "problem", "issue", "fix", "bad", "improve")):
        obs_list = analysis.get("obs", [])
        if obs_list:
            flagged = [o.get("title", "") for o in obs_list[:3]]
            return (
                f"The analysis flagged these potential issues: {', '.join(flagged)}. "
                "These are starting points for your ears to investigate, not definitive problems."
            )
        return "No significant issues were flagged automatically. The mix looks technically clean across loudness, clipping, and stereo measurements."

    g = analysis.get("meta", {}).get("genre", "your track")
    return (
        f"This {g} mix has a peak of {analysis.get('peak', 'N/A')} dBFS, "
        f"loudness of {analysis.get('lufs', 'N/A')} LUFS, and mono status is {analysis.get('stereo', {}).get('mono', 'N/A')}. "
        "Ask me about loudness, clipping, bass, stereo width, genre fit, or what to fix — I will give you targeted advice."
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

    # Deterministic fallback when no key or API failure
    return {
        "text": _fallback_answer(analysis, question),
        "source": "fallback",
        "used_fields": ["analysis"],
    }
