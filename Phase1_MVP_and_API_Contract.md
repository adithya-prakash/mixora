# Mixora Phase 1: MVP and API Contract

**Status:** Phase 1 baseline, ready for team review  
**Contract version:** 1.0  
**Scope:** One uploaded mix, optional reference, one completed analysis response

This document resolves the initial scope and interface decisions across `Idea.md`, the HTML prototype, and `Mixora_Backend_and_AI_Spec.md`. The JSON Schema in `api-contract-v1.schema.json` is authoritative for the `/analyze` response. A code change to either side must update the schema and its contract tests in the same change.

## Product and MVP

Mixora helps a producer inspect measurable characteristics of their mix and investigate possible issues. It does not automatically alter audio, grade artistic quality, or tell the producer to copy a reference.

The MVP workflow is: choose a genre, upload a mix, optionally upload a reference, request analysis, and review measurements, visualizations, and up to three evidence-backed observations. A reference is optional context and is analyzed with the same DSP settings as the mix.

## Decisions

| Topic | Phase 1 decision |
| --- | --- |
| Product name | Mixora, matching the prototype and backend spec. |
| Backend and frontend | FastAPI backend; the prototype is a visual reference, not the production React app. |
| Upload formats | WAV and MP3 for mix and reference. Reject other formats after content/decode validation, regardless of filename. |
| Upload limits | 100 MB per file and 15 minutes maximum decoded duration. Apply limits to the reference independently. |
| Genres | Accept the 12 names shown in the prototype: Dubstep, Bass House, Pop, Rock, Hip Hop, EDM, Techno, Drum & Bass, Trap, Lo-fi, R&B, Indie. Build and validate genre profiles for Dubstep and Bass House first. Other accepted genres use an explicitly marked generic-v0 profile with low confidence; do not make calibrated genre-percentile claims for them. |
| Frequency bands | Preserve the prototype's four-item `bands` array: Sub-bass, Bass, Midrange, Highs. Also return five `bands_full` values, splitting High-mid from Highs for analysis and audit. Highs in `bands` is the sum of High-mid and Highs. |
| Spectrum and EQ | Return 16 normalized spectrum points and eight EQ band objects for the mix. Include reference `B` curves only when a reference was uploaded. EQ values describe measured tonal balance; they are not suggested corrective settings. |
| Feedback | Return zero to three observations. Confirmed is reserved for objectively measurable conditions; genre- or reference-relative findings are potential. Every observation must cite fields and values from this response. Deterministic templates are the MVP baseline; an LLM is not required for Phase 1. |
| Reference | Optional on `/analyze`; absent reference fields are omitted, never fabricated. Reference comparison is contextual, not a target or quality verdict. |
| Persistence | No account or saved-history promise in MVP. Treat uploads as temporary processing data and delete them after analysis unless an explicitly approved retention policy is introduced. Return an analysis `id` for correlation and future compatibility; retrieval is not part of this phase. |
| Processing | Synchronous `POST /analyze` for the first integration. Add job status or streaming only if measured latency makes synchronous requests unreliable. |
| Explicit deferrals | Revision comparison (`/compare`), voice service (`/voice/ask`), TTS, saved-analysis retrieval, user accounts, real-time analysis, DAW integration, source separation, and automatic mixing. Prototype-only controls for these features must be labeled as mock/unavailable until implemented. |

## `POST /analyze`

Request content type: `multipart/form-data`.

| Field | Required | Contract |
| --- | --- | --- |
| `file` | Yes | WAV or MP3; at most 100 MB and 15 minutes after decoding. |
| `reference` | No | Same validation and limits as `file`. |
| `genre` | Yes | One of the 12 exact display names listed above. |

Success is HTTP `200` with a JSON body conforming to [`api-contract-v1.schema.json`](api-contract-v1.schema.json). All numeric measurements are JSON numbers, not formatted strings. Decibels are rounded to two decimal places. The response includes `id`, `meta`, core levels/loudness/clipping, four display bands, five full bands, `spectrum.A`, `eq.A`, stereo metrics, and `obs`. `ref`, `spectrum.B`, and `eq.B` are present only when a reference was supplied.

### Measurement conventions

- `peak`: sample peak in dBFS across channels. `true_peak` may be included when implemented and validated.
- Decode to float audio and resample to 44.1 kHz for analysis; retain the original rate in `meta.sample_rate`. Use an 8192-sample Hann STFT with 50% overlap for frequency analysis. Do not normalize the audio before measuring.
- `rms`: full-file RMS in dBFS, computed as the mean of per-channel mean-square energy.
- `lufs`: integrated loudness using ITU-R BS.1770-4 gating. `lra` follows EBU Tech 3342. `crest` is `peak - rms` in dB.
- `clipping`: true when the documented clipping detector finds clipped samples or true peak above 0 dBTP. Return `clipping_detail` so the result is auditable; `percent_samples` is the fraction of samples affected in `[0,1]`. Distinguish samples at the digital limit from true-peak overs when possible.
- `bands_full`: percentages of total in-range power at 20-60 Hz, 60-250 Hz, 250-2000 Hz, 2000-6000 Hz, and 6000-20000 Hz, in that order. `bands` uses the first three values plus the sum of the final two. Use identical band edges for mix and reference; percentages sum to approximately 100.
- `spectrum`: 16 log-spaced points from 20 Hz to 20 kHz, normalized by mapping smoothed spectral dB values from a fixed `-90..-20 dB` range to `[0,1]` and clamping outside the range. Use identical smoothing and scale for mix and reference.
- `stereo.balance`: `20*log10(rms_left/rms_right)` in dB; negative means left-heavy. Mono input reports balance `0`, correlation `1`, and width `0`.
- `stereo.correlation`: Pearson left/right correlation in `[-1,1]`. `width` is side energy divided by mid-plus-side energy, in `[0,1]`. `mono` is `OK`, `CAUTION`, or `PROBLEM`, based on documented mono-sum level and low-band cancellation checks.
- The eight `eq` entries have fixed order and types: `hpf`, `ls`, four `bell`, `hs`, `lpf`. They are a deterministic fit to deviation from the selected profile, not instructions to apply those gains.

### Errors

All API errors use this envelope:

```json
{
  "error": {
    "code": "unsupported_format",
    "message": "Use a WAV or MP3 audio file."
  }
}
```

| HTTP | Code | Meaning |
| --- | --- | --- |
| 400 | `unsupported_format` | Unsupported content or invalid audio container. |
| 400 | `invalid_genre` | Missing or unsupported genre value. |
| 400 | `invalid_request` | Missing required request data or invalid request fields. |
| 413 | `too_large` | File exceeds 100 MB or decoded duration exceeds 15 minutes. |
| 422 | `decode_failed` | File cannot be safely decoded as valid audio. |
| 404 | `not_found` | Analysis identifier does not exist or has expired. |
| 429 | `rate_limited` | Request rate limit exceeded. |
| 500 | `analysis_failed` | Valid audio could not be analyzed. |

The frontend must parse the JSON error envelope and show `error.message`; a generic status-only message is not sufficient. CORS origins and upload limits are configuration, not hard-coded secrets or wildcard production settings.

## Integration requirements

Before Phase 2 implementation begins, the team should approve this document and schema together. The HTML prototype needs a few contract-alignment changes before it can consume real responses:

1. Handle a missing reference without dereferencing `ref`, `spectrum.B`, or `eq.B`; show reference comparisons only when those fields exist.
2. Read and display the five `bands_full` values where the full analysis is shown, while preserving the four-band presentation where the current UI needs it.
3. Use the same exact genre names as the request enum, and label generic-v0 results as low-confidence.
4. Parse the defined error envelope. Do not silently replace backend-down or failed-analysis states with plausible-looking mock results.
5. Clearly gate mock-only revision and voice interactions so a demo cannot imply those backend features are live.

## Phase 1 exit gate

- Product and team approve the decisions table and scope boundaries.
- Frontend and backend owners approve the schema and measurement conventions.
- A sample response with a reference and one without a reference both validate against the schema.
- The prototype's missing-reference and error behavior is captured as integration work before live API wiring.
