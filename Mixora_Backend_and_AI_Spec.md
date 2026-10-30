# Mixora: Backend and AI Model Build Specification

**Audience:** backend / DSP / AI developers
**Stack:** Python 3.11+, FastAPI
**Frontend contract source:** the current Mixora UI (`API_BASE` + `/analyze`, `/voice/ask`, mock schema inside `MOCK`)

> **Golden rule:** the frontend only *displays* results. Every measurement, spectrum, EQ band and observation is calculated by the backend. The AI must explain *actual measured numbers*, never give generic mixing advice.

---

## 1. What you are building (overview)

| # | Component | Purpose |
|---|-----------|---------|
| 1 | **API layer** (FastAPI) | Upload, analysis, comparison, voice, retrieval |
| 2 | **Audio-analysis engine** (DSP) | Peak, RMS, LUFS, LRA, crest, clipping, bands, spectrum, stereo |
| 3 | **Tonal-balance band extractor** | The 8 parametric EQ bands (type, freq, gain, Q) shown in the EQ analyzer |
| 4 | **Genre profile store** | Reference statistics per genre used for genre-aware comparison |
| 5 | **Observation engine (the "AI model")** | Rules + scoring that produce up to 3 prioritized, evidence-backed observations |
| 6 | **LLM explanation layer** | Turns structured observations into clear wording, strictly grounded |
| 7 | **Voice service** | Question in, grounded spoken answer out (STT optional, LLM, TTS) |
| 8 | **Reference + revision comparison** | Deltas between tracks and between versions |
| 9 | **Storage, jobs, security** | Analysis persistence, async processing, file cleanup, limits |

```
Frontend  ──►  FastAPI  ──►  Job queue ──► DSP engine ──► Band extractor
                  │                              │
                  │                              ▼
                  │                       Observation engine ──► LLM phrasing
                  ▼
             Voice service (STT → LLM grounded on analysis JSON → TTS)
                  │
             Storage (DB + temp audio)
```

---

## 2. API contract (must match the frontend)

Agree on this JSON with the frontend developer **before** coding the cards. The names below are exactly what the current UI reads. Keep all dB values as plain numbers (not strings), rounded to 2 decimals.

### 2.1 `POST /analyze`

**Request:** `multipart/form-data`

| Field | Type | Notes |
|-------|------|-------|
| `file` | file (WAV/MP3) | required, max 100 MB |
| `reference` | file (WAV/MP3) | optional |
| `genre` | string | one of the genre list in section 6 (UI sends the display name, e.g. `"Bass House"`, `"Drum & Bass"`); normalize server-side |

**Response `200`:**

```jsonc
{
  "id": "an_9f3c...",                 // used by /voice/ask and GET /analysis/{id}
  "peak": -0.2,                        // dBFS (sample peak; add "true_peak" if available)
  "rms": -7.56,                        // dBFS
  "lufs": -8.1,                        // integrated loudness, LUFS
  "lra": 4.2,                          // loudness range, LU
  "crest": 7.36,                       // dB (peak - rms)
  "clipping": true,                    // boolean flag; details in "clipping_detail"
  "bands": [["Sub-bass", 36.8], ["Bass", 24.39], ["Midrange", 21.91], ["Highs", 16.77]],
                                       // [name, percent of total energy]; sums to ~100
  "spectrum": { "A": [/*16 floats 0..1*/], "B": [/*16 floats 0..1*/] },
                                       // log-spaced 20 Hz -> 20 kHz, B only if reference sent
  "eq": { "A": [ /*8 band objects*/ ], "B": [ /*8 band objects*/ ] },
  "stereo": { "balance": -0.4, "correlation": 0.82, "width": 0.31, "mono": "OK" },
  "ref": {                             // measurements of the reference track (omit if none)
    "peak": -0.9, "rms": -9.1, "lufs": -9.4, "lra": 5.1, "crest": 8.2,
    "corr": 0.88, "width": 0.27, "bands": [31.2, 26.1, 24.8, 17.9]
  },
  "obs": [ /* max 3 observation objects, section 5 */ ]
}
```

**EQ band object** (always exactly 8, in this order):

```jsonc
{ "t": "hpf|ls|bell|bell|bell|bell|hs|lpf", "f": 120, "g": 2.2, "q": 1.1 }
```

| Index | Type code `t` | Meaning | Gain `g` |
|-------|---------------|---------|----------|
| 1 | `hpf` | Low cut | ignored (0) |
| 2 | `ls` | Low shelf (sub-bass region) | dB |
| 3-6 | `bell` | Peak/dip | dB |
| 7 | `hs` | High shelf | dB |
| 8 | `lpf` | High cut | ignored (0) |

Ranges: `f` 20-20000 Hz, `g` within -18 to +18 dB, `q` 0.3-8.

**Fields worth adding now (the UI does not read them yet, but they make the output auditable):**

```jsonc
"meta": { "filename": "...", "duration_s": 192.4, "sample_rate": 44100, "channels": 2, "genre": "Dubstep" },
"settings": {
  "band_edges_hz": { "sub": [20,60], "bass": [60,250], "mid": [250,2000], "high_mid": [2000,6000], "highs": [6000,20000] },
  "fft_size": 8192, "window": "hann", "loudness_standard": "ITU-R BS.1770-4"
},
"clipping_detail": { "count": 214, "percent_samples": 0.004, "max_run": 6 },
"bands_full": [["Sub-bass",36.8],["Bass",24.39],["Midrange",21.91],["High-mid",9.1],["Highs",7.67]]
```

> **Decision to settle with the frontend:** your brief defines **5** bands (incl. high-mid 2-6 kHz) but the UI currently shows **4** (Sub-bass, Bass, Midrange, Highs). Either send 4 (merge high-mid into highs) or update the UI to 5. Use identical settings for both tracks.

### 2.2 `POST /compare` (revision comparison)

**Request:** `multipart/form-data` with `previous_id` (analysis id) and `file` (revised mix), optional `genre`.

**Response:**

```jsonc
{
  "previous_id": "an_1", "revised_id": "an_2",
  "metrics": [ { "name": "peak", "previous": -0.2, "revised": -1.0, "delta": -0.8, "unit": "dBFS" }, ... ],
  "bands":   [ { "name": "Sub-bass", "previous": 36.8, "revised": 33.5, "delta": -3.3 }, ... ],
  "stereo":  { "correlation_delta": 0.03, "width_delta": -0.02 },
  "clipping": { "previous": true, "revised": false },
  "observation_changes": [ { "id": "clipping", "status": "resolved|persisted|new|worsened|improved_numerically" } ],
  "note": "Differences are numerical only. They do not show which version sounds better."
}
```

Never output "better/worse" language. Only what changed.

### 2.3 `POST /voice/ask`

**Request (JSON, text first):**
```json
{ "analysis_id": "an_9f3c", "question": "Why is my sub-bass flagged?", "history": [ {"role":"user","text":"..."}, {"role":"assistant","text":"..."} ] }
```
Later also accept `audio` (webm/wav) and run STT server-side.

**Response:**
```json
{ "text": "Your sub-bass holds 36.8% of the measured energy ...", "audio_url": "https://.../tts/abc.mp3", "used_fields": ["bands","obs[0]"] }
```
`audio_url` optional but the UI plays it automatically if present.

### 2.4 `GET /analysis/{id}`
Returns the stored `/analyze` response (plus compare data if present). Used to reload a session.

### 2.5 Cross-cutting API requirements
- **Errors:** `{ "error": { "code": "unsupported_format|too_large|decode_failed|analysis_failed|not_found", "message": "..." } }` with proper HTTP status. The UI shows `message`.
- **CORS:** allow the frontend origin only; read origins from env.
- **Env config:** `ALLOWED_ORIGINS`, `MAX_UPLOAD_MB`, `LLM_API_KEY`, `LLM_MODEL`, `TTS_PROVIDER`, `DB_URL`, `STORAGE_DIR`.
- **Progress (nice to have):** the UI shows 5 stages (upload, levels/loudness, bands, stereo, observations). Provide `GET /analysis/{id}/status` or SSE with `stage` and `percent`; until then `/analyze` may simply respond when finished.
- **Note:** a published claude.ai artifact page cannot call your server (its security policy blocks outside requests). Integrate with the real Vite/React build or any normally hosted frontend.

---

## 3. Audio-analysis engine (DSP)

Libraries: `soundfile` / `ffmpeg` (decode), `numpy`, `scipy`, `librosa`, `pyloudnorm`, optionally `essentia`.

### 3.1 Load and normalize
1. Validate by magic bytes, not just extension. Reject > duration limit (e.g. 15 min).
2. Decode to float32. Keep original channels. Resample to a fixed rate (e.g. 44.1 kHz) for all analysis so both tracks use identical settings.
3. Do **not** normalize loudness before measuring.

### 3.2 Measurements

| Metric | Method |
|--------|--------|
| **Peak (dBFS)** | `20*log10(max(abs(x)))` over all channels. Also compute 4x oversampled **true peak** (BS.1770) as an extra field. |
| **RMS (dBFS)** | `20*log10(sqrt(mean(x^2)))` over the full mix (state if mono-summed or per-channel average; be consistent). |
| **Integrated loudness (LUFS)** | ITU-R BS.1770-4 with gating (`pyloudnorm`). Validate against `ffmpeg -af ebur128`. |
| **Loudness range (LRA)** | EBU R128 / Tech 3342 (short-term loudness, 10th-95th percentile). |
| **Crest factor (dB)** | `peak_dBFS - rms_dBFS`. |
| **Clipping** | Flag if >= N consecutive samples (e.g. 3+) at >= 0.999 full scale, or true peak > 0 dBTP. Return count, percent of samples, longest run. Separate "at limit" (hard-limited but not clipped) from "clipped". |

### 3.3 Frequency analysis
- STFT (Hann, e.g. 8192 or 16384 samples for good low-end resolution, 50% overlap) on the mono sum; average power over time (Welch).
- **Band energy** = integrate power inside fixed edges: sub 20-60, bass 60-250, mid 250-2k, high-mid 2k-6k, highs 6k-20k Hz. Express as percent of total in-range energy. **Document the edges** and reuse them for reference and revisions.
- **Spectrum curve for the UI:** smooth to 1/3 or 1/6 octave, convert to dB, take 16 log-spaced points between 20 Hz and 20 kHz, normalize to 0..1 with a documented method (e.g. min-max over a fixed -90..-20 dB window, **the same for A and B** so they are comparable).
- **Optional (stretch):** time-based spectrum (per 1-2 s frame) so the UI can animate the EQ along the playhead. Return as `spectrum_over_time: { hop_s, frames: [[16 floats], ...] }`.

### 3.4 Stereo analysis
| Metric | Method |
|--------|--------|
| **L/R balance (dB)** | `20*log10(rms_L / rms_R)` (negative = left heavier; keep sign convention documented) |
| **Correlation** | Pearson correlation of L and R (-1..+1) |
| **Width (0..1)** | Side energy / (Mid + Side energy), or `S_rms / M_rms` scaled; document the formula |
| **Mono compatibility** | Sum to mono, compare level and band energy to the stereo mix. Flag (`"OK"`, `"CAUTION"`, `"PROBLEM"`) when level drops > ~3 dB or low bands (< 150 Hz) lose energy. Check sub-bass correlation separately. |

Handle mono files (width 0, correlation 1, note "mono source").

### 3.5 Tonal-balance band extraction (the EQ analyzer data)

The UI draws an 8-band parametric EQ per track. These bands **describe the measured tonal balance relative to a genre-typical curve**; they are not instructions to match the reference.

1. Compute the track's smoothed spectrum (dB).
2. Subtract the **genre target/average curve** (section 6) to get a deviation curve.
3. Fit the 8 fixed-type filters to that deviation with constrained least squares (`scipy.optimize`), bounds on `f` per band role (e.g. band 2 low shelf 30-120 Hz, bell 3 60-250 Hz, bell 4 250-800 Hz, bell 5 800-2.5k Hz, bell 6 2.5-7k Hz, high shelf 5-16 kHz), `g` ±18 dB, `q` 0.3-8.
4. `hpf` / `lpf` frequencies come from detected low/high spectral roll-off (e.g. frequency where energy falls 12 dB below the passband).
5. Same algorithm, same bounds, same random seed for A and B; deterministic output.
6. Use the response formulas the UI draws (bell = Gaussian in log-freq with width ~0.7/Q; shelves = logistic; cuts = 12 dB/oct) or tell the frontend your formulas so the curves agree with your fit.

---

## 4. Reference comparison logic
- Compute every measurement for the reference with **identical settings**.
- Return aggregates in `ref`, plus `spectrum.B` and `eq.B`.
- Deltas are computed (or at least verifiable) as `user - reference`; the UI shows them side by side.
- Because loudness differences distort spectral comparisons, also provide **loudness-matched spectra** (`spectrum_matched`) or at least note the LUFS offset.
- Never produce a verdict like "make your mix like the reference". Use phrasing such as "your sub-bass share is 5.6 points higher than the reference".

---

## 5. The AI model: observation engine

This is the product's core. Build it as **structured detection first, language second**, so every sentence is traceable to numbers.

### 5.1 Pipeline
```
metrics + genre profile
      │
      ▼
 Detectors (rules/thresholds/ML)  →  candidate observations (with evidence, severity, confidence)
      │
      ▼
 Ranker / de-duplicator           →  top 3, ordered by priority
      │
      ▼
 LLM phrasing (grounded, validated) → final text fields
```

### 5.2 Observation schema (what `/analyze` returns in `obs`, max 3)

```jsonc
{
  "id": "low_end_concentration",        // stable code, needed for revision tracking
  "t": "Low-end energy concentration",  // title
  "p": "High|Medium|Low",               // priority
  "c": "confirmed|potential",           // CONFIRMED = objectively measurable problem; POTENTIAL = depends on genre/taste
  "conf": "High|Medium|Low",            // confidence in the detection
  "ev": "Sub-bass 36.8% · Bass 24.39%", // evidence: real numbers only
  "ex": "Why it might matter ...",
  "a": "What to try in the DAW ...",
  "evidence_fields": ["bands.0","bands.1"]   // optional, for auditing
}
```

**Confirmed vs potential rule:** only things that are objectively verifiable (e.g. clipping detected, severe phase cancellation, DC offset, true peak over 0 dBTP) are `confirmed`. Anything compared to a genre norm or a reference is `potential`.

### 5.3 Detectors to implement (MVP first)

| Priority | Detector id | Trigger (tune on data) | Kind |
|----------|-------------|------------------------|------|
| MVP | `clipping` | clipped samples / true peak > 0 dBTP | confirmed |
| MVP | `low_end_concentration` | sub+bass % far above genre percentile (e.g. > P90) | potential |
| MVP | `loudness_vs_genre` | LUFS outside genre typical range | potential |
| MVP | `dynamics_dense` / `dynamics_wide` | crest/LRA vs genre range | potential |
| MVP | `mono_compat` | mono sum level drop or low-band cancellation | confirmed if severe |
| MVP | `stereo_width_low_end` | correlation of < 150 Hz below threshold | potential |
| Should | `sub_bass_energy_low/high` | sub share vs genre | potential |
| Should | `mid_scoop / mid_buildup` | mid-band deviation from genre curve | potential |
| Should | `harshness_highmid` | 2-6 kHz excess | potential |
| Should | `kick_bass_overlap` | low-band transient + sustained bass overlap (band-limited onset detection, sidechain/ducking measure) | potential |
| Stretch | `lr_imbalance`, `dc_offset`, `dropout/silence` | | confirmed |

Each detector returns: `severity (0-1)`, `confidence`, `evidence` (field paths + values), and a template id for explanation/action.

### 5.4 Ranking
`score = severity * confidence * genre_weight * (1.5 if confirmed else 1)`. Pick the top 3, drop near-duplicates (e.g. `low_end_concentration` vs `sub_bass_energy_high` merge). Priority label from score thresholds.

**Confidence** (High/Medium/Low) from: distance beyond threshold, measurement reliability (e.g. short track, mono source lowers it), and genre-profile sample size.

### 5.5 Genre-aware logic
Compare against that genre's profile, not universal rules. Clearly say what was compared ("typical for Dubstep in the profile set") and keep language hedged for `potential` items.

### 5.6 LLM phrasing layer (grounded)
- Input: **only** the structured observation + relevant metrics + genre label. Output: JSON with `ex` and `a` text.
- Use structured/JSON output. Model name from env (`LLM_MODEL`); check the provider docs for the current model string.
- **Guardrails** (system prompt + validator):
  1. Use only numbers present in the input; no invented measurements.
  2. Never say a version/mix is "better", "worse", "fixed" or "ruined".
  3. Never tell the user to match the reference.
  4. Hedge `potential` items ("may", "could", "check whether").
  5. Actions must be DAW-realistic (EQ, sidechain, limiter ceiling, stereo widener on lows, etc.).
- **Post-validation:** regex-extract every number in the output and confirm it exists in the input (within rounding). If validation fails, fall back to the deterministic template text for that detector.
- Keep deterministic template text for every detector so the app still works without the LLM.

### 5.7 Optional ML components (only if time permits)
- Genre sanity check (classifier on embeddings) to warn "this sounds closer to X than the selected genre".
- Learned detectors (e.g. kick/bass collision, vocal/lead masking) trained on labelled stems. Always output a confidence and keep the rule-based fallback.
- Model evaluation: precision/recall per detector on a labelled set of mixes with known issues.

---

## 6. Genre profiles

Genres in the UI: **Dubstep, Bass House** (primary, defined in the brief) plus Pop, Rock, Hip Hop, EDM, Techno, Drum & Bass, Trap, Lo-fi, R&B, Indie.

Per genre store (JSON/YAML, versioned):
- Band-energy percentile ranges (P10/P50/P90) for each band
- Typical LUFS, crest, LRA ranges
- Typical stereo correlation/width ranges
- Average smoothed spectrum curve (16-point and fine-grained) used as the target for the EQ-band fit
- Detector thresholds and genre weights
- `sample_count` and `version`

**How to build them:** collect a licensed/consented set of reference tracks per genre (50+ each is a reasonable start), run the *same* analysis engine, compute percentiles. For the buildathon, finish **Dubstep and Bass House** properly and mark other genres `"profile": "generic-v0"` (clearly marked low confidence) until data exists. Do not claim precision you don't have.

---

## 7. Voice service

Pipeline: `question (text or audio) → [STT] → build context → LLM → [TTS] → response`.

1. **STT (only when audio is sent):** Whisper / faster-whisper or a hosted STT. The current UI uses the browser's speech recognition and sends **text**, so STT can come later.
2. **Context assembly:** load the stored analysis (`analysis_id`) and pass the relevant JSON (metrics, bands, stereo, obs, genre, reference deltas) plus the short conversation history for that analysis.
3. **LLM answer rules:** answer only from the provided analysis; if a value isn't present say it isn't measured; same guardrails as 5.6; keep answers short for speech (2-5 sentences); avoid reading long number lists aloud.
4. **TTS:** produce `audio_url` (mp3/wav) via a TTS provider; cache by text hash; set expiry; consider streaming later.
5. **Latency target:** under ~4 s for text-only, under ~7 s with audio. Add timeouts and friendly errors.
6. **Do not** implement a separate AI engine in the frontend. All reasoning is here.

---

## 8. Revision comparison
- Store each analysis with a stable `id`; `/compare` analyses the revised file with identical settings, then diffs.
- Observation tracking via the stable detector `id`: `resolved`, `persisted`, `new`, `worsened`, `improved_numerically` (based on severity change).
- Include clipping change, band deltas, stereo deltas, loudness deltas.
- Output the standing disclaimer: numbers only, not a quality judgement.

---

## 9. Storage, jobs and operations
- **DB:** PostgreSQL (or SQLite for the buildathon): `analyses`, `observations`, `compare_runs`, `voice_turns`, `genre_profiles`.
- **Files:** temp storage with TTL (e.g. delete uploads after 24 h unless the user opts in); hash files to cache repeat analyses.
- **Jobs:** analysis can take 5-30 s; run in a worker (RQ / Celery / ARQ / FastAPI `BackgroundTasks` for the MVP). Limit concurrency.
- **Security:** validate file type and size, sandbox decoding (ffmpeg with timeouts), rate-limit per IP, never execute uploaded content, never log API keys, privacy notice (users' unreleased music is sensitive).
- **Observability:** structured logs with analysis id, timings per stage, error codes.

---

## 10. Testing and validation
- **Metric accuracy:** synthetic signals (sine at known level, pink noise, sine sweep, hard-clipped sine, correlated/anti-phase stereo, mono) with expected results; compare LUFS/LRA/true peak against `ffmpeg ebur128` and a trusted meter (tolerance ±0.1 LU).
- **Determinism:** same file twice gives identical output.
- **Contract tests:** validate responses against a JSON Schema shared with the frontend; fail the build on drift.
- **Rule engine:** unit tests per detector with fixture metric sets (just-below, just-above threshold, missing fields).
- **LLM grounding tests:** numbers in text must exist in input; banned phrases ("better", "match the reference") never appear.
- **Human evaluation:** 5-10 producers review observations for usefulness and correctness; track false-positive rate per detector.

---

## 11. Suggested repo layout

```
backend/
  app/
    main.py                 # FastAPI app, CORS, routers
    api/{analyze,compare,voice,analysis}.py
    core/{config,errors,schemas}.py      # Pydantic models = the JSON contract
    dsp/{loader,levels,loudness,spectrum,stereo,clipping,eqfit}.py
    genre/{profiles/*.yaml,loader.py}
    ai/{detectors/*.py,ranker.py,templates.py,llm.py,validator.py}
    voice/{context.py,stt.py,tts.py}
    storage/{db.py,files.py}
    jobs/worker.py
  tests/{dsp,ai,api,fixtures}
  scripts/build_genre_profiles.py
  pyproject.toml / requirements.txt
```

---

## 12. Build order (buildathon priorities)

**Must have**
1. Pydantic schemas = JSON contract; `/analyze` returning real peak, RMS, LUFS, LRA, crest, clipping, band percentages.
2. Genre profiles for Dubstep and Bass House.
3. Detectors: `clipping`, `low_end_concentration`, `loudness_vs_genre`, `dynamics`; deterministic template text.

**Should have**
4. 16-point spectrum, stereo correlation/width/balance, reference measurements + `ref`, `spectrum.B`.
5. 8-band EQ fit (`eq.A`, `eq.B`).
6. LLM phrasing with validation; `/voice/ask` (text in, text out), then TTS.

**If time permits**
7. Mono compatibility details, `/compare` revision diff, time-based spectrum, kick/bass overlap detector, progress/SSE, other genre profiles with real data.

---

## 13. Open questions to settle with the frontend developer
1. 4 or 5 frequency bands in `bands`?
2. Exact spectrum point count (16) and normalization method.
3. Sign conventions: L/R balance and correlation meaning.
4. `mono` value set (`OK|CAUTION|PROBLEM`).
5. Does `/analyze` return immediately or a job id + status endpoint?
6. Is the reference's data returned inside the same `/analyze` response (current UI assumes yes)?
7. Voice: text-only first, or audio upload too? Where is TTS audio hosted?
8. Which genre names/codes are final (UI sends display names)?

---

## 14. Definition of done
- Every number shown in the UI comes from the backend and matches an independent meter within tolerance.
- No observation contains a number or claim that isn't in the analysis data.
- No text says a mix is better/worse or tells the user to imitate the reference.
- The frontend has no hardcoded or mock values in the final demo.
- Errors (bad file, too large, backend down) produce clear messages in the UI.
