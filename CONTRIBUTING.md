# Contributing to Mixora ✦

Thank you for your interest in contributing to **Mixora**! Whether you're fixing bugs, improving the DSP audio algorithms, enhancing the Voice AI interaction, or polishing the frontend UI, your contributions are welcome.

---

## 🛠️ Local Development Setup

### 1. Prerequisites
- **Python 3.11+** installed
- **FFmpeg & ffprobe** installed and accessible in your system `PATH` (required for decoding MP3s; uncompressed WAV works out of the box)
- A modern web browser (Chrome, Edge, Safari, or Firefox)
- *(Optional)* A free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/apikey) for real-time voice interactions.

### 2. Clone the Repository
```bash
git clone https://github.com/adithya-prakash/mixora.git
cd mixora
```

### 3. Setup Backend
Create and activate a virtual environment:

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements-dev.txt
```

### 4. Configure Environment
Copy the example environment file:
```bash
cp backend/.env.example .env
```
*(Optional)* Add your `GEMINI_API_KEY`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Services

**Start the FastAPI Backend:**
```bash
uvicorn backend.app.main:app --reload --port 8000
```
Interactive API documentation will be available at `http://127.0.0.1:8000/docs`.

**Serve the Frontend:**
In a separate terminal, serve `index.html`:
```bash
# Using Python
python -m http.server 3000

# Or using Node
npx serve -l 3000 .
```
Visit `http://localhost:3000` in your browser.

---

## 🧪 Running Tests

Ensure all unit and DSP measurement tests pass before submitting changes:

```bash
python -m unittest discover -s backend/tests -v
```

Tests cover:
- **`test_dsp.py`**: Sine tone levels, LUFS gating, frequency energy summation, clipping checks, antiphase stereo detection.
- **`test_audio.py`**: Audio decoding, resampling accuracy (44.1 kHz), duration limits, and format rejection.
- **`test_api.py`**: FastAPI endpoint status codes, contract schema compliance, and payload validation.

---

## 📐 Coding Standards & Guidelines

- **Type Hints:** All Python code must include modern type annotations (`from __future__ import annotations`).
- **Vectorized NumPy:** All DSP computations (spectral energy, filters, resampling) must use vectorized NumPy or SciPy routines rather than slow Python loops.
- **Preserve Contract:** The API contract is defined in `api-contract-v1.schema.json`. If you alter an endpoint response, update the schema and contract tests in the same PR.
- **Clean Git Commits:** Write clear, concise commit messages following standard conventions (`feat: ...`, `fix: ...`, `docs: ...`, `perf: ...`).

---

## 🚀 Pull Request Process

1. **Fork** the repository and create your feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Commit your changes with clear messages.
3. Verify tests pass: `python -m unittest discover -s backend/tests -v`.
4. Push to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```
5. Open a **Pull Request** against the `main` branch. Provide a concise summary of what changed and any relevant screenshots or test results.

---

## 💬 Community & Discussions

Have questions, suggestions, or feature requests? Open an issue on GitHub or join our community discussions. We'd love to hear your feedback!
