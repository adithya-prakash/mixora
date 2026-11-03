FROM python:3.11-slim

# Install system dependencies (FFmpeg for audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY backend/requirements.txt backend/

# Install python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy source code and frontend
COPY backend/ backend/
COPY index.html ./

# Environment defaults
ENV STORAGE_DIR=/app/data \
    DB_URL=sqlite:////app/data/mixora.sqlite3 \
    PORT=8000

# Create persistent storage dir
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
