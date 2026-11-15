FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    STORAGE_DIR=/app/storage \
    STATIC_DIR=/app/static

WORKDIR /app

# FFmpeg for muxing, espeak-ng as the offline narration fallback, fonts for title cards and web pages,
# plus the Chromium runtime libraries (playwright install --with-deps adds anything missing).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    curl \
    ca-certificates \
    fonts-liberation \
    fonts-inter \
    fonts-dejavu-core \
    fonts-noto-core \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install chromium --with-deps \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app
# Optional: the dashboard's static export (deploy.sh builds it); served from / when present.
COPY static* ./static/

EXPOSE 8080

# One worker only: jobs, the SQLite database and the recording slot live in this process.
# Only Cloud Run's proxy can reach the container, so its X-Forwarded-* headers are trusted (https:// redirects).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --proxy-headers --forwarded-allow-ips='*' --timeout-graceful-shutdown 5"]
