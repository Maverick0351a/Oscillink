# Simple development Dockerfile for Oscillink Cloud API
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

WORKDIR /app

# System deps (add build-essential if future native deps appear)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 10001 appuser

COPY pyproject.toml README.md ./
COPY oscillink ./oscillink
COPY cloud ./cloud

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[cloud-all]

# Runtime env
ENV PORT=8000 \
    UVICORN_WORKERS=2 \
    UVICORN_TIMEOUT=60 \
    OSCILLINK_FORCE_HTTPS=1

EXPOSE ${PORT}

# Basic healthcheck hitting /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

# Switch to non-root
USER appuser

# Use gunicorn with uvicorn workers for production
CMD gunicorn cloud.app.main:app \
    --bind 0.0.0.0:${PORT} \
    --workers ${UVICORN_WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout ${UVICORN_TIMEOUT} \
    --access-logfile '-' --error-logfile '-'
