# Simple development Dockerfile for Oscillink Cloud API
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps (add build-essential if future native deps appear)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY oscillink ./oscillink
COPY cloud ./cloud

RUN pip install --upgrade pip && pip install .[cloud-all]

ENV PORT=8000
EXPOSE ${PORT}
CMD ["bash", "-lc", "uvicorn cloud.app.main:app --host 0.0.0.0 --port ${PORT}"]
