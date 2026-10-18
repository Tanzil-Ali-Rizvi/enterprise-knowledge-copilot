FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[neural,ui]"

COPY data ./data
COPY ui ./ui
COPY evals ./evals

# Run as non-root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV KC_EMBEDDER=fastembed KC_RERANKER=cross-encoder KC_INDEX_DIR=/app/.index
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "kcopilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
