FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

FROM base AS deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir "numpy>=2.0"

FROM deps AS runtime
COPY ie ./ie
RUN python -c "import ie.engine.engine"

RUN useradd --system --uid 10001 serving && chown -R serving /srv
USER serving

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=3 \
    CMD python -m ie.serve.health || exit 1

ENTRYPOINT ["python", "-m", "ie.serve.health"]
