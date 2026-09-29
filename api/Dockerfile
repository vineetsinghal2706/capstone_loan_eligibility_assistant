# Dockerfile — LoanAssist API v2
# Adds new modules: calculators.py, rules_engine.py, profile_extractor.py
# Copies prompts/registry.yaml so the registry is available inside the container.
FROM python:3.11-slim

ARG HOST_UID=1000
ARG HOST_GID=1000

RUN groupadd --gid ${HOST_GID} appuser \
    && useradd --uid ${HOST_UID} --gid ${HOST_GID} --create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /home/appuser/.cache/chroma \
    && chown -R appuser /home/appuser/.cache/chroma

RUN mkdir -p /app/logs \
    && chown -R appuser /app/logs

# API modules
COPY app.py guardrails.py redact.py calculators.py rules_engine.py profile_extractor.py ./

# Prompt registry (needed by loader.py at startup)
COPY prompts ./prompts/

RUN chown -R appuser /app

USER appuser

ENV PORT=8001

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
