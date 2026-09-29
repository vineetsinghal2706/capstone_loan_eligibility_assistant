# Dockerfile — Session 6 model server (image: llm-server:openai)
# Same container discipline as Session 2 — the only functional change is the
# app: it now serves the OpenAI protocol on port 8090.

# 1. BASE IMAGE: slim, pinned version. Not 'latest' (unpredictable),
#    not full python (bloated), not a CUDA image (we serve on CPU here).
FROM python:3.11-slim

# 2. Match the host user for mounted-volume write access.
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd --gid ${HOST_GID} appuser \
  && useradd --uid ${HOST_UID} --gid ${HOST_GID} --create-home appuser

WORKDIR /app

# 3. Copy requirements FIRST — Docker caches this layer, so editing app.py
#    later doesn't re-install packages (layer caching discipline)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
#   ^ CPU-only torch wheel: ~200 MB instead of ~2.5 GB with CUDA.

# 4. PRE-BAKE the model into the image (trade-off discussed in Session 2:
#    faster cold start vs ~1 GB larger image; big models use a volume instead)
ENV HF_HOME=/app/hf-cache
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')" \
    && chown -R appuser /app

# 5. App code goes LAST — it changes most often, so it invalidates
#    the fewest cached layers.
COPY app.py .

USER appuser

# 6. ClaimAssist convention (Sessions 6-10): the model server owns port 8090.
#    $PORT keeps the image cloud-ready — managed runtimes inject their own.
ENV PORT=8090

# 7. Container-native health check: Docker marks the container "healthy"
#    only after the model has loaded and /health responds.
#    start-period allows up to ~2 min for model load before checks count.
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=20 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')" || exit 1

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
