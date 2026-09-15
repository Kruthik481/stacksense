FROM python:3.12-slim AS builder

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
# CPU-only torch first: the default PyPI wheel pulls ~2.5GB of CUDA libraries this image never uses.
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

# Hugging Face Spaces runs containers as UID 1000, so use that user everywhere.
RUN useradd --create-home --uid 1000 app

ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/home/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PUBLIC_DEMO=true

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app backend/ ./backend/
COPY --chown=app:app frontend/ ./frontend/
USER app

# Bake the embedding model and a demo index of StackSense's own source into the image,
# so a fresh container answers questions without downloads or manual ingestion.
RUN mkdir -p backend/data && cd backend && python ingest.py /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend", "--workers", "1"]
