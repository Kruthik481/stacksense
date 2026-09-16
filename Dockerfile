FROM python:3.12-slim AS builder

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PUBLIC_DEMO=true

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app backend/ ./backend/
COPY --chown=app:app frontend/ ./frontend/
RUN mkdir -p models && chown app:app models
USER app

# Bake the embedding model and a demo index of StackSense's own source into the image,
# so a fresh container answers questions without downloads or manual ingestion.
RUN python backend/build_demo.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend", "--workers", "1"]
