FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/app/nltk_data

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
# stopwords (~few MB) se descargan en runtime-stage para no ensuciar la capa builder
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data')" \
    && rm -rf /root/.cache/pip

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8084"]
