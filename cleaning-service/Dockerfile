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
    NLTK_DATA=/app/nltk_data \
    SPACY_DATA=/app/spacy_data

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
# Modelos spaCy + NLTK en la etapa final (necesarios en runtime).
# Van a volumen en prod si se quiere, pero no quedan en la capa builder.
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download es_core_news_sm \
    && python -m nltk.downloader -d /app/nltk_data punkt punkt_tab \
    && rm -rf /root/.cache/pip

CMD ["uvicorn", "--no-access-log", "main:app", "--host", "0.0.0.0", "--port", "8089"]
