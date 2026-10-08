FROM proyecto/ml-base AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# torch ya viene en la base (cu121): filtramos "torch" de requirements (CRLF-safe)
# para no reinstalar la variante de PyPI
RUN --mount=type=cache,target=/root/.cache/pip \
    tr -d '\r' < requirements.txt | grep -viE '^torch($|[[ =<>;])' > /tmp/req-no-torch.txt && \
    pip install --prefix=/install -r /tmp/req-no-torch.txt

FROM proyecto/ml-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    NLTK_DATA=/app/nltk_data

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data')" \
    && rm -rf /root/.cache/pip

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8086"]
