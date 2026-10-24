FROM proyecto/ml-base AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    NLTK_DATA=/app/nltk_data

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

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

CMD ["uvicorn", "--no-access-log", "main:app", "--host", "0.0.0.0", "--port", "8080"]
