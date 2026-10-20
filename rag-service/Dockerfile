FROM proyecto/ml-base AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# torch ya viene en la base (cu121): sentence-transformers lo ve satisfecho y no trae otra variante
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

FROM proyecto/ml-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
RUN mkdir -p /app/data/rag_docs

CMD ["uvicorn", "--no-access-log", "main:app", "--host", "0.0.0.0", "--port", "8088"]
