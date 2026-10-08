FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# torch CPU primero para que transformers no arrastre otra variante
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

EXPOSE 8091

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8091"]
