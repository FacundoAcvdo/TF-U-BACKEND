# Base GPU compartida: torch cu121 instalado UNA vez.
# dataset/classification/rag hacen FROM de esta imagen y Docker comparte
# la capa de ~6GB (torch + nvidia) en disco en vez de triplicarla.
# Buildear primero: docker build -f TF-U-BACKEND/ml-base/Dockerfile -t proyecto/ml-base TF-U-BACKEND/ml-base
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Solo torch: el resto (transformers, chromadb, ...) lo instala cada servicio
# para evitar mezclar versiones al superponer /usr/local.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1"
