FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Python + build-essential en una sola capa (se purga al final del pip install)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip3 install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir -r requirements.txt && \
    (pip3 install --no-cache-dir --extra-index-url=https://pypi.nvidia.com "cuml-cu12==24.12.*" "cupy-cuda12x" || echo "[clustering-service] cuML no instalado - fallback CPU") && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
