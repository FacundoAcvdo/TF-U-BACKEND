FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Python 3.12 en base CUDA (no trae python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip3 install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --extra-index-url=https://pypi.nvidia.com "cuml-cu12==24.12.*" "cupy-cuda12x[ctk]" || pip install --no-cache-dir --extra-index-url=https://pypi.nvidia.com "cuml-cu12==24.12.*" "cupy-cuda12x" || echo "[clustering-service] cuML no instalado - fallback CPU"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]