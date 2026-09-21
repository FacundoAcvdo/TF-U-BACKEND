FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip3 install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -c "import nltk; nltk.download('stopwords')"
# torch GPU (RTX 3070) - si falla queda CPU
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 "torch" || echo "torch GPU no instalado - fallback CPU"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8086"]
