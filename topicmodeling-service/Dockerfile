FROM python:3.12-slim

# Evita que Python bufferice logs y no guarde bytecode innecesario
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1. Herramientas del sistema necesarias para compilar hdbscan/umap
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Pre-descargar las stopwords en la imagen (para que el contenedor arranque al instante)
RUN python -m nltk.downloader stopwords

# 4. Copiar el código fuente
COPY . .

# 5. Puerto y comando de inicio
EXPOSE 8088
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]