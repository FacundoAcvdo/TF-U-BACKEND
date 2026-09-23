FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Modelos de spaCy para inglés y español
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download es_core_news_sm \
    && python -m nltk.downloader punkt punkt_tab

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8089"]