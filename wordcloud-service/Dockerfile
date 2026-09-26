FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8084"]