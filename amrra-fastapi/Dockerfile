# Dockerfile for API/Worker
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]