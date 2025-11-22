# Multi-stage Dockerfile for AI Salary Prediction Pipeline

# Base stage
FROM python:3.11-slim as base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Training stage
FROM base as training

COPY config.yaml .
COPY src/ ./src/

RUN mkdir -p data/raw data/processed models

ENTRYPOINT ["python", "-m", "src.main", "train"]


# API stage
FROM base as api

COPY config.yaml .
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Complete stage for development
FROM base as complete

COPY . .

RUN mkdir -p data/raw data/processed models

CMD ["bash"]
