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


# API stage (optimized with minimal runtime dependencies)
FROM python:3.11-slim as api

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Use minimal runtime requirements for faster builds
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY config.yaml .
COPY configs/ ./configs/
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Pre-compile Python bytecode for faster startup
RUN python -m compileall -q api/ src/ || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Frontend build stage
FROM node:20-alpine as frontend-build

WORKDIR /app

COPY frontend-react/package*.json ./
RUN npm install

COPY frontend-react/ ./
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Frontend serve stage
FROM nginx:alpine as frontend

COPY --from=frontend-build /app/dist /usr/share/nginx/html
COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen 8501;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
    }
}
EOF

EXPOSE 8501

CMD ["nginx", "-g", "daemon off;"]


# Complete stage for development
FROM base as complete

COPY . .

RUN mkdir -p data/raw data/processed models

CMD ["bash"]
