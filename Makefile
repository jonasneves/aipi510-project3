# Makefile for AI Salary Prediction Pipeline

.PHONY: help install collect merge train predict pipeline api mlflow frontend frontend-dev frontend-build test docker-build docker-up docker-down format lint clean setup s3-backup

help:
	@echo "AI Salary Prediction Pipeline - Available Commands:"
	@echo ""
	@echo "  make install      - Install Python dependencies"
	@echo "  make collect      - Collect data from all sources"
	@echo "  make merge        - Merge collected data"
	@echo "  make train        - Train the salary prediction model"
	@echo "  make predict      - Run interactive prediction"
	@echo "  make pipeline     - Run complete pipeline (collect + merge + train)"
	@echo "  make mlflow       - Start MLFlow UI (port 5000)"
	@echo "  make api          - Start FastAPI server (port 8000)"
	@echo "  make frontend     - Start React frontend dev server (port 5173)"
	@echo "  make frontend-build - Build React frontend for production"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start services with Docker Compose"
	@echo "  make docker-down  - Stop Docker services"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Remove generated files and cache"
	@echo "  make format       - Format code with black and isort"
	@echo "  make lint         - Run code linters"
	@echo "  make s3-backup    - Download complete S3 bucket backup"
	@echo ""

install:
	pip install -r requirements.txt

collect:
	python -m src.main collect --source all

merge:
	python -m src.main merge

train:
	python -m src.main train

predict:
	python -m src.main predict

pipeline: collect merge train

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

frontend:
	cd frontend-react && npm run dev

frontend-build:
	cd frontend-react && npm run build

frontend-install:
	cd frontend-react && npm install

test:
	pytest tests/ -v

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

format:
	black src/ api/ --line-length 100
	isort src/ api/

lint:
	flake8 src/ api/ --max-line-length=100

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/raw/*.parquet data/processed/*.parquet 2>/dev/null || true

setup: install collect merge train
	@echo "Setup complete. Run 'make predict' to test predictions."

s3-backup:
	@bash scripts/download_s3_backup.sh
