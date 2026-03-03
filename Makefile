.DEFAULT_GOAL := help

.PHONY: help install collect merge train predict pipeline api mlflow frontend frontend-dev frontend-build test docker-build docker-up docker-down format lint clean setup s3-backup

help:
	@echo ""
	@echo "\033[2mPipeline\033[0m"
	@echo "  \033[36minstall\033[0m          Install Python dependencies"
	@echo "  \033[36mcollect\033[0m          Collect data from all sources"
	@echo "  \033[36mmerge\033[0m            Merge collected data"
	@echo "  \033[36mtrain\033[0m            Train the salary prediction model"
	@echo "  \033[36mpredict\033[0m          Run interactive prediction"
	@echo "  \033[36mpipeline\033[0m         Run complete pipeline (collect + merge + train)"
	@echo ""
	@echo "\033[2mServers\033[0m"
	@echo "  \033[36mapi\033[0m              Start FastAPI server (port 8000)"
	@echo "  \033[36mmlflow\033[0m           Start MLFlow UI (port 5000)"
	@echo "  \033[36mfrontend\033[0m         Start React frontend dev server (port 5173)"
	@echo "  \033[36mfrontend-build\033[0m   Build React frontend for production"
	@echo ""
	@echo "\033[2mDocker\033[0m"
	@echo "  \033[36mdocker-build\033[0m     Build Docker image"
	@echo "  \033[36mdocker-up\033[0m        Start services with Docker Compose"
	@echo "  \033[36mdocker-down\033[0m      Stop Docker services"
	@echo ""
	@echo "\033[2mUtility\033[0m"
	@echo "  \033[36mtest\033[0m             Run tests"
	@echo "  \033[36mclean\033[0m            Remove generated files and cache"
	@echo "  \033[36mformat\033[0m           Format code with black and isort"
	@echo "  \033[36mlint\033[0m             Run code linters"
	@echo "  \033[36ms3-backup\033[0m        Download complete S3 bucket backup"
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
