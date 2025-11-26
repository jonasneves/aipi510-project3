# Setup Guide

## Prerequisites

- Python 3.11+
- Docker (optional)

## Installation

```bash
git clone https://github.com/jonasneves/aipi510-project3.git
cd aipi510-project3

python -m venv venv
source venv/bin/activate

make install
```

## Environment Variables

Create `.env` from template:

```bash
cp .env.example .env
```

Required keys:
- `ADZUNA_APP_ID` / `ADZUNA_API_KEY` - [Adzuna API](https://developer.adzuna.com/)
- `AWS_PROFILE` - AWS credentials for LinkedIn data (S3)

## Running the Pipeline

```bash
make collect   # Fetch data from APIs
make merge     # Process and combine sources
make train     # Train model (logs to MLFlow)
```

Or all at once:

```bash
make pipeline
```

## Services

```bash
make api       # FastAPI at localhost:8000
make frontend  # React at localhost:5173
make mlflow    # MLFlow UI at localhost:5000
```

## Docker

```bash
docker-compose build
docker-compose up -d api frontend
```

## Available Commands

```bash
make help      # Show all commands
```
