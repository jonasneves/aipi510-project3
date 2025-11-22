# AI Salary Prediction Pipeline

Reproducible ML pipeline for predicting AI/ML salaries. Built with XGBoost, FastAPI, Streamlit, and MLFlow.

**Live Demo:** [Frontend App](https://your-app.streamlit.app) | [API Endpoint](https://your-api.run.app)

## Overview

This project predicts salaries for AI/ML roles based on:
- Job title and seniority level
- Location (US states)
- Years of experience
- Company tier (FAANG, startups, etc.)
- Technical skills

## Quick Start

```bash
# Install
make install

# Run full pipeline
make pipeline

# Or run steps individually
make collect   # Collect data from APIs
make merge     # Merge data sources
make train     # Train model (logs to MLFlow)

# Start services
make api       # API at localhost:8000
make frontend  # UI at localhost:8501
make mlflow    # MLFlow at localhost:5000
```

## Data Sources

| Source | Description | Records |
|--------|-------------|---------|
| [H1B Visa Data](https://www.dol.gov/agencies/eta/foreign-labor/performance) | DOL disclosure data with exact salaries | ~14,000 |
| [Adzuna API](https://developer.adzuna.com/) | Job postings with salary ranges | ~2,700 |
| [BLS](https://www.bls.gov/developers/) | National wage benchmarks | 7 |
| Google Trends | Regional interest data | ~100 |

Data is stored in AWS S3 (`s3://ai-salary-predictor/`) and cached to avoid redundant API calls.

## Model

**Algorithm:** XGBoost Regressor

**Features (27 total):**
- Skill indicators: deep_learning, nlp, computer_vision, mlops, cloud_ml, big_data
- Experience level: is_senior, is_lead, is_principal, is_staff, is_director
- Role type: engineer, scientist, analyst, architect, researcher
- Location: cost-of-living multiplier, tech hub flags
- Company tier: FAANG, tier-1, finance, startup

**Evaluation Metrics:**
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R2: Coefficient of Determination
- MAPE: Mean Absolute Percentage Error

All experiments are tracked in MLFlow.

## Project Structure

```
.
├── src/
│   ├── data_collectors/    # H1B, BLS, Adzuna, Trends
│   ├── processing/         # Feature engineering, data merging
│   ├── models/             # XGBoost predictor
│   └── main.py             # CLI entry point
├── api/
│   └── main.py             # FastAPI endpoints
├── frontend/
│   └── app.py              # Streamlit UI
├── config.yaml             # Pipeline configuration
├── Makefile                # Convenience commands
├── Dockerfile              # Multi-stage builds
└── docker-compose.yml      # Service orchestration
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "Senior ML Engineer",
    "location": "CA",
    "experience_years": 5,
    "company": "Google",
    "skills": ["pytorch", "nlp"]
  }'
```

## Local Development

### Prerequisites
- Python 3.11+
- Docker (optional)

### Setup
```bash
# Clone repository
git clone https://github.com/jonasneves/aipi510-project3.git
cd aipi510-project3

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
make install

# Set up API keys in .env
cp .env.example .env
# Edit .env with your keys
```

### Environment Variables
```
BLS_API_KEY=your_bls_key
ADZUNA_APP_ID=your_adzuna_id
ADZUNA_API_KEY=your_adzuna_key
AWS_ROLE_ARN=arn:aws:iam::xxx:role/xxx  # For S3 access
```

### Docker
```bash
# Build and run all services
docker-compose up -d

# Or run individually
docker-compose up api
docker-compose up frontend
```

## Cloud Deployment

### API Deployment (Google Cloud Run)

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/salary-api

# Deploy
gcloud run deploy salary-api \
  --image gcr.io/PROJECT_ID/salary-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Frontend Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from `frontend/app.py`
4. Set `API_URL` secret to your deployed API endpoint

See [AWS_SETUP.md](AWS_SETUP.md) for S3 configuration with OIDC.

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`):

1. **Collect** - Fetch data from APIs (or use S3 cache)
2. **Merge** - Process and combine data sources
3. **Train** - Train model with MLFlow tracking
4. **Upload** - Save artifacts to S3

The pipeline uses S3 as a cache to avoid redundant API calls. Use `workflow_dispatch` with `force_collect: true` to refresh data.

## MLFlow Tracking

```bash
# Start MLFlow UI
make mlflow
# Open http://localhost:5000
```

Tracked per run:
- Parameters: model hyperparameters, dataset size
- Metrics: train/val/test MAE, RMSE, R2, MAPE
- Artifacts: model.pkl, feature_importance.csv

## Testing

```bash
make test
```

## Configuration

All settings in `config.yaml`:

```yaml
model:
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1

mlflow:
  experiment_name: "salary_prediction"
  tracking_uri: "file:./mlruns"

s3:
  bucket: "ai-salary-predictor"
  region: "us-east-1"
```

## License

MIT
