# AI Salary Prediction Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Pipeline](https://github.com/jonasneves/aipi510-project3/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/jonasneves/aipi510-project3/actions/workflows/ml-pipeline.yml)
[![API Status](https://img.shields.io/endpoint?url=https://aisalary.neevs.io/api/badge/api&label=API)](https://aisalary.neevs.io/api)
[![App Status](https://img.shields.io/endpoint?url=https://aisalary.neevs.io/api/badge/app&label=App)](https://aisalary.neevs.io)

**Live Demo:** [aisalary.neevs.io](https://aisalary.neevs.io) | [API Docs](https://aisalary.neevs.io/api/docs)

## Overview

Predict AI/ML salaries using machine learning. Built for Duke AIPI 510 Module Project 3.

**Problem:** Estimate salary ranges for AI/ML roles based on job title, location, experience, and skills.

**Solution:** XGBoost regression model trained on H1B visa filings and BLS wage data, deployed as a FastAPI service with a React frontend.

## Dataset

| Source | Description | Records |
|--------|-------------|---------|
| [H1B Visa Data](https://www.dol.gov/agencies/eta/foreign-labor/performance) | DOL certified visa applications with actual salaries | ~50k AI/ML jobs |
| [BLS OES](https://www.bls.gov/oes/) | Occupational wage statistics by state | Benchmark data |

Data hosted on AWS S3. Pipeline downloads and merges sources automatically.

## Model

**Architecture:** XGBoost Regressor

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |

**Evaluation Metrics:**
- MAE: ~$15,000
- RMSE: ~$22,000
- R²: ~0.65

**Key Features:** Job title seniority, state location, years of experience, skills (Python, PyTorch, Kubernetes, etc.)

## Architecture

![Architecture Diagram](architecture.png)

## Quick Start

```bash
make install           # Install Python dependencies
make frontend-install  # Install frontend dependencies
make pipeline          # Collect data, merge, train
make api               # Start API (port 8000)
make frontend          # Start React dev server (port 5173)
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML | XGBoost, scikit-learn, pandas |
| API | FastAPI, Pydantic |
| Frontend | React, Vite, Tailwind CSS |
| Tracking | MLFlow |
| Storage | AWS S3 |
| Hosting | GitHub Actions + Cloudflare Tunnel |

## Project Structure

```
src/                  # ML pipeline (collectors, processing, models)
api/                  # FastAPI endpoints
frontend-react/       # React frontend
config.yaml           # Pipeline configuration
Makefile              # Commands
Dockerfile            # Container build
```

## API

```bash
# Predict salary
curl -X POST https://aisalary.neevs.io/api/predict \
  -H "Content-Type: application/json" \
  -d '{"job_title": "ML Engineer", "location": "CA", "experience_years": 5}'

# Get options
curl https://aisalary.neevs.io/api/options
```

## Documentation

- [Setup Guide](docs/SETUP.md) - Local development
- [Deployment Guide](docs/DEPLOYMENT.md) - Cloud deployment
- [AWS Setup](AWS_SETUP.md) - S3 configuration

## Limitations

- **Geographic bias:** H1B data skews toward CA, NY, WA where most visa sponsors operate
- **Role coverage:** Limited to AI/ML titles; doesn't cover adjacent roles well
- **Temporal lag:** H1B filings reflect offers made 6-12 months prior
- **Company representation:** Large tech companies overrepresented vs. startups

## License

MIT
