# AI Salary Prediction Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Pipeline](https://github.com/jonasneves/aipi510-project3/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/jonasneves/aipi510-project3/actions/workflows/ml-pipeline.yml)
[![API Status](https://img.shields.io/endpoint?url=https://aisalary.neevs.io/api/badge/api&label=API)](https://aisalary.neevs.io/api)
[![App Status](https://img.shields.io/endpoint?url=https://aisalary.neevs.io/api/badge/app&label=App)](https://aisalary.neevs.io)

Predict AI/ML salaries using XGBoost, FastAPI, React, and MLFlow.

**Live Demo:** [aisalary.neevs.io](https://aisalary.neevs.io) | [API Docs](https://aisalary.neevs.io/api/docs)

## Features

- **Resume Upload** - Drag & drop your resume (PDF/DOCX) to auto-fill job details
- **Real-time Predictions** - Salary updates instantly as you change inputs
- **Smart Parsing** - Extracts job title, skills, experience, and location from resumes
- **Confidence Intervals** - Shows 90% CI salary range, not just point estimate

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
frontend-react/       # React + Vite + Tailwind frontend
config.yaml           # Pipeline configuration
Makefile              # Convenience commands
Dockerfile            # Multi-stage builds
docker-compose.yml    # Service orchestration
```

## API

```bash
# Predict salary
curl -X POST https://aisalary.neevs.io/api/predict \
  -H "Content-Type: application/json" \
  -d '{"job_title": "ML Engineer", "location": "CA", "experience_years": 5}'

# Parse resume
curl -X POST https://aisalary.neevs.io/api/parse-resume \
  -F "file=@resume.pdf"

# Get options
curl https://aisalary.neevs.io/api/options
```

## Documentation

- [Setup Guide](docs/SETUP.md) - Local development
- [Deployment Guide](docs/DEPLOYMENT.md) - Cloud deployment
- [Cloudflare Tunnel](docs/CLOUDFLARE_TUNNEL.md) - Hosting setup
- [AWS Setup](AWS_SETUP.md) - S3 and OIDC configuration

## License

MIT
