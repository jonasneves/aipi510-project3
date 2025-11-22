# Deployment Guide

## API Deployment (AWS)

### Option 1: AWS App Runner

```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker build --target api -t salary-api .
docker tag salary-api:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/salary-api:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/salary-api:latest

# Deploy via AWS Console: App Runner > Create Service > Container Registry
```

### Option 2: AWS Lambda + API Gateway

```bash
# Build container
docker build --target api -t salary-api .

# Push to ECR, then create Lambda function from container image
# Configure API Gateway HTTP API as trigger
```

## Frontend Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository, set main file: `frontend/app.py`
4. Add secret: `API_URL=https://your-api-endpoint.amazonaws.com`

### HuggingFace Spaces

1. Create new Space (Streamlit SDK)
2. Upload `frontend/app.py`
3. Set `API_URL` in Space settings

## S3 Configuration

See [AWS_SETUP.md](../AWS_SETUP.md) for OIDC authentication.

## CI/CD

GitHub Actions (`.github/workflows/ml-pipeline.yml`):
- Triggers on push to main
- Uses S3 cache for data
- Uploads models to S3 with versioning

Force fresh data: Actions > ML Pipeline > Run workflow > Check "Force data collection"
