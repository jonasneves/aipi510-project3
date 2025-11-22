# Deployment Guide

## API Deployment

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/salary-api

# Deploy
gcloud run deploy salary-api \
  --image gcr.io/PROJECT_ID/salary-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS Lambda (with container)

```bash
# Build for Lambda
docker build --target api -t salary-api .

# Push to ECR and deploy via AWS Console or SAM
```

## Frontend Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file: `frontend/app.py`
5. Add secret: `API_URL=https://your-deployed-api.run.app`

### HuggingFace Spaces

1. Create new Space (Streamlit SDK)
2. Upload `frontend/app.py` and `requirements.txt`
3. Configure API_URL in Space settings

## S3 Configuration

See [AWS_SETUP.md](../AWS_SETUP.md) for OIDC authentication setup.

## CI/CD

GitHub Actions pipeline (`.github/workflows/ml-pipeline.yml`):

- Triggers on push to main
- Uses S3 cache for data (avoids redundant API calls)
- Uploads models to S3 with versioning

Force fresh data collection:
- Go to Actions > ML Pipeline > Run workflow
- Check "Force data collection from APIs"
