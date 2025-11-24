# Deployment Guide

## Cloud Hosting (GitHub Actions + Cloudflare)

This project uses GitHub Actions runners with Cloudflare Tunnel for hosting. The React frontend and FastAPI backend run on the same runner.

### Prerequisites

1. Cloudflare account with domain
2. AWS S3 bucket with trained model
3. GitHub secrets configured

### Setup Steps

#### 1. Create Cloudflare Tunnel

In Cloudflare Zero Trust dashboard:
1. Go to **Networks** > **Tunnels** > **Create a tunnel**
2. Name it (e.g., `salary-predictor`)
3. Copy the tunnel token

#### 2. Configure Public Hostname

| Setting | Value |
|---------|-------|
| Subdomain | `aisalary` |
| Domain | your domain |
| Service | `HTTP://localhost:8501` |

#### 3. Add GitHub Secrets

| Secret | Description |
|--------|-------------|
| `CLOUDFLARE_TUNNEL_TOKEN` | Tunnel token |
| `AWS_ROLE_ARN` | IAM role ARN |

### Usage

```bash
# Start hosting
gh workflow run tunnel-hosting.yml

# Custom duration
gh workflow run tunnel-hosting.yml -f duration_hours=2
```

---

## AWS S3 Setup

### Bucket Structure

```
ai-salary-predictor/
├── data/raw/
├── data/processed/
└── models/
```

### Setup Steps

#### 1. Create Bucket

```bash
aws s3 mb s3://ai-salary-predictor --region us-east-1
```

#### 2. Create OIDC Provider

AWS Console: IAM > Identity providers > Add provider

- **Type**: OpenID Connect
- **URL**: `https://token.actions.githubusercontent.com`
- **Audience**: `sts.amazonaws.com`

#### 3. Create IAM Role

Trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:YOUR_USER/YOUR_REPO:*"
      }
    }
  }]
}
```

#### 4. Attach S3 Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::ai-salary-predictor",
      "arn:aws:s3:::ai-salary-predictor/*"
    ]
  }]
}
```

#### 5. Add GitHub Secret

- **Name**: `AWS_ROLE_ARN`
- **Value**: `arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME`
