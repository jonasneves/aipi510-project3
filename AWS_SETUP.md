# AWS S3 Setup for ML Pipeline

## S3 Bucket Structure

```
ai-salary-predictor/
├── data/
│   ├── raw/              # Raw collected data
│   └── processed/        # Merged/processed data
└── models/
    ├── latest/           # Most recent model
    └── YYYYMMDD_HHMMSS/  # Versioned models
```

## Setup Steps

### 1. Create S3 Bucket

```bash
aws s3 mb s3://ai-salary-predictor --region us-east-1
```

### 2. Create OIDC Identity Provider

AWS Console: IAM > Identity providers > Add provider

- **Provider type**: OpenID Connect
- **Provider URL**: `https://token.actions.githubusercontent.com`
- **Audience**: `sts.amazonaws.com`

### 3. Create IAM Role

Trust policy (replace `YOUR_ACCOUNT_ID`):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:jonasneves/aipi510-project3:*"
      }
    }
  }]
}
```

### 4. Attach S3 Permissions

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

### 5. Add GitHub Secret

GitHub repo > Settings > Secrets and variables > Actions:

- **Name**: `AWS_ROLE_ARN`
- **Value**: `arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_ROLE_NAME`
