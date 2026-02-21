# Deployment Guide

## Architecture

| Layer | Service | Cost |
|-------|---------|------|
| Backend (FastAPI) | [Render](https://render.com) | Free tier (or $7/mo Starter for always-on) |
| Frontend (React) | [Cloudflare Pages](https://pages.cloudflare.com) | Free |
| Model storage | AWS S3 (`ai-salary-predictor`) | Existing |

The frontend proxies `/api/*` to the Render backend via a Cloudflare Pages `_redirects` rule, so all requests appear same-origin and no CORS config is needed.

---

## Backend: Render

### 1. Create an IAM user for Render

Render uses static AWS credentials (not OIDC), so create a dedicated read-only user:

**AWS Console → IAM → Users → Create user → name: `aisalary-render`**

Attach this inline policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::ai-salary-predictor",
      "arn:aws:s3:::ai-salary-predictor/models/*"
    ]
  }]
}
```

Create an access key (type: *Application running outside AWS*) and save the key ID and secret.

### 2. Deploy

1. Push `render.yaml` to the repo (already present at repo root)
2. Go to [render.com](https://render.com) → **New → Blueprint** → connect the repo
3. Render auto-detects `render.yaml` and creates the `aisalary-api` service
4. In the service **Environment** tab, add these two secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

On first deploy, `scripts/render_download_model.py` fetches the model from S3 before uvicorn starts. Subsequent deploys do the same.

### 3. Note your service URL

Render assigns a URL like `https://aisalary-api.onrender.com`. You need it for the next step.

> **Free tier note:** The service spins down after 15 min of inactivity. Cold starts take ~1–2 min (model download + load). Upgrade to Starter ($7/mo) for always-on with no spin-down.

---

## Frontend: Cloudflare Pages

### 1. Point the API proxy at your Render URL

Edit `frontend-react/public/_redirects` and replace the placeholder with your actual Render service URL:

```
/api/* https://aisalary-api.onrender.com/api/:splat 200
```

### 2. Deploy

1. **Cloudflare Dashboard → Pages → Create project → Connect to Git** → select this repo
2. Set build settings:

   | Setting | Value |
   |---------|-------|
   | Build command | `cd frontend-react && npm install && npm run build` |
   | Build output directory | `frontend-react/dist` |

3. Click **Save and Deploy**

Cloudflare Pages auto-deploys on every push to `main`.

---

## AWS S3 Setup (for setting up from scratch)

### Bucket structure

```
ai-salary-predictor/
├── data/raw/
├── data/processed/
└── models/
    └── latest/
        └── salary_model_latest.pkl
```

### Create bucket

```bash
aws s3 mb s3://ai-salary-predictor --region us-east-1
```

### OIDC for the ML training pipeline

The `pipeline-and-deploy.yml` GitHub Actions workflow uses OIDC role assumption (no stored secrets) to push trained models to S3. To set this up:

**AWS Console → IAM → Identity providers → Add provider:**
- Type: OpenID Connect
- URL: `https://token.actions.githubusercontent.com`
- Audience: `sts.amazonaws.com`

**Create IAM role with trust policy** (replace `ACCOUNT_ID` and repo path):

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
        "token.actions.githubusercontent.com:sub": "repo:jonasneves/aipi510-project3:*"
      }
    }
  }]
}
```

**Attach S3 permissions to the role:**

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

**Add GitHub secret:**
- Name: `AWS_ROLE_ARN`
- Value: `arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME`
