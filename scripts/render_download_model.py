"""Download salary model from S3 before server startup. Run by Render's startCommand."""

import os
import sys
from pathlib import Path

import boto3

bucket = os.environ.get("S3_BUCKET", "ai-salary-predictor")
key = "models/latest/salary_model_latest.pkl"
dest = Path("models/salary_model_latest.pkl")

dest.parent.mkdir(exist_ok=True)

print(f"Downloading model from s3://{bucket}/{key}...")
try:
    boto3.client("s3").download_file(bucket, key, str(dest))
    print(f"Done ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
except Exception as e:
    print(f"ERROR: Failed to download model: {e}", file=sys.stderr)
    sys.exit(1)
