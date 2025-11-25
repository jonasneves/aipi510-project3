#!/bin/bash
# Download/sync complete S3 bucket backup to local directory

set -e

# Configuration
S3_BUCKET="ai-salary-predictor"
BACKUP_DIR="s3_data_backup"
AWS_PROFILE="${AWS_PROFILE:-AdministratorAccess-950495744806}"

echo "--- S3 Bucket Backup Download ---"
echo "Bucket: s3://$S3_BUCKET"
echo "Target: $BACKUP_DIR/"
echo "AWS Profile: $AWS_PROFILE"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Export AWS profile
export AWS_PROFILE="$AWS_PROFILE"

# Sync entire bucket
echo ""
echo "Downloading all data from S3..."
aws s3 sync "s3://$S3_BUCKET/" "$BACKUP_DIR/" \
    --exclude ".git/*" \
    --exclude "*.pyc" \
    --exclude "__pycache__/*"

# Get bucket size
BUCKET_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)

echo ""
echo "Download Complete! (Total size: $BUCKET_SIZE)"
echo "Location: $BACKUP_DIR/"
echo ""
