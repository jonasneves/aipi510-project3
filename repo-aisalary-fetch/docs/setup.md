# AI Salary Data Collection

Automated pipeline for collecting AI job salary data from LinkedIn.

## Features

- Incremental saving to JSONL format
- 15+ features extracted per job
- Automated GitHub Actions workflow
- S3 storage with AWS OIDC authentication
- Daily scheduled runs with manual trigger option

## Data Format

Each job is saved as a JSON line with the following structure:

```json
{
  "collected_at": "2025-11-24T10:30:00Z",
  "position": "Machine Learning Engineer",
  "company": "Acme AI",
  "location": "San Francisco, CA",
  "posted_date": "2025-11-23",
  "job_url": "https://...",
  "salary": "$150,000/yr - $200,000/yr",
  "seniority_level": "Mid-Senior level",
  "employment_type": "Full-time",
  "job_function": "Engineering and IT",
  "industries": "Software Development",
  "applicant_count": 150,
  "experience_years": "3-5 years",
  "education": "Bachelors",
  "skills": ["python", "tensorflow", "aws", "docker"],
  "company_logo": "https://...",
  "description": "..."
}
```

## Local Usage

Run data collection:

```bash
npm run collect
```

Output is saved to `./data/ai-jobs-TIMESTAMP.jsonl`

Customize search queries by editing the `searchQueries` array in `scripts/collect-data.js`:

```javascript
const searchQueries = [
  {
    keyword: "machine learning engineer",
    location: "United States",
    salary: "100000",
    limit: "50",
  },
  // Add more queries...
];
```

## Workflow Options

Two GitHub Actions workflows are available:

### Sequential Collection
- File: `.github/workflows/collect-salary-data.yml`
- Runs queries one after another
- Use for: Testing, small runs (<10 queries)
- Time: ~45-60 minutes for 20 queries

### Parallel Collection (Recommended)
- File: `.github/workflows/parallel-collect-salary-data.yml`
- Runs up to 20 queries simultaneously
- Use for: Production, large-scale collection
- Time: ~6-8 minutes for 20 queries
- See [parallel-collection.md](parallel-collection.md) for details

## GitHub Actions Setup

### 1. Create S3 Bucket

```bash
aws s3 mb s3://ai-salary-predictor
```

### 2. Create IAM Role

Create an IAM role with OIDC for GitHub Actions.

Trust Policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
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
          "token.actions.githubusercontent.com:sub": "repo:YOUR_USERNAME/linkedin-jobs-api:*"
        }
      }
    }
  ]
}
```

Permissions Policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::ai-salary-predictor",
        "arn:aws:s3:::ai-salary-predictor/data/*",
        "arn:aws:s3:::ai-salary-predictor/consolidated/*",
        "arn:aws:s3:::ai-salary-predictor/metadata/*"
      ]
    }
  ]
}
```

### 3. Add GitHub Secret

Go to Settings → Secrets and variables → Actions and add:
- Name: `AWS_ROLE_ARN`
- Value: `arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_ROLE_NAME`

### 4. Run Workflow

The workflow runs automatically daily at 2 AM UTC, or manually via Actions tab → "Collect Salary Data" → Run workflow

## S3 Bucket Structure

```
ai-salary-predictor/
├── data/
│   └── raw/
│       ├── ai-jobs-2025-11-24T02-00-00Z.jsonl
│       ├── ai-jobs-2025-11-25T02-00-00Z.jsonl
│       └── ...
├── consolidated/
│   └── consolidated-2025-11-24T02-00-00Z.jsonl
└── metadata/
    └── latest-run.json
```

## Using the Data

Python example:

```python
import json
import pandas as pd

# Read JSONL file
data = []
with open('data/ai-jobs-2025-11-24T02-00-00Z.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Filter jobs with salary data
df_with_salary = df[df['salary'] != 'Not specified']

print(f"Total jobs: {len(df)}")
print(f"Jobs with salary: {len(df_with_salary)}")
```

Download from S3:

```bash
aws s3 sync s3://ai-salary-predictor/data/raw/ ./data/
```

## Available Fields

- Target: `salary`
- Categorical: `position`, `company`, `location`, `seniority_level`, `employment_type`, `job_function`, `industries`, `education`
- Numerical: `applicant_count`
- Text: `description`, `experience_years`
- Array: `skills`
- Temporal: `posted_date`, `collected_at`

## Configuration

Adjust collection frequency in `.github/workflows/collect-salary-data.yml`:

```yaml
schedule:
  - cron: '0 2 * * *'  # Daily at 2 AM
  # - cron: '0 */6 * * *'  # Every 6 hours
  # - cron: '0 0 * * 0'  # Weekly on Sunday
```

Change search parameters in `scripts/collect-data.js` → `searchQueries` array

Change AWS region in workflow file → `aws-region: us-east-1`

## Troubleshooting

**Rate limiting:**
- Increase delays between queries in `scripts/collect-data.js`
- Reduce `limit` per query
- Schedule runs less frequently

**No salary data:**
- Some locations/companies don't post salaries
- Try different locations (CA, NY, WA have better salary transparency)
- Set `requireSalary: false` to collect more jobs for other features

**AWS permissions:**
- Verify IAM role trust policy includes your repo
- Check S3 bucket permissions
- Ensure `AWS_ROLE_ARN` secret is set correctly

## Data Cleaning and Deduplication

### Within-Run Deduplication

The collection script automatically deduplicates jobs within a single run using `job_url` as the unique identifier. This prevents the same job from being saved multiple times when it appears in multiple search queries.

### Cross-Run Deduplication

To consolidate historical data and remove duplicates across multiple collection runs:

```bash
npm run deduplicate
```

This creates `./data/deduplicated-jobs.jsonl` with:
- All duplicates removed (keeps most recent/complete record)
- Invalid records filtered out
- Data normalized and cleaned

Custom input/output:

```bash
node scripts/deduplicate-data.js ./data ./data/cleaned-jobs.jsonl
```

### Data Quality Checks

The deduplication script performs:
- **Validation**: Removes records missing required fields (job_url, position, company)
- **Normalization**: Standardizes salary fields (null vs "Not specified")
- **Cleaning**: Trims whitespace, ensures proper data types
- **Best Record Selection**: Keeps records with salary data when duplicates exist

### Recommended Workflow

1. Daily collection runs create timestamped JSONL files
2. Before ML training, run deduplication on all files
3. Use deduplicated file for model training
4. Keep original files for audit trail

## Setup Steps

1. Test locally: `npm run collect`
2. Verify output in `./data/` folder
3. Create AWS S3 bucket and IAM role
4. Add `AWS_ROLE_ARN` secret to GitHub
5. Push changes and trigger workflow
