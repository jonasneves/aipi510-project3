# LinkedIn AI Salary Data Scraper

Automated Node.js scraper that collects AI/ML job salary data from LinkedIn and uploads to S3 for use by the ML training pipeline.

## Overview

This scraper runs daily via GitHub Actions (at 1 AM UTC, before the ML pipeline at 2 AM) to collect fresh AI job salary data from LinkedIn. It uses parallel execution with up to 20 concurrent runners to collect data quickly and efficiently.

## Architecture

```
Generate Query Matrix
     |
     v
+----+----+----+----+----+----+----+----+----+----+
| B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10|  (Parallel Runners)
+----+----+----+----+----+----+----+----+----+----+
     |    |    |    |    |    |    |    |    |
     v    v    v    v    v    v    v    v    v
   S3  S3  S3  S3  S3  S3  S3  S3  S3  S3  (Upload Batches)
     |    |    |    |    |    |    |    |    |
     +----+----+----+----+----+----+----+----+
                        |
                        v
                  Consolidate & Deduplicate
                        |
                        v
            S3 consolidated/ (Final Dataset)
                        |
                        v
              ML Pipeline (Consumes Data)
```

## Features

- **Parallel Collection**: Up to 20 concurrent runners for fast data collection
- **Deduplication**: Two-stage deduplication (within batch + cross-batch)
- **Rich Feature Extraction**:
  - Salary ranges
  - Job titles and companies
  - Locations (state-level)
  - Skills (python, tensorflow, aws, etc.)
  - Experience requirements
  - Education requirements
  - Seniority levels
  - Employment types
  - Industries
  - Applicant counts

## Data Flow

1. **Collection** (Parallel): Each batch scrapes 2-5 LinkedIn searches
   - Saves to `batch-{N}-{TIMESTAMP}.jsonl`
   - Uploads to `s3://ai-salary-predictor/linkedin/raw/`

2. **Consolidation** (After all batches complete):
   - Downloads all batch files
   - Deduplicates by job_url
   - Uploads to `s3://ai-salary-predictor/linkedin/consolidated/`

3. **ML Pipeline** (Runs at 2 AM UTC):
   - Downloads consolidated file from S3
   - Standardizes schema
   - Merges with H1B, BLS, Adzuna data
   - Trains salary prediction model

## Output Format

Each job is stored as a JSON line:

```json
{
  "collected_at": "2025-11-25T01:00:00Z",
  "position": "Machine Learning Engineer",
  "company": "Google",
  "location": "San Francisco, CA",
  "posted_date": "2025-11-24",
  "job_url": "https://linkedin.com/jobs/...",
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

## Local Development

### Install Dependencies

```bash
cd linkedin-scraper
npm install
```

### Run Collection Locally

```bash
# Basic collection (uses queries in scripts/collect-data.js)
npm run collect

# Output: ./data/ai-jobs-{TIMESTAMP}.jsonl
```

### Deduplicate Data

```bash
npm run deduplicate

# Output: ./data/deduplicated-jobs.jsonl
```

## GitHub Actions Workflow

The workflow is defined in `.github/workflows/linkedin-scraper.yml`.

### Schedule

- **Automatic**: Daily at 1 AM UTC
- **Manual**: Via GitHub Actions UI

### Manual Trigger

1. Go to Actions → "LinkedIn Salary Data Collection"
2. Click "Run workflow"
3. Select number of parallel runners (5-20)
4. Click "Run workflow"

### Monitor Progress

Check the workflow run to see:
- Parallel batch execution
- Upload progress to S3
- Consolidation and deduplication
- Final summary with statistics

## Configuration

### Search Queries

Edit `.github/workflows/linkedin-scraper.yml` in the `generate-matrix` step to modify search queries:

```yaml
{
  "batch": "1",
  "queries": [
    {
      "keyword": "machine learning engineer",
      "location": "United States",
      "salary": "100000",
      "limit": "100"
    }
  ]
}
```

### Batch Sizing

- **Current**: 10 batches, 2 queries per batch
- **Optimal**: 2-5 queries per batch for best parallelism
- **Max parallel runners**: 20 (GitHub Actions limit)

## S3 Structure

```
s3://ai-salary-predictor/
├── linkedin/
│   ├── raw/
│   │   ├── batch-1-2025-11-25T01-00-00Z.jsonl
│   │   ├── batch-2-2025-11-25T01-00-00Z.jsonl
│   │   └── ...
│   └── consolidated/
│       └── consolidated-2025-11-25T01-00-00Z.jsonl  <- ML pipeline uses this
├── data/
│   ├── raw/
│   │   ├── parquet/          (ML pipeline output)
│   │   └── h1b-source/       (H1B Excel files)
│   └── processed/
│       └── merged_salary_data.parquet
├── models/
│   └── latest/
└── metadata/
    └── latest-run.json
```

## Troubleshooting

### Rate Limiting

If you encounter rate limiting:
- Reduce `limit` per query (e.g., 50 instead of 100)
- Increase delays in `index.js` (CONFIG.MIN_DELAY)
- Reduce number of parallel runners

### No Data Collected

- Check if LinkedIn changed their HTML structure
- Verify AWS credentials are configured
- Check workflow logs for errors

### Batch Failures

- Workflow continues even if some batches fail (`fail-fast: false`)
- Check individual batch logs
- Re-run failed batches manually if needed

## Integration with ML Pipeline

The ML pipeline automatically:
1. Downloads latest consolidated file from S3
2. Parses JSONL format
3. Standardizes schema (maps to H1B format)
4. Merges with other data sources
5. Uses features for model training

See `src/data_collectors/linkedin_collector.py` for integration code.

## Performance

- **Collection Time**: ~6-8 minutes (10 parallel runners)
- **Data Size**: ~10-50 KB per batch JSONL
- **Total Jobs**: ~500-1000 per day
- **With Salary**: ~200-400 jobs with salary data

## Dependencies

- **axios**: HTTP client for LinkedIn requests
- **cheerio**: HTML parsing
- **random-useragent**: Rotate user agents to avoid detection

## License

MIT
