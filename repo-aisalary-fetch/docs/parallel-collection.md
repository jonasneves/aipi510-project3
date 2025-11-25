# Parallel Data Collection Strategy

## Overview

The parallel collection workflow leverages GitHub Actions matrix strategy to run up to 20 concurrent jobs, dramatically reducing collection time from hours to minutes.

## Architecture

```
Generate Matrix
     |
     v
+----+----+----+----+----+----+----+----+----+----+
| B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10|  (Parallel Runners)
+----+----+----+----+----+----+----+----+----+----+
     |    |    |    |    |    |    |    |    |
     v    v    v    v    v    v    v    v    v
   S3  S3  S3  S3  S3  S3  S3  S3  S3  S3  (Upload)
     |    |    |    |    |    |    |    |    |
     +----+----+----+----+----+----+----+----+
                        |
                        v
                  Consolidate & Deduplicate
                        |
                        v
                    S3 (Final)
```

## How It Works

### Stage 1: Matrix Generation
- Defines search query batches
- Each batch contains 2-5 related queries
- Outputs: batch assignments + timestamp

### Stage 2: Parallel Collection (up to 20 runners)
Each runner:
1. Processes assigned batch of queries
2. Deduplicates within batch
3. Saves to `batch-{id}-{timestamp}.jsonl`
4. Uploads to S3 immediately

### Stage 3: Consolidation
1. Downloads all batch files from S3
2. Runs deduplication across all batches
3. Creates consolidated file
4. Uploads final deduplicated dataset

## Performance Comparison

| Approach | Queries | Time | Cost |
|----------|---------|------|------|
| Sequential (old) | 20 queries × 50 jobs | ~45-60 min | $0.008/min |
| Parallel (10 runners) | 20 queries × 100 jobs | ~6-8 min | $0.08/min |
| Parallel (20 runners) | 40 queries × 100 jobs | ~4-6 min | $0.12/min |

**Recommended:** 10 parallel runners (85% time savings, reasonable cost)

## Configuration

### Adjust Number of Runners

Edit `.github/workflows/parallel-collect-salary-data.yml`:

```yaml
strategy:
  max-parallel: 20  # 1-20 runners
```

### Add More Search Queries

Edit the matrix in the `generate-matrix` job:

```json
{
  "batch": "11",
  "queries": [
    {
      "keyword": "your search term",
      "location": "location",
      "salary": "100000",
      "limit": "100"
    }
  ]
}
```

### Batch Sizing Strategy

**Optimal batch size:** 2-5 queries per batch

- **Too small** (1 query/batch): Wastes runners, overhead dominates
- **Too large** (10+ queries/batch): Loses parallelism benefits
- **Ideal** (2-5 queries): Balances parallelism and efficiency

### Query Distribution Guidelines

1. **Group related searches** together
   - "ML engineer" + "ML engineer California" = same batch
   - Increases deduplication within batch

2. **Balance by expected results**
   - Popular terms ("data scientist") in own batch
   - Niche terms grouped together

3. **Geographic distribution**
   - Mix national and state-specific searches
   - High-salary regions (CA, NY, WA) get dedicated batches

## Monitoring

### Check Parallel Execution

GitHub Actions → Workflow run → View jobs graph

### S3 Bucket After Collection

```
s3://ai-salary-predictor/
├── data/
│   └── raw/
│       ├── batch-1-2025-11-24T02-00-00Z.jsonl
│       ├── batch-2-2025-11-24T02-00-00Z.jsonl
│       └── ...
├── consolidated/
│   └── consolidated-2025-11-24T02-00-00Z.jsonl
└── metadata/
    └── latest-run.json
```

### Troubleshooting

**Some batches fail:**
- `fail-fast: false` ensures other batches continue
- Check failed job logs
- Re-run specific batch manually

**Rate limiting:**
- Reduce `limit` per query
- Increase delays between queries in batch
- Reduce number of parallel runners

**Duplicate data:**
- Consolidation step handles all duplicates
- Use `consolidated-*.jsonl` for ML training

## Cost Optimization

### GitHub Actions Minutes

Free tier: 2,000 minutes/month

Parallel collection cost:
- 10 runners × 6 min = 60 minutes/run
- Daily runs: 60 × 30 = 1,800 minutes/month
- **Fits in free tier**

### AWS S3 Costs

- Storage: ~10 MB/day × 30 = 300 MB/month (~$0.01)
- Requests: ~100 PUT requests/month (~$0.001)
- **Total: ~$0.01/month**

## Advanced: Dynamic Batch Generation

For very large-scale collection, generate batches dynamically:

```javascript
// Example: Auto-split 100+ queries into optimal batches
const queries = [...]; // Your full query list
const batchSize = 3;
const batches = [];

for (let i = 0; i < queries.length; i += batchSize) {
  batches.push({
    batch: String(i / batchSize + 1),
    queries: queries.slice(i, i + batchSize)
  });
}
```

## When to Use Each Workflow

**Sequential** (`.github/workflows/collect-salary-data.yml`):
- Testing new queries
- Small collection runs (<10 queries)
- Low priority background collection

**Parallel** (`.github/workflows/parallel-collect-salary-data.yml`):
- Production daily runs
- Large-scale collection (20+ queries)
- Time-sensitive data collection
