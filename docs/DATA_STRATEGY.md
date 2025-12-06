# Data Strategy

## Sources

| Source | Active | Records | Priority | Reason |
|--------|--------|---------|----------|--------|
| H1B | ✓ | 10,000 | 1 | Government-verified salaries |
| LinkedIn | ✓ | ~1,000 | 1 | Skills, education, experience |
| Adzuna | ✓ | 16,500 | 2 | Market coverage |
| BLS | ✗ | 7 | - | Too few records |

**Target:** 5,000+ LinkedIn records for better model performance.

## Merge Strategy

### Quality Scoring (0-1)
```
Skills:     30%    Experience: 20%    Education:  10%
Seniority:  10%    Employer:   10%    Location:   10%
Priority:   10%
```

### Deduplication
Match on: `job_title + employer_name + location_state + year`

**Conflict Resolution:**
1. Higher quality score wins
2. If tied, higher priority source wins (H1B=LinkedIn=1, Adzuna=2)

### Validation
- Required: job_title, annual_salary
- Salary range: $30K - $1.5M
- Min records per source: 100

## Usage

```bash
# Merge data (enhanced by default)
python -m src.main merge

# Check quality
python -c "import pandas as pd; df = pd.read_parquet('data/processed/merged_salary_data.parquet'); print(df['data_source'].value_counts())"

# View logs
tail -f logs/data_merge_*.log
```

## Key Metrics

| Metric | Target | Warning |
|--------|--------|---------|
| Duplication rate | 10-15% | >30% |
| Avg quality score | >0.5 | <0.4 |
| Invalid records | <10% | >20% |
| LinkedIn quality | >0.7 | <0.5 |

## Config

All settings in `configs/data_sources.yaml`:
- Source enable/disable
- Priority levels
- Merge strategy
- Quality thresholds
