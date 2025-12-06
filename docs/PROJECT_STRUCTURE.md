# Project Structure

## Directory Layout

```
aipi510-project3/
├── configs/              # YAML configurations
│   └── data_sources.yaml # Source definitions & merge config
├── data/
│   ├── raw/             # Source parquet files
│   └── processed/       # Merged data & reports
├── src/
│   ├── data_collectors/ # Data collection modules
│   ├── processing/      # Merge & feature engineering
│   │   ├── data_merger_enhanced.py  # Use this
│   │   └── data_merger.py           # Legacy
│   ├── models/          # ML models
│   └── utils/           # Config loader, etc.
├── logs/                # Merge & training logs
├── models/              # Trained model files
└── docs/                # Documentation
```

## File Naming

```bash
# Data
data/raw/source_name.parquet
data/processed/merged_salary_data.parquet
data/processed/merge_report_YYYYMMDD_HHMMSS.json

# Models
models/salary_model_YYYYMMDD_HHMMSS.pkl
models/latest/salary_model_latest.pkl

# Logs
logs/data_merge_YYYYMMDD_HHMMSS.log
```

## What to Commit

**YES:**
- Source code (src/, api/, configs/)
- Docs, tests, requirements
- Notebooks (clear outputs first)

**NO:**
- Data files (data/)
- Models (models/)
- Logs (logs/)
- .env, venv/, __pycache__/

## Best Practices

1. **Config over code** - Use YAML configs, not hardcoded values
2. **Logging over printing** - Use logger, not print()
3. **One module, one job** - Separate concerns
4. **Test before commit** - Ensure merge works
