# Quick Start Guide

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure AWS
export AWS_PROFILE=your-aws-profile
```

## Basic Workflow

### 1. Collect Data

```bash
# Collect from public API sources
python -m src.main collect --source all

# Or specific sources
python -m src.main collect --source h1b
python -m src.main collect --source adzuna
```

### 2. Merge Data

```bash
# Merge with quality validation & deduplication
python -m src.main merge

# Check logs
tail -f logs/data_merge_*.log
```

Expected output: ~26,000+ records from H1B and Adzuna.

### 3. Generate EDA Report

```bash
# Create HTML report with visualizations
python -m src.main eda

# View report
open docs/reports/eda_report_latest.html
```

### 4. Train Model

```bash
# Train with hyperparameter tuning
python -m src.main train --tune

# Train with cross-validation
python -m src.main train --cv
```

### 5. Make Predictions

```bash
python -m src.main predict \
  --title "ML Engineer" \
  --location CA \
  --experience 5 \
  --skills "python,tensorflow,aws"
```

## Data Sources

| Source | Status | Priority | Records |
|--------|--------|----------|---------|
| H1B | ✓ | 1 | ~10,000 |
| Adzuna | ✓ | 1 | ~16,500 |
| BLS | ✗ Disabled | - | 7 (too few) |

## File Locations

```
data/processed/merged_salary_data.parquet  # Merged data
logs/data_merge_*.log                       # Merge logs
docs/reports/eda_report_latest.html         # Latest EDA report
models/latest/salary_model_latest.pkl       # Latest model
```

## Configuration

Edit `configs/data_sources.yaml` to:
- Enable/disable sources
- Adjust quality thresholds
- Modify merge strategy
- Change validation rules

## Troubleshooting

**No data after merge?**
- Check `logs/data_merge_*.log`
- Verify source files in `data/raw/`

**Low quality scores?**
- Check merge report: `cat data/processed/merge_report_*.json`
- Review quality thresholds in config

## Documentation

- **Data Strategy**: `docs/DATA_STRATEGY.md`
- **Project Structure**: `docs/PROJECT_STRUCTURE.md`
- **Setup**: `docs/SETUP.md`
