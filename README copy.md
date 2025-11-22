# AI Salary Negotiation Intelligence

XGBoost-based salary prediction for AI/ML roles using H1B visa data, BLS statistics, and job postings.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` with API keys:
```
BLS_API_KEY=your_key
ADZUNA_APP_ID=your_id
ADZUNA_API_KEY=your_key
```

## Usage

```bash
# Collect data
python -m src.main collect --source all

# Merge and train
python -m src.main merge
python -m src.main train

# Predict
python -m src.main predict --title "ML Engineer" --location CA --experience 5
```

## Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| H1B Visa | ~14,000 | Exact salaries from DOL disclosure data |
| Adzuna | ~2,700 | Job postings with salary ranges |
| BLS | 7 | National wage benchmarks by occupation |
| Google Trends | ~100 | Regional interest data |

## Project Structure

```
src/
├── data_collectors/
│   ├── h1b_collector.py     # H1B visa data
│   ├── bls_collector.py     # BLS wage benchmarks
│   ├── trends_collector.py  # Google Trends
│   └── adzuna_collector.py  # Job postings
├── processing/
│   ├── feature_engineering.py
│   └── data_merger.py
├── models/
│   └── salary_predictor.py
└── main.py
```

## Features

Extracted from job data:
- **Skills**: Deep learning, NLP, MLOps, cloud ML, computer vision
- **Experience**: Junior, mid, senior, lead, staff, principal, director
- **Location**: Cost-of-living multiplier, tech hub classification
- **Company**: FAANG, tier-1, finance, other

## CLI Reference

```bash
# Collection
python -m src.main collect --source [all|h1b|bls|trends|jobs]

# Processing
python -m src.main merge

# Training
python -m src.main train [--tune] [--cv]

# Prediction
python -m src.main predict --title "..." --location XX [--experience N] [--skills "a,b,c"]

# Demo (synthetic data)
python -m src.main demo
```

## Data Flow

1. **Collect**: Download H1B files, fetch BLS/Adzuna APIs, scrape Trends
2. **Merge**: Standardize schemas, combine sources, save to parquet
3. **Train**: Feature engineering, XGBoost regression, model persistence
4. **Predict**: Load model, engineer features, return salary estimate

## License

MIT
