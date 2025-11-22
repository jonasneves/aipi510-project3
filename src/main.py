"""
AI Salary Negotiation Intelligence Tool

Main entry point for data collection, model training, and salary predictions.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def collect_data(args):
    """Collect data from various sources."""
    print("=" * 60)
    print("AI Salary Negotiation Intelligence - Data Collection")
    print("=" * 60)

    from .data_collectors import (
        H1BSalaryCollector,
        BLSDataCollector,
        GoogleTrendsCollector,
        AdzunaJobsCollector,
    )

    if args.source == "all" or args.source == "h1b":
        print("\n[1/4] Collecting H1B Salary Data...")
        try:
            h1b = H1BSalaryCollector(data_dir=args.data_dir)
            df = h1b.collect(years=args.years or [2023, 2024])
            if not df.empty:
                stats = h1b.get_summary_stats(df)
                print(f"  Collected {len(df)} H1B records")
                if stats:
                    print(f"  Median salary: ${stats.get('median', 0):,.0f}")
        except Exception as e:
            print(f"  Error collecting H1B data: {e}")

    if args.source == "all" or args.source == "bls":
        print("\n[2/4] Collecting BLS Wage Data...")
        try:
            bls = BLSDataCollector(data_dir=args.data_dir)
            df = bls.collect(start_year=args.start_year or 2022, end_year=2024)
            if not df.empty:
                print(f"  Collected {len(df)} BLS records")
        except Exception as e:
            print(f"  Error collecting BLS data: {e}")

    if args.source == "all" or args.source == "trends":
        print("\n[3/4] Collecting Google Trends Data...")
        try:
            trends = GoogleTrendsCollector(data_dir=args.data_dir)
            results = trends.collect()
            for key, df in results.items():
                print(f"  {key}: {len(df)} records")
        except Exception as e:
            print(f"  Error collecting trends data: {e}")

    if args.source == "all" or args.source == "jobs":
        print("\n[4/4] Collecting Job Posting Data...")
        try:
            adzuna = AdzunaJobsCollector(data_dir=args.data_dir)
            results = adzuna.collect()
            if "jobs" in results:
                print(f"  Collected {len(results['jobs'])} job postings")
        except Exception as e:
            print(f"  Error collecting job data: {e}")

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)


def merge_data(args):
    """Merge collected data from all sources."""
    print("=" * 60)
    print("AI Salary Negotiation Intelligence - Data Merging")
    print("=" * 60)

    from .processing import DataMerger

    merger = DataMerger(data_dir=args.data_dir)
    merged_df = merger.merge_all_sources()

    if not merged_df.empty:
        stats = merger.get_summary_statistics(merged_df)
        print("\nDataset Summary:")
        print(f"  Total records: {stats['total_records']}")
        print(f"  Unique employers: {stats.get('unique_employers', 'N/A')}")
        print(f"  Salary range: ${stats.get('salary_min', 0):,.0f} - ${stats.get('salary_max', 0):,.0f}")
        print(f"  Median salary: ${stats.get('salary_median', 0):,.0f}")

        if "records_by_source" in stats:
            print("\n  Records by source:")
            for source, count in stats["records_by_source"].items():
                print(f"    {source}: {count}")


def train_model(args):
    """Train the salary prediction model."""
    print("=" * 60)
    print("AI Salary Negotiation Intelligence - Model Training")
    print("=" * 60)

    from .processing import DataMerger, FeatureEngineer
    from .models import SalaryPredictor

    # Load and prepare data
    print("\nLoading data...")
    merger = DataMerger(data_dir=args.data_dir)

    processed_path = Path(args.data_dir) / "processed" / "merged_salary_data.parquet"

    if processed_path.exists():
        df = pd.read_parquet(processed_path)
        print(f"Loaded {len(df)} records from merged data")
    else:
        print("Merged data not found. Running merge first...")
        df = merger.merge_all_sources()

    if df.empty:
        print("No data available for training!")
        print("Run data collection first: python -m src.main collect")
        return

    # Feature engineering
    print("\nEngineering features...")
    engineer = FeatureEngineer()
    df, feature_cols, target_col = engineer.prepare_for_modeling(df)

    if df.empty or not feature_cols:
        print("No valid features generated!")
        return

    # Ensure unique feature columns
    feature_cols = list(dict.fromkeys(feature_cols))

    # Create train/test split
    print("\nCreating train/test split...")
    train_df, val_df, test_df = merger.create_train_test_split(df, test_year=2024)

    def to_numeric_df(data, cols):
        """Convert to numeric DataFrame with unique columns."""
        result = pd.DataFrame(index=data.index)
        for col in cols:
            if col in data.columns:
                s = data[col]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                result[col] = pd.to_numeric(s, errors='coerce').fillna(0)
        return result

    X_train = to_numeric_df(train_df, feature_cols)
    y_train = train_df[target_col].astype(float)

    X_val = to_numeric_df(val_df, feature_cols) if not val_df.empty else None
    y_val = val_df[target_col].astype(float) if not val_df.empty else None

    X_test = to_numeric_df(test_df, feature_cols)
    y_test = test_df[target_col].astype(float)

    # Train model
    print("\nTraining XGBoost model...")
    predictor = SalaryPredictor(model_dir=args.model_dir)

    if args.tune:
        print("\nPerforming hyperparameter tuning...")
        predictor.tune_hyperparameters(X_train, y_train)

    predictor.train(X_train, y_train, X_val, y_val)

    # Cross-validation
    if args.cv:
        predictor.cross_validate(
            pd.concat([X_train, X_val]) if X_val is not None else X_train,
            pd.concat([y_train, y_val]) if y_val is not None else y_train,
        )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = predictor.evaluate(X_test, y_test)

    # Feature importance
    print("\nTop 15 Most Important Features:")
    importance_df = predictor.get_feature_importance(top_n=15)
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Save model
    predictor.save("salary_model")

    print("\n" + "=" * 60)
    print("Model training complete!")
    print("=" * 60)


def predict(args):
    """Make salary predictions."""
    print("=" * 60)
    print("AI Salary Negotiation Intelligence - Salary Prediction")
    print("=" * 60)

    from .models import SalaryPredictor
    from .processing import FeatureEngineer

    # Load model
    predictor = SalaryPredictor(model_dir=args.model_dir)

    try:
        predictor.load()
    except FileNotFoundError:
        print("No trained model found!")
        print("Train a model first: python -m src.main train")
        return

    # Get prediction inputs
    job_title = args.title or input("Job title: ")
    location = args.location or input("Location (state abbreviation, e.g., CA): ")
    company = args.company or input("Company (optional): ") or None
    yoe = args.experience or int(input("Years of experience: ") or "3")
    skills = args.skills.split(",") if args.skills else None

    # Create feature DataFrame
    data = pd.DataFrame([{
        "job_title": job_title,
        "employer_name": company,
        "worksite_state": location.upper(),
        "annual_salary": 0,
    }])

    # Engineer features
    engineer = FeatureEngineer()
    data, feature_cols, _ = engineer.prepare_for_modeling(data, fit=False)

    if not feature_cols:
        print("Error: Could not generate features")
        return

    # Set experience
    if "estimated_yoe" in data.columns:
        data["estimated_yoe"] = yoe

    # Set skills if provided
    if skills:
        skills_lower = [s.lower().strip() for s in skills]
        for col in data.columns:
            if col.startswith("skill_"):
                skill_name = col.replace("skill_", "").replace("_", " ")
                if any(skill_name in s or s in skill_name for s in skills_lower):
                    data[col] = 1

    # Make prediction
    X = data[feature_cols].fillna(0)
    result = predictor.predict_with_range(X)

    # Display results
    print("\n" + "-" * 40)
    print("SALARY PREDICTION RESULTS")
    print("-" * 40)
    print(f"\nJob Title: {job_title}")
    print(f"Location: {location.upper()}")
    if company:
        print(f"Company: {company}")
    print(f"Experience: {yoe} years")
    if skills:
        print(f"Skills: {', '.join(skills)}")

    print(f"\n{'='*40}")
    print(f"PREDICTED SALARY: ${result['predicted_salary'].iloc[0]:,.0f}")
    print(f"{'='*40}")
    print(f"\n90% Confidence Range:")
    print(f"  Low:  ${result['salary_low'].iloc[0]:,.0f}")
    print(f"  High: ${result['salary_high'].iloc[0]:,.0f}")

    # Negotiation tips
    print("\n" + "-" * 40)
    print("NEGOTIATION INSIGHTS")
    print("-" * 40)

    pred_salary = result['predicted_salary'].iloc[0]

    if location.upper() in ["CA", "NY", "WA"]:
        print("- Major tech hub: Salaries tend to be 20-30% above national average")

    if "senior" in job_title.lower() or "lead" in job_title.lower():
        print("- Senior/Lead roles: Total compensation often includes significant equity")

    if skills and any(s in ["nlp", "llm", "pytorch", "deep learning"] for s in [s.lower() for s in skills]):
        print("- High-demand AI skills detected: You have strong negotiating leverage")

    print(f"- Suggested negotiation range: ${pred_salary * 0.95:,.0f} - ${pred_salary * 1.15:,.0f}")


def demo(args):
    """Run a demonstration with synthetic data."""
    print("=" * 60)
    print("AI Salary Negotiation Intelligence - Demo Mode")
    print("=" * 60)

    from .models import SalaryPredictor

    print("\nGenerating synthetic AI/ML salary data...")

    np.random.seed(42)
    n_samples = 2000

    # Generate realistic synthetic features
    X = pd.DataFrame({
        "skill_deep_learning": np.random.binomial(1, 0.35, n_samples),
        "skill_nlp": np.random.binomial(1, 0.25, n_samples),
        "skill_computer_vision": np.random.binomial(1, 0.15, n_samples),
        "skill_mlops": np.random.binomial(1, 0.30, n_samples),
        "skill_cloud_ml": np.random.binomial(1, 0.45, n_samples),
        "skill_traditional_ml": np.random.binomial(1, 0.50, n_samples),
        "is_senior": np.random.binomial(1, 0.35, n_samples),
        "is_lead": np.random.binomial(1, 0.12, n_samples),
        "is_principal": np.random.binomial(1, 0.05, n_samples),
        "is_staff": np.random.binomial(1, 0.08, n_samples),
        "estimated_yoe": np.random.randint(1, 15, n_samples),
        "is_major_tech_hub": np.random.binomial(1, 0.35, n_samples),
        "is_secondary_hub": np.random.binomial(1, 0.25, n_samples),
        "col_multiplier": np.random.uniform(0.90, 1.40, n_samples),
        "role_engineer": np.random.binomial(1, 0.60, n_samples),
        "role_scientist": np.random.binomial(1, 0.30, n_samples),
        "year": np.random.choice([2022, 2023, 2024], n_samples, p=[0.2, 0.3, 0.5]),
    })

    # Feature interactions
    X["skill_count"] = X[[c for c in X.columns if c.startswith("skill_")]].sum(axis=1)
    X["yoe_location_interaction"] = X["estimated_yoe"] * X["col_multiplier"]

    # Generate realistic salaries
    base_salary = 95000
    y = (
        base_salary
        + X["skill_deep_learning"] * 28000
        + X["skill_nlp"] * 35000
        + X["skill_computer_vision"] * 22000
        + X["skill_mlops"] * 18000
        + X["skill_cloud_ml"] * 12000
        + X["is_senior"] * 45000
        + X["is_lead"] * 30000
        + X["is_principal"] * 55000
        + X["is_staff"] * 65000
        + X["estimated_yoe"] * 6000
        + X["is_major_tech_hub"] * 35000
        + X["is_secondary_hub"] * 15000
        + (X["col_multiplier"] - 1) * 100000
        + X["role_scientist"] * 12000
        + (X["year"] - 2022) * 8000  # Yearly increase
        + np.random.normal(0, 18000, n_samples)
    )

    # Clip to reasonable range
    y = y.clip(lower=70000, upper=600000)

    print(f"\nGenerated {n_samples} synthetic salary samples")
    print(f"Salary range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Median salary: ${y.median():,.0f}")

    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]

    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]

    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]

    # Train model
    print("\nTraining XGBoost model...")
    predictor = SalaryPredictor(model_dir=args.model_dir)

    predictor.train(X_train, y_train, X_val, y_val)

    # Cross-validation
    print("\nPerforming cross-validation...")
    predictor.cross_validate(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    # Test evaluation
    print("\nEvaluating on held-out test set...")
    predictor.evaluate(X_test, y_test)

    # Feature importance
    print("\n" + "=" * 40)
    print("TOP 10 SALARY PREDICTORS")
    print("=" * 40)
    importance = predictor.get_feature_importance(top_n=10)
    for i, (_, row) in enumerate(importance.iterrows(), 1):
        bar = "█" * int(row["importance"] * 50)
        print(f"{i:2}. {row['feature']:25} {bar} {row['importance']:.3f}")

    # Save demo model
    predictor.save("demo_salary_model")

    # Example predictions
    print("\n" + "=" * 40)
    print("EXAMPLE SALARY PREDICTIONS")
    print("=" * 40)

    examples = [
        {"name": "Junior ML Engineer, Texas", "yoe": 2, "senior": 0, "hub": 0, "nlp": 0, "dl": 1, "col": 1.0},
        {"name": "Senior Data Scientist, SF", "yoe": 6, "senior": 1, "hub": 1, "nlp": 1, "dl": 1, "col": 1.35},
        {"name": "Staff ML Engineer, Seattle", "yoe": 10, "senior": 0, "hub": 1, "nlp": 0, "dl": 1, "col": 1.20},
        {"name": "Lead NLP Engineer, NYC", "yoe": 8, "senior": 0, "hub": 1, "nlp": 1, "dl": 1, "col": 1.30},
    ]

    for ex in examples:
        sample = pd.DataFrame([{
            "skill_deep_learning": ex["dl"],
            "skill_nlp": ex["nlp"],
            "skill_computer_vision": 0,
            "skill_mlops": 0,
            "skill_cloud_ml": 1,
            "skill_traditional_ml": 1,
            "is_senior": ex["senior"],
            "is_lead": int("Lead" in ex["name"]),
            "is_principal": 0,
            "is_staff": int("Staff" in ex["name"]),
            "estimated_yoe": ex["yoe"],
            "is_major_tech_hub": ex["hub"],
            "is_secondary_hub": 0,
            "col_multiplier": ex["col"],
            "role_engineer": int("Engineer" in ex["name"]),
            "role_scientist": int("Scientist" in ex["name"]),
            "year": 2024,
            "skill_count": ex["dl"] + ex["nlp"] + 2,
            "yoe_location_interaction": ex["yoe"] * ex["col"],
        }])

        pred = predictor.predict_with_range(sample)
        print(f"\n{ex['name']}:")
        print(f"  Predicted: ${pred['predicted_salary'].iloc[0]:,.0f}")
        print(f"  Range: ${pred['salary_low'].iloc[0]:,.0f} - ${pred['salary_high'].iloc[0]:,.0f}")

    print("\n" + "=" * 60)
    print("Demo complete! Model saved for use with predict command.")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Salary Negotiation Intelligence Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with synthetic data
  python -m src.main demo

  # Collect data from all sources
  python -m src.main collect --source all

  # Train model
  python -m src.main train --tune

  # Make a prediction
  python -m src.main predict --title "ML Engineer" --location CA --experience 5
        """,
    )

    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Data directory (default: data/raw)",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Model directory (default: models)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect data from sources")
    collect_parser.add_argument(
        "--source",
        choices=["all", "h1b", "bls", "trends", "jobs"],
        default="all",
        help="Data source to collect (default: all)",
    )
    collect_parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Years to collect H1B data for",
    )
    collect_parser.add_argument(
        "--start-year",
        type=int,
        help="Start year for BLS data",
    )

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge collected data")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train prediction model")
    train_parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning",
    )
    train_parser.add_argument(
        "--cv",
        action="store_true",
        help="Perform cross-validation",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make salary prediction")
    predict_parser.add_argument("--title", help="Job title")
    predict_parser.add_argument("--location", help="US state abbreviation")
    predict_parser.add_argument("--company", help="Company name")
    predict_parser.add_argument("--experience", type=int, help="Years of experience")
    predict_parser.add_argument("--skills", help="Comma-separated skills")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with synthetic data")

    args = parser.parse_args()

    if args.command == "collect":
        collect_data(args)
    elif args.command == "merge":
        merge_data(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "demo":
        demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
