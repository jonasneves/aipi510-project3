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
        AdzunaJobsCollector,
        LinkedInJobsCollector,
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
            else:
                print("  Warning: No BLS data collected (check API key or series IDs)")
        except Exception as e:
            print(f"  Error collecting BLS data: {e}")

    if args.source == "all" or args.source == "jobs":
        print("\n[3/4] Collecting Job Posting Data...")
        try:
            adzuna = AdzunaJobsCollector(data_dir=args.data_dir)
            results = adzuna.collect()
            if "jobs" in results:
                print(f"  Collected {len(results['jobs'])} job postings")
        except Exception as e:
            print(f"  Error collecting job data: {e}")

    if args.source == "all" or args.source == "linkedin":
        print("\n[4/4] Collecting LinkedIn Job Data from S3...")
        try:
            linkedin = LinkedInJobsCollector(data_dir=args.data_dir)
            df = linkedin.collect(use_consolidated=True)
            if not df.empty:
                stats = linkedin.get_summary_stats(df)
                print(f"  Collected {len(df)} LinkedIn records")
                if stats:
                    print(f"  Median salary: ${stats.get('median', 0):,.0f}")
        except Exception as e:
            print(f"  Error collecting LinkedIn data: {e}")

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

    import yaml
    import mlflow

    from .processing import DataMerger, FeatureEngineer
    from .models import SalaryPredictor

    # Load config for MLFlow settings
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        mlflow_config = config.get("mlflow", {})
    else:
        mlflow_config = {}

    # Set up MLFlow
    tracking_uri = mlflow_config.get("tracking_uri", "file:./mlruns")
    experiment_name = mlflow_config.get("experiment_name", "salary_prediction")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"\nMLFlow tracking URI: {tracking_uri}")
    print(f"MLFlow experiment: {experiment_name}")

    # Load and prepare data
    print("\nLoading data...")
    merger = DataMerger(data_dir=args.data_dir)

    # Match DataMerger's path logic: processed_dir = data_dir.parent / "processed"
    processed_path = Path(args.data_dir).parent / "processed" / "merged_salary_data.parquet"

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

    # Train model with MLFlow tracking
    print("\nTraining XGBoost model...")
    predictor = SalaryPredictor(model_dir=args.model_dir)

    with mlflow.start_run() as run:
        print(f"MLFlow run ID: {run.info.run_id}")

        # Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", len(feature_cols))

        # Log model parameters
        for param, value in predictor.params.items():
            mlflow.log_param(param, value)

        if args.tune:
            print("\nPerforming hyperparameter tuning...")
            predictor.tune_hyperparameters(X_train, y_train)

        predictor.train(X_train, y_train, X_val, y_val)

        # Log training metrics
        if "train" in predictor.training_metrics:
            for metric, value in predictor.training_metrics["train"].items():
                mlflow.log_metric(f"train_{metric}", value)
        if "validation" in predictor.training_metrics:
            for metric, value in predictor.training_metrics["validation"].items():
                mlflow.log_metric(f"val_{metric}", value)

        # Cross-validation
        if args.cv:
            predictor.cross_validate(
                pd.concat([X_train, X_val]) if X_val is not None else X_train,
                pd.concat([y_train, y_val]) if y_val is not None else y_train,
            )

        # Evaluate on test set
        print("\nEvaluating on test set...")
        metrics = predictor.evaluate(X_test, y_test)

        # Log test metrics
        for metric, value in metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        # Feature importance
        print("\nTop 15 Most Important Features:")
        importance_df = predictor.get_feature_importance(top_n=15)
        for _, row in importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Save model
        model_path = predictor.save("salary_model")

        # Log model artifact to MLFlow
        mlflow.log_artifact(str(model_path))

        # Log feature importance as artifact
        importance_path = Path(args.model_dir) / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        print(f"\nMLFlow run completed: {run.info.run_id}")

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
    from .utils.config_loader import ConfigLoader

    # Load encodings from config (same as API)
    features_config = ConfigLoader.get_features()
    STATE_INDEX = features_config["encodings"]["state_index"]
    TIER_INDEX = features_config["encodings"]["tier_index"]

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
    }])

    # Engineer features (don't use prepare_for_modeling - it filters by salary)
    engineer = FeatureEngineer()
    data = engineer.engineer_features(data)

    # Use fixed encodings from config (same as API) - NOT fit=True which creates new encoders
    state = location.upper()
    data["state_clean_encoded"] = STATE_INDEX.get(state, 20)
    company_tier = data["company_tier"].iloc[0] if "company_tier" in data.columns else "unknown"
    data["company_tier_encoded"] = TIER_INDEX.get(company_tier, 6)

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

    # Build feature vector matching model's expected features
    # Deduplicate feature names to avoid reindex errors
    unique_feature_names = list(dict.fromkeys(predictor.feature_names))
    X = pd.DataFrame(index=[0])
    for col in unique_feature_names:
        if col in data.columns:
            val = data[col].iloc[0]
            X[col] = pd.to_numeric(val, errors='coerce') if not isinstance(val, (int, float)) else val
        else:
            X[col] = 0
    X = X.fillna(0)

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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Salary Negotiation Intelligence Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main collect --source all
  python -m src.main train --tune
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
        choices=["all", "h1b", "bls", "jobs", "linkedin"],
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

    args = parser.parse_args()

    if args.command == "collect":
        collect_data(args)
    elif args.command == "merge":
        merge_data(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
