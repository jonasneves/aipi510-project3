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
    print("--- Data Collection ---")

    from .data_collectors import (
        H1BSalaryCollector,
        AdzunaJobsCollector,
        LinkedInJobsCollector,
    )

    # Handle source argument (list of sources or None)
    sources = args.source if args.source else ["all"]
    should_collect = lambda source: "all" in sources or source in sources

    if should_collect("h1b"):
        print("\n[1/3] Collecting H1B Salary Data...")
        try:
            h1b = H1BSalaryCollector(data_dir=args.data_dir)
            df = h1b.collect(
                years=args.years or [2023, 2024],
                n_samples=args.h1b_samples,
            )
            if not df.empty:
                stats = h1b.get_summary_stats(df)
                print(f"  Collected {len(df)} H1B records")
                if stats:
                    print(f"  Median salary: ${stats.get('median', 0):,.0f}")
        except Exception as e:
            print(f"  Error collecting H1B data: {e}")

    if should_collect("jobs"):
        print("\n[2/3] Collecting Job Posting Data...")
        try:
            adzuna = AdzunaJobsCollector(data_dir=args.data_dir)
            results = adzuna.collect(
                max_queries=args.adzuna_queries,
                max_locations=args.adzuna_locations,
                max_pages=args.adzuna_pages,
            )
            if "jobs" in results:
                print(f"  Collected {len(results['jobs'])} job postings")
        except Exception as e:
            print(f"  Error collecting job data: {e}")

    if should_collect("linkedin"):
        print("\n[3/3] Collecting LinkedIn Job Data from S3...")
        try:
            linkedin = LinkedInJobsCollector(data_dir=args.data_dir)
            use_all = getattr(args, 'linkedin_all_history', False)
            df = linkedin.collect(
                use_consolidated=True,
                use_all_consolidated=use_all
            )
            if not df.empty:
                stats = linkedin.get_summary_stats(df)
                print(f"  Collected {len(df)} LinkedIn records")
                if stats:
                    print(f"  Median salary: ${stats.get('median', 0):,.0f}")
        except Exception as e:
            print(f"  Error collecting LinkedIn data: {e}")

    print("\nData collection complete!")


def merge_data(args):
    """Merge collected data from all sources."""
    print("--- Data Merging ---")

    from .processing import DataMerger

    # Show profile info if specified
    if hasattr(args, 'profile') and args.profile:
        print(f"\nUsing training profile: {args.profile}")

    merger = DataMerger(
        data_dir="data",
        training_profile=getattr(args, 'profile', None)
    )  # Uses new hierarchical structure
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


def generate_eda(args):
    """Generate EDA report."""
    print("--- EDA Report ---")

    from .analysis.eda_report import EDAReportGenerator

    generator = EDAReportGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        keep_latest_only=args.keep_latest_only
    )
    generator.run()


def train_model(args):
    """Train the salary prediction model."""
    # Handle --list-profiles first
    if hasattr(args, 'list_profiles') and args.list_profiles:
        from .utils.config_loader import ConfigLoader
        config = ConfigLoader.get_data_sources()
        profiles = config.get("training_profiles", {})

        print("--- Available Training Profiles ---\n")

        if not profiles:
            print("No training profiles defined in configs/data_sources.yaml")
            return

        for name, profile in profiles.items():
            print(f"{name}:")
            print(f"  Description: {profile.get('description', 'N/A')}")
            print(f"  Data sources:")
            sources = profile.get('sources', {})
            for source, enabled in sources.items():
                status = "✓" if enabled else "✗"
                print(f"    {status} {source}")
            print()

        print("=" * 60)
        print(f"\nUsage: python -m src.main train --profile <profile_name>")
        return

    print("--- Model Training ---")

    import yaml
    import mlflow

    from .processing import DataMerger, FeatureEngineer
    from .models import SalaryPredictor
    from .utils.config_loader import ConfigLoader

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

    # Show profile info if specified
    profile = getattr(args, 'profile', None)
    if profile:
        print(f"\nUsing training profile: {profile}")

    merger = DataMerger(
        data_dir="data",
        training_profile=profile
    )  # Uses new hierarchical structure

    # Look for merged data in new location
    merged_path = Path("data/merged/merged_salary_data.parquet")

    if merged_path.exists():
        df = pd.read_parquet(merged_path)
        print(f"Loaded {len(df)} records from merged data")

        # Apply profile filtering to existing merged data if specified
        if profile and profile != 'all':
            print(f"  Filtering data for profile: {profile}")
            profile_config = ConfigLoader.get_data_sources().get("training_profiles", {}).get(profile, {})
            if profile_config:
                sources_config = profile_config.get("sources", {})
                # Filter to only include enabled sources from profile
                enabled_sources = [src for src, enabled in sources_config.items() if enabled]
                if 'data_source' in df.columns and enabled_sources:
                    df = df[df['data_source'].isin(enabled_sources)].copy()
                    print(f"  After profile filter: {len(df)} records")
                    if len(df) == 0:
                        print(f"  ERROR: No data for sources: {enabled_sources}")
                        print(f"  Available sources in merged data: {df['data_source'].unique().tolist()}")
                        return
    else:
        print("Merged data not found. Running merge first...")
        df = merger.merge_all_sources()

    if df.empty:
        print("No data available for training!")
        print("Run data collection first: python -m src.main collect")
        return

    # Filter to recent data only (configurable lookback period)
    from datetime import datetime
    current_year = datetime.now().year
    training_config = config.get("training", {})
    lookback_years = training_config.get("lookback_years", 2)
    cutoff_year = current_year - lookback_years + 1

    print(f"\nFiltering to recent data (last {lookback_years} years: {cutoff_year}-{current_year})")
    df_before = len(df)
    df = df[df['year'] >= cutoff_year].copy()
    print(f"  Kept {len(df)} / {df_before} records ({len(df)/df_before*100:.1f}%)")

    if df.empty:
        print("No recent data available after filtering!")
        return

    # Fix column name mismatch: location_state -> worksite_state
    if 'location_state' in df.columns and 'worksite_state' not in df.columns:
        print("  Renaming location_state to worksite_state for feature engineering")
        df = df.rename(columns={'location_state': 'worksite_state'})

    # Feature engineering
    print("\nEngineering features...")
    engineer = FeatureEngineer()
    df, feature_cols, target_col = engineer.prepare_for_modeling(df)

    if df.empty or not feature_cols:
        print("No valid features generated!")
        return

    # Ensure unique feature columns
    feature_cols = list(dict.fromkeys(feature_cols))

    # Create train/test split using hybrid approach (random + stratified)
    print("\nCreating train/test split...")
    from sklearn.model_selection import train_test_split

    test_size = config.get("data", {}).get("test_size", 0.15)
    val_size = config.get("data", {}).get("validation_size", 0.15)
    random_state = config.get("data", {}).get("random_state", 42)
    stratify_by_year = training_config.get("stratify_by_year", True)

    # Stratify by year if enabled and year column exists
    stratify_col = df['year'] if (stratify_by_year and 'year' in df.columns) else None

    # Split: test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    # Split: validation set from remaining
    # Adjust val_size to be relative to train_val_df
    val_size_adjusted = val_size / (1 - test_size)
    stratify_col_val = train_val_df['year'] if (stratify_by_year and 'year' in train_val_df.columns) else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_col_val
    )

    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

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

    print("\nModel training complete!")


def predict(args):
    """Make salary predictions."""
    print("--- Salary Prediction ---")

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
        action="append",
        choices=["all", "h1b", "jobs", "linkedin"],
        help="Data source to collect (can be specified multiple times, default: all)",
    )
    collect_parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Years to collect H1B data for",
    )
    collect_parser.add_argument(
        "--h1b-samples",
        type=int,
        default=5000,
        help="Number of H1B samples to generate if real data unavailable (default: 5000)",
    )
    collect_parser.add_argument(
        "--adzuna-queries",
        type=int,
        default=None,
        help="Number of Adzuna search queries to use (default: all 14)",
    )
    collect_parser.add_argument(
        "--adzuna-locations",
        type=int,
        default=None,
        help="Number of Adzuna locations to search (default: all 11)",
    )
    collect_parser.add_argument(
        "--adzuna-pages",
        type=int,
        default=10,
        help="Maximum pages per Adzuna query (default: 10)",
    )
    collect_parser.add_argument(
        "--linkedin-all-history",
        action="store_true",
        help="Fetch ALL historical LinkedIn consolidated files (189%% more data, recommended)",
    )

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge collected data")
    merge_parser.add_argument(
        "--profile",
        help="Training profile to use (filters which sources to include)",
    )

    # EDA command
    eda_parser = subparsers.add_parser("eda", help="Generate EDA report")
    eda_parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Data directory (default: data/processed)",
    )
    eda_parser.add_argument(
        "--output-dir",
        default="docs/reports",
        help="Output directory (default: docs/reports)",
    )
    eda_parser.add_argument(
        "--keep-latest-only",
        action="store_true",
        help="Remove old reports, keep only latest",
    )

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
    train_parser.add_argument(
        "--profile",
        help="Training profile to use (e.g., h1b_only, linkedin_only, h1b_linkedin, all)",
    )
    train_parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available training profiles and exit",
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
    elif args.command == "eda":
        generate_eda(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
