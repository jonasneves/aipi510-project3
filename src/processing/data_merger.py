"""
Data Merger Module

Combines data from multiple sources (H1B, BLS, job postings)
into a unified dataset for salary prediction modeling.
"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Built data integration pipeline combining H1B, BLS, Adzuna, and LinkedIn sources with
## schema alignment, deduplication, validation, and temporal train/test splitting.

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Import config loader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import ConfigLoader


class DataMerger:
    """Merge and align data from multiple sources."""

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data merger.

        Args:
            data_dir: Directory containing raw parquet files
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir.parent / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self._config = ConfigLoader.get_data_sources()
        self._sources = self._config["sources"]
        self._metro_to_state = self._config["metro_to_state"]
        self._validation = self._config["validation"]
        self._state_mapping = self._config["state_mapping"]

    def _load_source(self, source_name: str) -> pd.DataFrame:
        """
        Generic data loader using registry pattern.

        Args:
            source_name: Name of data source (h1b, bls, adzuna)

        Returns:
            DataFrame with loaded data
        """
        if source_name not in self._sources:
            print(f"Unknown data source: {source_name}")
            return pd.DataFrame()

        config = self._sources[source_name]
        filepath = self.data_dir / config["filename"]

        if not filepath.exists():
            print(f"{source_name.upper()} data not found at {filepath}")
            return pd.DataFrame()

        df = pd.read_parquet(filepath)
        df["data_source"] = source_name

        return df

    def _standardize_source(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Generic standardization using registry pattern.

        Args:
            df: DataFrame to standardize
            source_name: Name of data source

        Returns:
            Standardized DataFrame
        """
        if df.empty or source_name not in self._sources:
            return df

        config = self._sources[source_name]
        columns = config.get("columns", {})

        if not columns:
            return df

        standardized = {}

        for target_col, source_cols in columns.items():
            if source_cols is None:
                standardized[target_col] = None
            elif isinstance(source_cols, list):
                # Try multiple possible source columns
                value = None
                for col in source_cols:
                    if col in df.columns:
                        value = df[col]
                        break
                standardized[target_col] = value
            else:
                # Single source column
                if source_cols in df.columns:
                    # Special handling for BLS area_code
                    if source_name == "bls" and target_col == "location_state":
                        standardized[target_col] = df[source_cols].apply(
                            self._extract_state_from_metro
                        )
                    else:
                        standardized[target_col] = df[source_cols]
                else:
                    standardized[target_col] = None

        standardized["data_source"] = source_name if source_name != "adzuna" else "job_posting"

        return pd.DataFrame(standardized)

    def load_h1b_data(self) -> pd.DataFrame:
        """Load H1B salary data."""
        return self._load_source("h1b")

    def load_bls_data(self) -> pd.DataFrame:
        """Load BLS wage statistics."""
        return self._load_source("bls")

    def load_job_postings(self) -> pd.DataFrame:
        """Load job posting data from Adzuna."""
        return self._load_source("adzuna")

    def load_linkedin_data(self) -> pd.DataFrame:
        """Load LinkedIn job data."""
        return self._load_source("linkedin")

    def standardize_h1b(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize H1B data to common schema."""
        return self._standardize_source(df, "h1b")

    def standardize_bls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize BLS data to common schema."""
        return self._standardize_source(df, "bls")

    def standardize_job_postings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize job posting data to common schema."""
        return self._standardize_source(df, "adzuna")

    def standardize_linkedin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize LinkedIn data to common schema."""
        return self._standardize_source(df, "linkedin")

    def _extract_state_from_metro(self, area_code: str) -> str:
        """Extract state from metro area code."""
        return self._metro_to_state.get(str(area_code), "Unknown")

    def merge_all_sources(
        self,
        include_h1b: bool = True,
        include_bls: bool = True,
        include_jobs: bool = True,
        include_linkedin: bool = True,
    ) -> pd.DataFrame:
        """
        Merge all data sources into unified dataset.

        Args:
            include_h1b: Include H1B data
            include_bls: Include BLS data
            include_jobs: Include job posting data
            include_linkedin: Include LinkedIn data

        Returns:
            Merged DataFrame
        """
        all_data = []

        if include_h1b:
            print("Loading H1B data...")
            h1b_df = self.load_h1b_data()
            if not h1b_df.empty:
                h1b_std = self.standardize_h1b(h1b_df)
                all_data.append(h1b_std)
                print(f"  Loaded {len(h1b_std)} H1B records")

        if include_bls:
            print("Loading BLS data...")
            bls_df = self.load_bls_data()
            if not bls_df.empty:
                bls_std = self.standardize_bls(bls_df)
                all_data.append(bls_std)
                print(f"  Loaded {len(bls_std)} BLS records")

        if include_jobs:
            print("Loading job posting data...")
            jobs_df = self.load_job_postings()
            if not jobs_df.empty:
                jobs_std = self.standardize_job_postings(jobs_df)
                all_data.append(jobs_std)
                print(f"  Loaded {len(jobs_std)} job posting records")

        if include_linkedin:
            print("Loading LinkedIn data...")
            linkedin_df = self.load_linkedin_data()
            if not linkedin_df.empty:
                linkedin_std = self.standardize_linkedin(linkedin_df)
                all_data.append(linkedin_std)
                print(f"  Loaded {len(linkedin_std)} LinkedIn records")

        if not all_data:
            print("No data sources available!")
            return pd.DataFrame()

        # Combine all data
        print("\nMerging data sources...")
        merged_df = pd.concat(all_data, ignore_index=True)

        # Clean merged data
        merged_df = self._clean_merged_data(merged_df)

        # Save merged data
        output_path = self.processed_dir / "merged_salary_data.parquet"
        merged_df.to_parquet(output_path, index=False)
        print(f"\nSaved merged data to {output_path}")
        print(f"Total records: {len(merged_df)}")

        return merged_df

    def _clean_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate merged data."""
        df = df.copy()

        salary_min = self._validation["salary"]["min"]
        salary_max = self._validation["salary"]["max"]
        default_year = self._validation["year"]["default"]

        if "annual_salary" in df.columns:
            df = df[df["annual_salary"].notna()]
            df = df[(df["annual_salary"] >= salary_min) & (df["annual_salary"] <= salary_max)]

        if "location_state" in df.columns:
            df["location_state"] = df["location_state"].str.upper().str.strip()
            df["location_state"] = df["location_state"].replace(self._state_mapping)

        # Fill missing years
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["year"] = df["year"].fillna(default_year).astype(int)

        # Remove duplicates
        subset_cols = ["job_title", "employer_name", "annual_salary", "year"]
        existing_cols = [col for col in subset_cols if col in df.columns]
        if existing_cols:
            df = df.drop_duplicates(subset=existing_cols, keep="first")

        return df

    def get_summary_statistics(self, df: pd.DataFrame) -> dict:
        """
        Calculate summary statistics for merged data.

        Args:
            df: Merged DataFrame

        Returns:
            Dictionary with summary statistics
        """
        stats = {
            "total_records": len(df),
            "unique_employers": df["employer_name"].nunique() if "employer_name" in df.columns else 0,
            "unique_titles": df["job_title"].nunique() if "job_title" in df.columns else 0,
        }

        if "annual_salary" in df.columns:
            salary_stats = df["annual_salary"].describe()
            stats["salary_mean"] = salary_stats["mean"]
            stats["salary_median"] = salary_stats["50%"]
            stats["salary_std"] = salary_stats["std"]
            stats["salary_min"] = salary_stats["min"]
            stats["salary_max"] = salary_stats["max"]

        if "data_source" in df.columns:
            stats["records_by_source"] = df["data_source"].value_counts().to_dict()

        if "location_state" in df.columns:
            stats["records_by_state"] = df["location_state"].value_counts().head(10).to_dict()

        if "year" in df.columns:
            stats["records_by_year"] = df["year"].value_counts().sort_index().to_dict()

        return stats

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_year: int = 2024,
        val_ratio: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/validation/test split.

        Args:
            df: Merged DataFrame
            test_year: Year to use as test set
            val_ratio: Ratio of training data for validation

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if "year" not in df.columns:
            # Fall back to random split
            from sklearn.model_selection import train_test_split
            train_val, test = train_test_split(df, test_size=0.2, random_state=42)
            train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)
            return train, val, test

        # Time-based split
        test_df = df[df["year"] >= test_year].copy()
        train_val_df = df[df["year"] < test_year].copy()

        # Split remaining into train/val
        val_size = int(len(train_val_df) * val_ratio)
        train_val_df = train_val_df.sample(frac=1, random_state=42)  # Shuffle

        val_df = train_val_df.iloc[:val_size]
        train_df = train_val_df.iloc[val_size:]

        print(f"Train set: {len(train_df)} records (years < {test_year})")
        print(f"Validation set: {len(val_df)} records")
        print(f"Test set: {len(test_df)} records (year >= {test_year})")

        return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    merger = DataMerger()

    # Merge all available data sources
    merged_df = merger.merge_all_sources()

    if not merged_df.empty:
        # Get summary statistics
        stats = merger.get_summary_statistics(merged_df)
        print("\nSummary Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Create train/test split
        train, val, test = merger.create_train_test_split(merged_df)
