"""
Data Merger Module

Combines data from multiple sources (H1B, BLS, job postings, trends)
into a unified dataset for salary prediction modeling.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


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

    def load_h1b_data(self) -> pd.DataFrame:
        """Load H1B salary data."""
        filepath = self.data_dir / "h1b_ai_salaries.parquet"

        if not filepath.exists():
            print(f"H1B data not found at {filepath}")
            return pd.DataFrame()

        df = pd.read_parquet(filepath)
        df["data_source"] = "h1b"

        return df

    def load_bls_data(self) -> pd.DataFrame:
        """Load BLS wage statistics."""
        filepath = self.data_dir / "bls_wage_data.parquet"

        if not filepath.exists():
            print(f"BLS data not found at {filepath}")
            return pd.DataFrame()

        df = pd.read_parquet(filepath)
        df["data_source"] = "bls"

        return df

    def load_job_postings(self) -> pd.DataFrame:
        """Load job posting data from Adzuna."""
        filepath = self.data_dir / "adzuna_jobs.parquet"

        if not filepath.exists():
            print(f"Job posting data not found at {filepath}")
            return pd.DataFrame()

        df = pd.read_parquet(filepath)
        df["data_source"] = "adzuna"

        return df

    def load_trends_data(self) -> pd.DataFrame:
        """Load Google Trends data."""
        filepath = self.data_dir / "google_trends_regional.parquet"

        if not filepath.exists():
            print(f"Trends data not found at {filepath}")
            return pd.DataFrame()

        df = pd.read_parquet(filepath)

        return df

    def standardize_h1b(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize H1B data to common schema."""
        if df.empty:
            return df

        standardized = pd.DataFrame({
            "job_title": df.get("job_title", df.get("soc_title")),
            "employer_name": df.get("employer_name"),
            "location_city": df.get("worksite_city", df.get("employer_city")),
            "location_state": df.get("worksite_state", df.get("employer_state")),
            "annual_salary": df.get("annual_salary", df.get("wage_from")),
            "year": df.get("fiscal_year", df.get("year")),
            "soc_code": df.get("soc_code"),
            "data_source": "h1b",
        })

        return standardized

    def standardize_bls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize BLS data to common schema."""
        if df.empty:
            return df

        standardized = pd.DataFrame({
            "job_title": df.get("occupation"),
            "employer_name": None,  # BLS doesn't have employer data
            "location_city": df.get("area_name"),
            "location_state": df.get("area_code").apply(
                self._extract_state_from_metro
            ) if "area_code" in df.columns else None,
            "annual_salary": df.get("mean_annual_wage"),
            "year": df.get("year"),
            "soc_code": df.get("soc_code"),
            "data_source": "bls",
        })

        return standardized

    def standardize_job_postings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize job posting data to common schema."""
        if df.empty:
            return df

        standardized = pd.DataFrame({
            "job_title": df.get("title"),
            "employer_name": df.get("company"),
            "location_city": df.get("location"),
            "location_state": df.get("state"),
            "annual_salary": df.get("salary_avg"),
            "year": df.get("year"),
            "soc_code": None,  # Job postings don't have SOC codes
            "data_source": "job_posting",
        })

        return standardized

    def _extract_state_from_metro(self, area_code: str) -> str:
        """Extract state from metro area code."""
        # BLS metro areas often have state in the area name
        metro_to_state = {
            "41860": "CA",  # San Francisco
            "35620": "NY",  # New York
            "42660": "WA",  # Seattle
            "14460": "MA",  # Boston
            "47900": "DC",  # Washington DC
            "26420": "TX",  # Houston
            "19100": "TX",  # Dallas
            "31080": "CA",  # Los Angeles
            "12060": "GA",  # Atlanta
            "16980": "IL",  # Chicago
            "41940": "CA",  # San Jose
            "12420": "TX",  # Austin
            "19740": "CO",  # Denver
            "0000000": "US",  # National
        }

        return metro_to_state.get(str(area_code), "Unknown")

    def add_trend_features(
        self,
        df: pd.DataFrame,
        trends_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add Google Trends features to salary data.

        Args:
            df: Main salary DataFrame
            trends_df: Google Trends data

        Returns:
            DataFrame with trend features added
        """
        if trends_df.empty or df.empty:
            return df

        # Regional trends don't have time series - just return original df
        if "interest" not in trends_df.columns:
            return df

        return df

    def merge_all_sources(
        self,
        include_h1b: bool = True,
        include_bls: bool = True,
        include_jobs: bool = True,
        include_trends: bool = True,
    ) -> pd.DataFrame:
        """
        Merge all data sources into unified dataset.

        Args:
            include_h1b: Include H1B data
            include_bls: Include BLS data
            include_jobs: Include job posting data
            include_trends: Include Google Trends features

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

        if not all_data:
            print("No data sources available!")
            return pd.DataFrame()

        # Combine all data
        print("\nMerging data sources...")
        merged_df = pd.concat(all_data, ignore_index=True)

        # Add trend features
        if include_trends:
            print("Adding trend features...")
            trends_df = self.load_trends_data()
            if not trends_df.empty:
                merged_df = self.add_trend_features(merged_df, trends_df)

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

        # Remove rows without salary
        if "annual_salary" in df.columns:
            df = df[df["annual_salary"].notna()]

            # Remove obviously incorrect salaries
            df = df[(df["annual_salary"] >= 30000) & (df["annual_salary"] <= 1500000)]

        # Standardize state codes
        if "location_state" in df.columns:
            df["location_state"] = df["location_state"].str.upper().str.strip()

            # Handle common variations
            state_mapping = {
                "CALIFORNIA": "CA",
                "NEW YORK": "NY",
                "WASHINGTON": "WA",
                "TEXAS": "TX",
                "MASSACHUSETTS": "MA",
            }

            df["location_state"] = df["location_state"].replace(state_mapping)

        # Fill missing years
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["year"] = df["year"].fillna(2024).astype(int)

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
