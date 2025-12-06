"""
Data Merger Module

Combines multiple data sources with quality validation, smart deduplication,
and source priority-based conflict resolution.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging

import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# Import config loader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import ConfigLoader


class DataMerger:
    """Data merger with quality validation and audit trail."""

    def __init__(self, data_dir: str = "data", training_profile: Optional[str] = None):
        """
        Initialize data merger.

        Args:
            data_dir: Root data directory (contains h1b/, linkedin/, adzuna/ subdirectories)
            training_profile: Optional training profile name to apply (e.g., 'h1b_only', 'linkedin_only')
        """
        self.data_dir = Path(data_dir)
        self.merged_dir = self.data_dir / "merged"
        self.merged_dir.mkdir(parents=True, exist_ok=True)

        # Source-specific processed directories
        self.h1b_dir = self.data_dir / "h1b" / "processed"
        self.linkedin_dir = self.data_dir / "linkedin" / "processed"
        self.adzuna_dir = self.data_dir / "adzuna" / "processed"

        # Load configuration
        self._config = ConfigLoader.get_data_sources()
        self._sources = self._config.get("sources", {})
        self._validation = self._config.get("validation", {})
        self._merge_strategy = self._config.get("merge_strategy", {})
        self._logging_config = self._config.get("logging", {})
        self._state_mapping = self._config.get("state_mapping", {})
        self._training_profiles = self._config.get("training_profiles", {})

        # Setup logging
        self._setup_logging()

        # Apply training profile if specified
        if training_profile:
            self._apply_training_profile(training_profile)

        # Track merge statistics
        self.merge_stats = {
            "sources_loaded": {},
            "records_before_cleaning": 0,
            "records_after_cleaning": 0,
            "duplicates_removed": 0,
            "invalid_records": 0,
            "conflicts_resolved": 0,
            "timestamp": None
        }

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self._logging_config.get("log_level", "INFO")

        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger("DataMerger")
        self.logger.setLevel(getattr(logging, log_level))

        # File handler
        log_file = logs_dir / f"data_merge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, log_level))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        self.logger.info(f"Data merger initialized. Log file: {log_file}")

    def _is_source_enabled(self, source_name: str) -> bool:
        """Check if a data source is enabled for training."""
        source = self._sources.get(source_name, {})
        # Support both old and new config format for backward compatibility
        if "training" in source:
            return source["training"].get("enabled", False)
        return source.get("enabled", False)

    def _get_source_priority(self, source_name: str) -> int:
        """Get priority of a data source (lower = higher priority)."""
        source = self._sources.get(source_name, {})
        # Support both old and new config format for backward compatibility
        if "training" in source:
            return source["training"].get("priority", 999)
        return source.get("priority", 999)

    def _apply_training_profile(self, profile_name: str) -> None:
        """
        Apply a training profile to override source enabled settings.

        Args:
            profile_name: Name of the training profile (e.g., 'h1b_only', 'linkedin_only')
        """
        if profile_name not in self._training_profiles:
            available = ", ".join(self._training_profiles.keys())
            raise ValueError(
                f"Unknown training profile '{profile_name}'. "
                f"Available profiles: {available}"
            )

        profile = self._training_profiles[profile_name]
        self.logger.info(f"Applying training profile: {profile_name}")
        self.logger.info(f"  Description: {profile.get('description', 'N/A')}")

        sources_config = profile.get("sources", {})
        for source_name, enabled in sources_config.items():
            if source_name in self._sources:
                # Update the training.enabled flag for this source
                if "training" not in self._sources[source_name]:
                    self._sources[source_name]["training"] = {}
                self._sources[source_name]["training"]["enabled"] = enabled
                status = "enabled" if enabled else "disabled"
                self.logger.info(f"  {source_name}: {status}")

    def _load_source(self, source_name: str) -> pd.DataFrame:
        """
        Load data source with validation.

        Args:
            source_name: Name of data source

        Returns:
            DataFrame with loaded data
        """
        if not self._is_source_enabled(source_name):
            self.logger.info(f"Source '{source_name}' is disabled, skipping")
            return pd.DataFrame()

        if source_name not in self._sources:
            self.logger.warning(f"Unknown data source: {source_name}")
            return pd.DataFrame()

        config = self._sources[source_name]
        filename = config["filename"]

        # Map source to its processed directory
        source_dirs = {
            "h1b": self.h1b_dir,
            "linkedin": self.linkedin_dir,
            "adzuna": self.adzuna_dir,
        }

        source_dir = source_dirs.get(source_name)
        if not source_dir:
            self.logger.warning(f"No directory configured for source: {source_name}")
            return pd.DataFrame()

        filepath = source_dir / filename

        if not filepath.exists():
            self.logger.warning(f"{source_name.upper()} data not found at {filepath}")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(filepath)
            df["data_source"] = source_name
            df["source_priority"] = self._get_source_priority(source_name)

            self.logger.info(
                f"Loaded {len(df)} records from {source_name} "
                f"(priority: {self._get_source_priority(source_name)})"
            )

            return df
        except Exception as e:
            self.logger.error(f"Error loading {source_name}: {e}")
            return pd.DataFrame()

    def _standardize_source(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Standardize data source to common schema.

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
                    standardized[target_col] = df[source_cols]
                else:
                    standardized[target_col] = None

        # Preserve source metadata
        standardized["data_source"] = df["data_source"]
        standardized["source_priority"] = df["source_priority"]

        return pd.DataFrame(standardized)

    def _validate_data_quality(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Validate and filter data based on quality thresholds.

        Args:
            df: DataFrame to validate
            source_name: Source name for logging

        Returns:
            Validated DataFrame
        """
        if df.empty:
            return df

        original_count = len(df)

        # Check required fields
        required_fields = self._validation.get("required_fields", [])
        for field in required_fields:
            if field in df.columns:
                before = len(df)
                df = df[df[field].notna()]
                removed = before - len(df)
                if removed > 0:
                    self.logger.info(
                        f"  {source_name}: Removed {removed} records missing '{field}'"
                    )

        # Validate salary range
        salary_config = self._validation.get("salary", {})
        if "annual_salary" in df.columns:
            min_sal = salary_config.get("min", 0)
            max_sal = salary_config.get("max", float('inf'))

            before = len(df)

            # Basic range filter
            df = df[
                (df["annual_salary"] >= min_sal) &
                (df["annual_salary"] <= max_sal)
            ]
            range_removed = before - len(df)

            # IQR-based outlier detection (if enabled)
            if salary_config.get("use_iqr_filter", False) and len(df) > 10:
                before_iqr = len(df)
                salaries = df["annual_salary"]
                q1 = salaries.quantile(0.25)
                q3 = salaries.quantile(0.75)
                iqr = q3 - q1
                multiplier = salary_config.get("iqr_multiplier", 3.0)

                lower_bound = q1 - multiplier * iqr
                upper_bound = q3 + multiplier * iqr

                df = df[
                    (df["annual_salary"] >= max(lower_bound, min_sal)) &
                    (df["annual_salary"] <= min(upper_bound, max_sal))
                ]
                iqr_removed = before_iqr - len(df)

                if iqr_removed > 0:
                    self.logger.info(
                        f"  {source_name}: Removed {iqr_removed} outliers via IQR method "
                        f"(bounds: ${lower_bound:,.0f} - ${upper_bound:,.0f})"
                    )

            total_removed = before - len(df)
            if total_removed > 0:
                self.logger.info(
                    f"  {source_name}: Removed {total_removed} total records with invalid/outlier salary"
                )

        # Check minimum record count
        quality_config = self._validation.get("quality", {})
        min_records = quality_config.get("min_salary_records_per_source", 0)

        if len(df) < min_records:
            self.logger.warning(
                f"  {source_name}: Only {len(df)} records, below minimum {min_records}. "
                f"Consider collecting more data."
            )

        invalid_count = original_count - len(df)
        self.merge_stats["invalid_records"] += invalid_count

        return df

    def _calculate_quality_score(self, row: pd.Series) -> float:
        """
        Calculate quality score for a record.

        Args:
            row: DataFrame row

        Returns:
            Quality score (0-1)
        """
        weights = self._merge_strategy.get("quality_scoring", {})
        score = 0.0

        # Check feature availability
        skills = row.get("skills")
        has_skills = False
        if skills is not None:
            # Handle different types of skills data
            import numpy as np
            if isinstance(skills, np.ndarray):
                has_skills = skills.size > 0
            elif isinstance(skills, (list, tuple)):
                has_skills = len(skills) > 0
            elif isinstance(skills, str):
                has_skills = bool(skills.strip())
            elif not pd.isna(skills):
                has_skills = True

        if has_skills:
            score += weights.get("has_skills", 0)

        if pd.notna(row.get("estimated_yoe")):
            score += weights.get("has_experience", 0)

        if pd.notna(row.get("education")):
            score += weights.get("has_education", 0)

        if pd.notna(row.get("seniority_level")):
            score += weights.get("has_seniority", 0)

        if pd.notna(row.get("employer_name")):
            score += weights.get("has_employer", 0)

        if pd.notna(row.get("location_state")):
            score += weights.get("has_location", 0)

        # Source priority (normalize: priority 1 = full weight, priority 3 = 1/3 weight)
        priority = row.get("source_priority", 999)
        if priority < 999:
            priority_weight = 1.0 / priority
            score += weights.get("source_priority", 0) * priority_weight

        return min(score, 1.0)

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (0-1)."""
        if pd.isna(s1) or pd.isna(s2):
            return 0.0
        return SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio()

    def _deduplicate_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate records with conflict resolution.

        Args:
            df: DataFrame to deduplicate

        Returns:
            Deduplicated DataFrame
        """
        if df.empty:
            return df

        dedup_config = self._merge_strategy.get("deduplication", {})
        match_fields = dedup_config.get("match_fields", [])
        similarity_threshold = dedup_config.get("similarity_threshold", 0.85)

        # Add quality scores
        self.logger.info("Calculating quality scores for deduplication...")
        df["quality_score"] = df.apply(self._calculate_quality_score, axis=1)

        # Simple exact match deduplication first
        exact_match_cols = [c for c in match_fields if c in df.columns]

        # Optionally deduplicate within each source (prevents cross-source collapse)
        if dedup_config.get("group_by_source", False):
            if "data_source" not in exact_match_cols and "data_source" in df.columns:
                exact_match_cols.append("data_source")
        if exact_match_cols:
            before_count = len(df)

            # Sort by quality score (descending) and source priority (ascending)
            df = df.sort_values(
                ["quality_score", "source_priority"],
                ascending=[False, True]
            )

            # Keep first (highest quality) of exact matches
            df = df.drop_duplicates(subset=exact_match_cols, keep="first")

            duplicates_removed = before_count - len(df)
            self.merge_stats["duplicates_removed"] = duplicates_removed

            if duplicates_removed > 0:
                self.logger.info(f"Removed {duplicates_removed} exact duplicate records")

        # TODO: Implement fuzzy matching for near-duplicates
        # This would use similarity_threshold for more sophisticated deduplication

        return df

    def merge_all_sources(self) -> pd.DataFrame:
        """
        Merge all enabled data sources with robust strategy.

        Returns:
            Merged and validated DataFrame
        """
        self.merge_stats["timestamp"] = datetime.now().isoformat()
        self.logger.info("--- Starting data merge process ---")

        all_data = []

        # Load and process each enabled source
        for source_name in self._sources.keys():
            if not self._is_source_enabled(source_name):
                continue

            self.logger.info(f"[{source_name.upper()}] Loading data...")

            # Load
            df = self._load_source(source_name)
            if df.empty:
                continue

            self.merge_stats["sources_loaded"][source_name] = len(df)

            # Standardize
            df = self._standardize_source(df, source_name)

            # Validate quality
            df = self._validate_data_quality(df, source_name)

            if not df.empty:
                all_data.append(df)
                self.logger.info(
                    f"  {source_name}: {len(df)} valid records after quality checks"
                )

        if not all_data:
            self.logger.error("No data sources available!")
            return pd.DataFrame()

        # Combine all data
        self.logger.info("Combining data sources...")
        merged_df = pd.concat(all_data, ignore_index=True)
        self.merge_stats["records_before_cleaning"] = len(merged_df)
        self.logger.info(f"Combined: {len(merged_df)} total records")

        # Deduplicate
        self.logger.info("Deduplicating records...")
        merged_df = self._deduplicate_records(merged_df)
        self.merge_stats["records_after_cleaning"] = len(merged_df)

        # Clean and standardize
        self.logger.info("Final cleaning and standardization...")
        merged_df = self._clean_merged_data(merged_df)

        # Save
        output_path = self.merged_dir / "merged_salary_data.parquet"
        merged_df.to_parquet(output_path, index=False)

        self.logger.info(f"Saved merged data to {output_path}")
        self.logger.info(f"Final record count: {len(merged_df)}")

        # Save merge report
        if self._logging_config.get("save_merge_report", False):
            self._save_merge_report(merged_df)

        return merged_df

    def _clean_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize merged data."""
        df = df.copy()

        # Standardize state names
        if "location_state" in df.columns:
            df["location_state"] = df["location_state"].str.upper().str.strip()
            df["location_state"] = df["location_state"].replace(self._state_mapping)

        # Fill missing years
        if "year" in df.columns:
            default_year = self._validation.get("year", {}).get("default", 2024)
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["year"] = df["year"].fillna(default_year).astype(int)

        return df

    def _save_merge_report(self, df: pd.DataFrame):
        """Save detailed merge report."""
        report_path = self.merged_dir / f"merge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        import json

        report = {
            **self.merge_stats,
            "summary_statistics": self.get_summary_statistics(df),
            "source_distribution": df["data_source"].value_counts().to_dict() if "data_source" in df.columns else {},
            "quality_score_stats": {
                "mean": float(df["quality_score"].mean()) if "quality_score" in df.columns else None,
                "median": float(df["quality_score"].median()) if "quality_score" in df.columns else None,
                "std": float(df["quality_score"].std()) if "quality_score" in df.columns else None
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Merge report saved to {report_path}")

    def get_summary_statistics(self, df: pd.DataFrame) -> dict:
        """Get summary statistics for merged data."""
        stats = {
            "total_records": len(df),
            "unique_employers": df["employer_name"].nunique() if "employer_name" in df.columns else 0,
            "unique_titles": df["job_title"].nunique() if "job_title" in df.columns else 0,
        }

        if "annual_salary" in df.columns:
            salary_stats = df["annual_salary"].describe()
            stats["salary_mean"] = float(salary_stats["mean"])
            stats["salary_median"] = float(salary_stats["50%"])
            stats["salary_std"] = float(salary_stats["std"])
            stats["salary_min"] = float(salary_stats["min"])
            stats["salary_max"] = float(salary_stats["max"])

        if "data_source" in df.columns:
            stats["records_by_source"] = df["data_source"].value_counts().to_dict()

        return stats

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_year: int = 2024,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        train_val_df = train_val_df.sample(frac=1, random_state=42)

        val_df = train_val_df.iloc[:val_size]
        train_df = train_val_df.iloc[val_size:]

        self.logger.info(f"Train set: {len(train_df)} records (years < {test_year})")
        self.logger.info(f"Validation set: {len(val_df)} records")
        self.logger.info(f"Test set: {len(test_df)} records (year >= {test_year})")

        return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    merger = DataMerger()

    # Merge all enabled sources
    merged_df = merger.merge_all_sources()

    if not merged_df.empty:
        print("\n" + "="*60)
        print("Merge Statistics:")
        print("="*60)
        for key, value in merger.merge_stats.items():
            print(f"  {key}: {value}")
