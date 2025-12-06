"""
Feature Engineering Pipeline

Creates features for salary prediction from multiple data sources.
Combines H1B data, LinkedIn Jobs, and Adzuna job postings.
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import config loader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import ConfigLoader


class FeatureEngineer:
    """Feature engineering for salary prediction model."""

    def __init__(self):
        """Initialize feature engineer with config-loaded values."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self._fitted = False

        config = ConfigLoader.get_features()
        self.SKILL_CATEGORIES = config["skill_categories"]
        self.COMPANY_TIERS = config["company_tiers"]
        self.LOCATION_COL = config["location_multipliers"]
        self._tech_hubs = config["tech_hubs"]

        # Fixed encodings for consistency across training and prediction
        encodings = config.get("encodings", {})
        self._state_index = encodings.get("state_index", {})
        self._tier_index = encodings.get("tier_index", {})

    def extract_skills(self, text: str) -> dict[str, int]:
        """
        Extract skill category flags from job title/description.

        Args:
            text: Job title or description text

        Returns:
            Dictionary with skill category flags
        """
        if pd.isna(text):
            return {f"skill_{cat}": 0 for cat in self.SKILL_CATEGORIES}

        text_lower = str(text).lower()
        skills = {}

        for category, keywords in self.SKILL_CATEGORIES.items():
            skills[f"skill_{category}"] = int(
                any(keyword in text_lower for keyword in keywords)
            )

        return skills

    def extract_company_tier(self, company_name: str) -> str:
        """
        Classify company into tier based on name.

        Args:
            company_name: Company name

        Returns:
            Company tier classification
        """
        if pd.isna(company_name):
            return "unknown"

        company_lower = str(company_name).lower()

        for tier, companies in self.COMPANY_TIERS.items():
            if any(comp in company_lower for comp in companies):
                return tier

        return "other"

    def extract_experience_level(self, title: str) -> dict[str, int]:
        """
        Extract experience level indicators from job title.

        Args:
            title: Job title

        Returns:
            Dictionary with experience level flags
        """
        if pd.isna(title):
            return {
                "is_senior": 0,
                "is_lead": 0,
                "is_principal": 0,
                "is_staff": 0,
                "is_manager": 0,
                "is_director": 0,
                "is_entry": 0,
                "estimated_yoe": 3,  # Default mid-level
            }

        title_lower = str(title).lower()

        levels = {
            "is_senior": int(any(t in title_lower for t in ["senior", "sr.", "sr "])),
            "is_lead": int("lead" in title_lower),
            "is_principal": int("principal" in title_lower),
            "is_staff": int("staff" in title_lower),
            "is_manager": int(any(t in title_lower for t in ["manager", "mgr"])),
            "is_director": int(any(t in title_lower for t in ["director", "head of", "vp"])),
            "is_entry": int(any(t in title_lower for t in ["junior", "jr.", "entry", "associate", "intern"])),
        }

        # Estimate years of experience based on level
        if levels["is_director"]:
            levels["estimated_yoe"] = 12
        elif levels["is_principal"] or levels["is_staff"]:
            levels["estimated_yoe"] = 10
        elif levels["is_lead"] or levels["is_manager"]:
            levels["estimated_yoe"] = 7
        elif levels["is_senior"]:
            levels["estimated_yoe"] = 5
        elif levels["is_entry"]:
            levels["estimated_yoe"] = 1
        else:
            levels["estimated_yoe"] = 3

        return levels

    def extract_role_type(self, title: str) -> dict[str, int]:
        """
        Extract role type indicators.

        Args:
            title: Job title

        Returns:
            Dictionary with role type flags
        """
        if pd.isna(title):
            return {
                "role_engineer": 0,
                "role_scientist": 0,
                "role_analyst": 0,
                "role_architect": 0,
                "role_researcher": 0,
            }

        title_lower = str(title).lower()

        return {
            "role_engineer": int("engineer" in title_lower),
            "role_scientist": int("scientist" in title_lower),
            "role_analyst": int("analyst" in title_lower),
            "role_architect": int("architect" in title_lower),
            "role_researcher": int(any(t in title_lower for t in ["research", "researcher"])),
        }

    def get_location_features(self, state: str) -> dict[str, float]:
        """
        Get location-based features.

        Args:
            state: US state abbreviation

        Returns:
            Dictionary with location features
        """
        col_multiplier = self.LOCATION_COL.get(state, 1.0)

        major_tech_hub = state in self._tech_hubs["major"]
        secondary_hub = state in self._tech_hubs["secondary"]
        remote_friendly = state in self._tech_hubs["remote_friendly"]

        return {
            "col_multiplier": col_multiplier,
            "is_major_tech_hub": int(major_tech_hub),
            "is_secondary_hub": int(secondary_hub),
            "is_remote_friendly": int(remote_friendly),
        }

    def engineer_features(
        self,
        df: pd.DataFrame,
        title_col: str = "job_title",
        company_col: str = "employer_name",
        state_col: str = "worksite_state",
        salary_col: str = "annual_salary",
    ) -> pd.DataFrame:
        """
        Engineer all features for salary prediction.

        Args:
            df: Input DataFrame
            title_col: Column containing job title
            company_col: Column containing company name
            state_col: Column containing state
            salary_col: Target salary column

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Extract skill features from skills column (if available) or job title
        if 'skills' in df.columns:
            # Use actual skills data from LinkedIn
            def extract_skills_from_list(skills):
                """Extract skill categories from skills list."""
                skill_flags = {f"skill_{cat}": 0 for cat in self.SKILL_CATEGORIES}

                if skills is None:
                    return skill_flags

                # Skip scalar/NA values
                if isinstance(skills, (int, float)):
                    if pd.isna(skills):
                        return skill_flags
                if isinstance(skills, str):
                    return skill_flags

                iterable = None
                if isinstance(skills, (list, tuple, set)):
                    iterable = list(skills)
                elif isinstance(skills, np.ndarray):
                    iterable = skills.flatten().tolist()
                else:
                    # Some sources may store JSON strings or other objects; skip those
                    return skill_flags

                if not iterable:
                    return skill_flags

                normalized = []
                for item in iterable:
                    if item is None:
                        continue
                    try:
                        if pd.isna(item):
                            continue
                    except TypeError:
                        # Non-scalar objects (e.g., dict) fall through to string conversion
                        pass
                    normalized.append(str(item).lower())

                if not normalized:
                    return skill_flags

                skills_text = ' '.join(normalized)

                # Check each category
                for category, keywords in self.SKILL_CATEGORIES.items():
                    if any(keyword in skills_text for keyword in keywords):
                        skill_flags[f"skill_{category}"] = 1

                return skill_flags

            skill_features = df['skills'].apply(extract_skills_from_list).apply(pd.Series)
            df = pd.concat([df, skill_features], axis=1)
        elif title_col in df.columns:
            # Fallback to job title if skills column not available
            skill_features = df[title_col].apply(self.extract_skills).apply(pd.Series)
            df = pd.concat([df, skill_features], axis=1)

        # Extract experience level
        if title_col in df.columns:
            exp_features = df[title_col].apply(self.extract_experience_level).apply(pd.Series)
            df = pd.concat([df, exp_features], axis=1)

        # Extract role type
        if title_col in df.columns:
            role_features = df[title_col].apply(self.extract_role_type).apply(pd.Series)
            df = pd.concat([df, role_features], axis=1)

        # Extract company tier
        if company_col in df.columns:
            df["company_tier"] = df[company_col].apply(self.extract_company_tier)

        # Extract location features
        if state_col in df.columns:
            # Standardize state column
            df["state_clean"] = df[state_col].str.upper().str.strip().str[:2]
            location_features = df["state_clean"].apply(self.get_location_features).apply(pd.Series)
            df = pd.concat([df, location_features], axis=1)

        # Create interaction features
        df = self._create_interactions(df)

        # Time-based features
        df = self._add_time_features(df)

        # Remove any duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        df = df.copy()

        # Skill count (total skills present)
        skill_cols = [col for col in df.columns if col.startswith("skill_")]
        if skill_cols:
            df["skill_count"] = df[skill_cols].sum(axis=1)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()

        # Check for date columns
        date_cols = ["received_date", "decision_date", "created_date", "date"]

        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

                if df[col].notna().any():
                    df["year"] = df[col].dt.year
                    df["quarter"] = df[col].dt.quarter
                    df["month"] = df[col].dt.month

                    # Hiring season flag (Q1 and Q4 typically higher)
                    df["is_peak_hiring"] = df["quarter"].isin([1, 4]).astype(int)
                    break

        return df

    def encode_categoricals(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[list[str]] = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Encode categorical variables using fixed mappings from config.

        Args:
            df: Input DataFrame
            categorical_cols: Columns to encode
            fit: Whether to fit encoders (ignored - always uses fixed mappings)

        Returns:
            DataFrame with encoded categoricals
        """
        df = df.copy()

        if categorical_cols is None:
            categorical_cols = ["company_tier", "state_clean"]

        for col in categorical_cols:
            if col not in df.columns:
                continue

            # Use fixed encodings from config for consistency
            if col == "state_clean" and self._state_index:
                default_idx = max(self._state_index.values()) + 1 if self._state_index else 20
                df[f"{col}_encoded"] = df[col].fillna("unknown").astype(str).apply(
                    lambda x: self._state_index.get(x.upper(), default_idx)
                )
            elif col == "company_tier" and self._tier_index:
                default_idx = self._tier_index.get("unknown", 6)
                df[f"{col}_encoded"] = df[col].fillna("unknown").astype(str).apply(
                    lambda x: self._tier_index.get(x, default_idx)
                )
            else:
                # Fallback to LabelEncoder for other columns
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(
                        df[col].fillna("unknown").astype(str)
                    )
                elif col in self.label_encoders:
                    df[col] = df[col].fillna("unknown").astype(str)
                    known_classes = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_classes else "unknown"
                    )
                    df[f"{col}_encoded"] = self.label_encoders[col].transform(df[col])

        return df

    def select_features(self, df: pd.DataFrame) -> tuple[list[str], str]:
        """
        Select features for modeling.

        Args:
            df: DataFrame with all features

        Returns:
            Tuple of (feature column names, target column name)
        """
        # Numeric feature columns
        feature_cols = []

        # Skill features
        feature_cols.extend([col for col in df.columns if col.startswith("skill_")])

        # Experience features
        experience_features = [
            "is_senior", "is_lead", "is_principal", "is_staff",
            "is_manager", "is_director", "is_entry", "estimated_yoe"
        ]
        feature_cols.extend([col for col in experience_features if col in df.columns])

        # Role features
        feature_cols.extend([col for col in df.columns if col.startswith("role_")])

        # Location features
        location_features = [
            "col_multiplier", "is_major_tech_hub",
            "is_secondary_hub", "is_remote_friendly"
        ]
        feature_cols.extend([col for col in location_features if col in df.columns])

        # Interaction features
        if "skill_count" in df.columns:
            feature_cols.append("skill_count")

        # Time features
        time_features = ["year", "quarter", "is_peak_hiring"]
        feature_cols.extend([col for col in time_features if col in df.columns])

        # Encoded categoricals
        feature_cols.extend([col for col in df.columns if col.endswith("_encoded")])

        # Target column
        target_col = None
        for col in ["annual_salary", "salary_avg", "mean_annual_wage", "wage_from"]:
            if col in df.columns:
                target_col = col
                break

        return feature_cols, target_col

    def prepare_for_modeling(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> tuple[pd.DataFrame, list[str], str]:
        """
        Full pipeline to prepare data for modeling.

        Args:
            df: Raw input DataFrame
            fit: Whether to fit transformers

        Returns:
            Tuple of (processed DataFrame, feature columns, target column)
        """
        # Engineer features
        df = self.engineer_features(df)

        # Encode categoricals
        df = self.encode_categoricals(df, fit=fit)

        # Select features
        feature_cols, target_col = self.select_features(df)

        # Remove rows with missing target
        if target_col:
            df = df[df[target_col].notna()]

            # Filter reasonable salary range
            df = df[(df[target_col] >= 50000) & (df[target_col] <= 1000000)]

        self._fitted = True

        print(f"\nPrepared {len(df)} samples with {len(feature_cols)} features")
        print(f"Target column: {target_col}")

        return df, feature_cols, target_col


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample data
    sample_data = pd.DataFrame({
        "job_title": [
            "Senior Machine Learning Engineer",
            "Data Scientist",
            "Junior ML Engineer",
            "Principal AI Researcher",
            "MLOps Engineer",
        ],
        "employer_name": [
            "Google",
            "Tech Startup",
            "Small Company",
            "Meta",
            "Amazon",
        ],
        "worksite_state": ["CA", "NY", "TX", "WA", "WA"],
        "annual_salary": [250000, 150000, 120000, 350000, 180000],
        "received_date": pd.date_range("2024-01-01", periods=5, freq="M"),
    })

    engineer = FeatureEngineer()
    df, features, target = engineer.prepare_for_modeling(sample_data)

    print("\nFeature columns:")
    print(features)

    print("\nSample features:")
    print(df[features].head())
