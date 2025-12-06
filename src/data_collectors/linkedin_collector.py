"""
LinkedIn Jobs Collector

Fetches consolidated LinkedIn job data from S3 with salary parsing.
Built with Claude Code CLI assistance.

S3 bucket: s3://ai-salary-predictor/data/linkedin/
"""

import json
import re
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

import pandas as pd
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


class LinkedInJobsCollector:
    """Collector for LinkedIn job salary data from S3."""

    def __init__(self, data_dir: str = "data/linkedin/raw", bucket: str = "ai-salary-predictor"):
        """
        Initialize the LinkedIn jobs collector.

        Args:
            data_dir: Local directory to cache downloaded data
            bucket: S3 bucket name
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.bucket = bucket

        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3')
        except NoCredentialsError:
            print("Warning: AWS credentials not found. S3 access will fail.")
            self.s3_client = None

    def _list_s3_files(
        self,
        prefix: str,
        file_pattern: Optional[str] = None,
        days_back: Optional[int] = None
    ) -> List[str]:
        """
        List files in S3 bucket with optional filtering.

        Args:
            prefix: S3 prefix/folder path
            file_pattern: Optional regex pattern to filter filenames
            days_back: Only include files from last N days

        Returns:
            List of S3 keys (file paths)
        """
        if not self.s3_client:
            print("S3 client not initialized")
            return []

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )

            if 'Contents' not in response:
                print(f"No files found in s3://{self.bucket}/{prefix}")
                return []

            files = []
            cutoff_date = None
            if days_back:
                cutoff_date = datetime.now() - timedelta(days=days_back)

            for obj in response['Contents']:
                key = obj['Key']

                # Filter by pattern
                if file_pattern and not re.search(file_pattern, key):
                    continue

                # Filter by date
                if cutoff_date and obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    continue

                files.append(key)

            return files

        except ClientError as e:
            print(f"Error listing S3 files: {e}")
            return []

    def _download_s3_file(self, s3_key: str) -> Optional[Path]:
        """
        Download a file from S3 to local cache.

        Args:
            s3_key: S3 object key (path)

        Returns:
            Path to downloaded file, or None if failed
        """
        if not self.s3_client:
            return None

        # Create local filename from S3 key
        filename = Path(s3_key).name
        local_path = self.data_dir / filename

        # Check if already cached
        if local_path.exists():
            print(f"Using cached file: {local_path}")
            return local_path

        try:
            print(f"Downloading s3://{self.bucket}/{s3_key}...")
            self.s3_client.download_file(
                self.bucket,
                s3_key,
                str(local_path)
            )
            return local_path

        except ClientError as e:
            print(f"Error downloading {s3_key}: {e}")
            return None

    def _parse_jsonl_file(self, filepath: Path) -> pd.DataFrame:
        """
        Parse a JSONL file into a DataFrame.

        Args:
            filepath: Path to JSONL file

        Returns:
            DataFrame with job records
        """
        records = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num} in {filepath.name}: {e}")
                    continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        return df

    def _parse_salary_text(self, salary_text: str) -> Optional[float]:
        """
        Parse salary text into annual salary amount.

        Examples:
            "$150,000 - $200,000" -> 175000 (midpoint)
            "$150K - $200K" -> 175000
            "$150,000/yr" -> 150000
            "$50/hr" -> 104000 (converted to annual)
            "Not specified" -> None

        Args:
            salary_text: Salary string from LinkedIn

        Returns:
            Annual salary as float, or None if not parseable
        """
        if not salary_text or pd.isna(salary_text):
            return None

        text = str(salary_text).lower()

        if text in ["not specified", "none", ""]:
            return None

        # Detect if this is an hourly rate
        is_hourly = '/hr' in text or 'per hour' in text or 'hourly' in text

        # Remove currency symbols, commas, and rate indicators
        text = text.replace('$', '').replace(',', '').replace('/yr', '').replace('/hr', '').replace('per hour', '').replace('hourly', '').strip()

        # Handle K notation (e.g., "150K")
        text = re.sub(r'(\d+)k', lambda m: str(int(m.group(1)) * 1000), text)

        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', text)

        if not numbers:
            return None

        if len(numbers) == 1:
            # Single salary value
            salary = float(numbers[0])
        else:
            # Range - return midpoint
            values = [float(n) for n in numbers]
            salary = sum(values) / len(values)

        # Convert hourly to annual if needed
        # Only convert if explicitly marked as hourly OR clearly hourly rate
        # Heuristic: values under $300 are likely hourly (minimum wage * 2080 hrs = ~$15k/yr)
        if is_hourly:
            # Convert hourly to annual: hourly_rate × 40 hours/week × 52 weeks/year
            salary = salary * 40 * 52
        elif salary < 300 and not is_hourly:
            # Likely hourly but not explicitly marked
            # Only convert if result would be reasonable ($15-300 * 2080 = $31k-$624k)
            annual_estimate = salary * 40 * 52
            if annual_estimate >= 30000 and annual_estimate <= 700000:
                salary = annual_estimate
            # Otherwise leave as-is (will be filtered by validation)

        return salary

    def _parse_experience_years(self, exp_text: str) -> Optional[int]:
        """
        Parse experience requirement into years.

        Examples:
            "5+ years" -> 5
            "3-5 years" -> 4 (midpoint)
            "Not specified" -> None

        Args:
            exp_text: Experience requirement text

        Returns:
            Years of experience as integer
        """
        if not exp_text or pd.isna(exp_text) or exp_text == "Not specified":
            return None

        text = str(exp_text).lower()

        # Match patterns like "5+", "3-5", "5 years"
        patterns = [
            r'(\d+)\+',  # "5+"
            r'(\d+)\s*-\s*(\d+)',  # "3-5"
            r'(\d+)',  # "5"
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    # Range - return midpoint
                    return (int(match.group(1)) + int(match.group(2))) // 2
                else:
                    return int(match.group(1))

        return None

    def _filter_ai_ml_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to only AI/ML relevant jobs, excluding data entry and clerical positions.

        Args:
            df: DataFrame with job_title column

        Returns:
            Filtered DataFrame with only AI/ML relevant jobs
        """
        if df.empty or 'job_title' not in df.columns:
            return df

        # Include keywords - jobs that should be kept
        ai_ml_keywords = [
            'machine learning', 'ml engineer', 'data scientist', 'ai engineer',
            'deep learning', 'nlp', 'natural language processing', 'computer vision',
            'mlops', 'research scientist', 'applied scientist', 'ai research',
            'neural network', 'tensorflow', 'pytorch', 'data science',
            'artificial intelligence', 'quantitative', 'analytics engineer',
            'modeling', 'algorithm', 'big data', 'hadoop', 'spark'
        ]

        # Exclude keywords - jobs that should be filtered out
        exclude_keywords = [
            'data entry', 'virtual assistant', 'clerk', 'typist', 'typing',
            'checkout', 'fashion', 'administrative', 'capacitador', 'receptionist',
            'medical office', 'office assistant', 'customer service', 'cashier',
            'retail', 'sales associate', 'warehouse', 'driver', 'delivery',
            'cleaner', 'janitor', 'security guard', 'restaurant', 'cook',
            'dishwasher', 'barista', 'waiter', 'waitress', 'hostess'
        ]

        # Create lowercase title column for matching
        df['_title_lower'] = df['job_title'].str.lower().fillna('')

        # Must match at least one AI/ML keyword
        include_mask = df['_title_lower'].apply(
            lambda x: any(kw in str(x) for kw in ai_ml_keywords)
        )

        # Must NOT match any exclude keyword
        exclude_mask = df['_title_lower'].apply(
            lambda x: any(kw in str(x) for kw in exclude_keywords)
        )

        # Keep jobs that match AI/ML keywords AND don't match exclude keywords
        filtered_df = df[include_mask & ~exclude_mask].drop('_title_lower', axis=1)

        if len(df) > 0:
            removed_count = len(df) - len(filtered_df)
            removed_pct = (removed_count / len(df)) * 100
            print(f"AI/ML Job Filtering: Kept {len(filtered_df):,} jobs, removed {removed_count:,} ({removed_pct:.1f}%)")

            # Show sample of removed jobs for debugging
            if removed_count > 0:
                removed_df = df[~(include_mask & ~exclude_mask)]
                print(f"  Sample removed jobs: {removed_df['job_title'].value_counts().head(3).to_dict()}")

        return filtered_df

    def _standardize_linkedin_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize LinkedIn data to match H1B schema.

        Args:
            df: Raw LinkedIn DataFrame

        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return df

        standardized = pd.DataFrame()

        # Parse and extract location (split "San Francisco, CA" -> CA)
        if 'location' in df.columns:
            # Extract state from "City, ST" format
            def extract_state(loc):
                if not loc or pd.isna(loc):
                    return None
                parts = str(loc).split(',')
                if len(parts) >= 2:
                    state = parts[-1].strip()
                    # Handle state abbreviations (e.g., "CA")
                    if len(state) == 2:
                        return state.upper()
                return None

            standardized['worksite_state'] = df['location'].apply(extract_state)

        # Map LinkedIn fields to standard schema
        field_mappings = {
            'position': 'job_title',
            'company': 'employer_name',
            'posted_date': 'received_date',
            'seniority_level': 'seniority_level',
            'employment_type': 'employment_type',
            'job_function': 'job_function',
            'industries': 'industries',
            'education': 'education',
        }

        for linkedin_col, standard_col in field_mappings.items():
            if linkedin_col in df.columns:
                standardized[standard_col] = df[linkedin_col]

        # Parse salary text to numeric
        if 'salary' in df.columns:
            standardized['annual_salary'] = df['salary'].apply(self._parse_salary_text)

        # Parse experience years
        if 'experience_years' in df.columns:
            standardized['estimated_yoe'] = df['experience_years'].apply(self._parse_experience_years)

        # Skills - keep as JSON array (will be processed by feature engineering)
        if 'skills' in df.columns:
            standardized['skills'] = df['skills'].apply(
                lambda x: x if isinstance(x, list) else []
            )

        # Add metadata
        standardized['data_source'] = 'linkedin'

        if 'collected_at' in df.columns:
            standardized['collection_date'] = pd.to_datetime(df['collected_at'], errors='coerce')
            standardized['year'] = standardized['collection_date'].dt.year
        else:
            standardized['year'] = datetime.now().year

        # Add URL for reference
        if 'job_url' in df.columns:
            standardized['job_url'] = df['job_url']

        # Filter to AI/ML jobs only (remove data entry, clerical jobs)
        standardized = self._filter_ai_ml_jobs(standardized)

        return standardized

    def fetch_latest_consolidated(self) -> pd.DataFrame:
        """
        Fetch the latest consolidated (deduplicated) LinkedIn data from S3.

        Returns:
            DataFrame with LinkedIn job data
        """
        print("Fetching latest consolidated LinkedIn data from S3...")

        # List consolidated files
        files = self._list_s3_files(
            prefix="data/linkedin/processed/",
            file_pattern=r"consolidated-.*\.jsonl$"
        )

        if not files:
            print("No consolidated files found in S3")
            return pd.DataFrame()

        # Get the most recent file (by name, which includes timestamp)
        latest_file = sorted(files)[-1]
        print(f"Latest file: {latest_file}")

        # Download and parse
        local_path = self._download_s3_file(latest_file)
        if not local_path:
            return pd.DataFrame()

        df = self._parse_jsonl_file(local_path)
        print(f"Loaded {len(df)} records from {latest_file}")

        return df

    def fetch_all_consolidated(self, include_raw: bool = True) -> pd.DataFrame:
        """
        Fetch ALL consolidated LinkedIn data files from S3 and merge them.

        This method loads all historical consolidated files and deduplicates
        them to create a comprehensive dataset. Use this to avoid data loss
        from point-in-time snapshots.

        Args:
            include_raw: Also include raw files (ai-jobs-* and batch-*) from data/linkedin/raw/
                        Default True to capture all data.

        Returns:
            DataFrame with all unique LinkedIn job data across all files
        """
        print("Fetching ALL historical LinkedIn data from S3...")

        all_data = []
        seen_urls = set()

        # 1. Fetch all consolidated files
        print("\n[1/2] Loading consolidated files...")
        consolidated_files = self._list_s3_files(
            prefix="data/linkedin/processed/",
            file_pattern=r"consolidated-.*\.jsonl$"
        )

        if consolidated_files:
            print(f"Found {len(consolidated_files)} consolidated files")

            for s3_key in sorted(consolidated_files):
                local_path = self._download_s3_file(s3_key)
                if not local_path:
                    continue

                df = self._parse_jsonl_file(local_path)
                if df.empty:
                    continue

                # Deduplicate by job_url
                if 'job_url' in df.columns:
                    mask = ~df['job_url'].isin(seen_urls)
                    unique_df = df[mask]
                    seen_urls.update(unique_df['job_url'].tolist())
                    all_data.append(unique_df)
                    print(f"  {local_path.name}: +{len(unique_df)} unique (total: {len(seen_urls)})")
                else:
                    all_data.append(df)
        else:
            print("No consolidated files found")

        # 2. Fetch raw files (ai-jobs-* and batch-*)
        if include_raw:
            print("\n[2/2] Loading raw files...")
            raw_files = self._list_s3_files(
                prefix="data/linkedin/raw/",
                file_pattern=r"(ai-jobs-|batch-).*\.jsonl$"
            )

            if raw_files:
                print(f"Found {len(raw_files)} raw files")

                for s3_key in sorted(raw_files):
                    local_path = self._download_s3_file(s3_key)
                    if not local_path:
                        continue

                    df = self._parse_jsonl_file(local_path)
                    if df.empty:
                        continue

                    # Deduplicate
                    if 'job_url' in df.columns:
                        mask = ~df['job_url'].isin(seen_urls)
                        unique_df = df[mask]
                        if not unique_df.empty:
                            seen_urls.update(unique_df['job_url'].tolist())
                            all_data.append(unique_df)
                            print(f"  {local_path.name}: +{len(unique_df)} unique (total: {len(seen_urls)})")
            else:
                print("No raw files found")
        else:
            print("\n[2/2] Skipping raw files (include_raw=False)")

        if not all_data:
            print("\nNo data found!")
            return pd.DataFrame()

        # Combine all unique data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Total unique jobs collected: {len(combined_df)}")

        return combined_df

    def fetch_raw_data(self, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch raw LinkedIn data from S3 (from last N days).

        Args:
            days_back: Number of days to look back for data

        Returns:
            DataFrame with LinkedIn job data
        """
        print(f"Fetching LinkedIn data from last {days_back} days...")

        # List raw data files
        files = self._list_s3_files(
            prefix="data/linkedin/raw/",
            file_pattern=r"(ai-jobs-|batch-).*\.jsonl$",
            days_back=days_back
        )

        if not files:
            print(f"No raw files found from last {days_back} days")
            return pd.DataFrame()

        print(f"Found {len(files)} files")

        # Download and parse all files
        all_data = []
        for s3_key in files:
            local_path = self._download_s3_file(s3_key)
            if not local_path:
                continue

            df = self._parse_jsonl_file(local_path)
            if not df.empty:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Deduplicate by job_url
        if 'job_url' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['job_url'], keep='first')

        print(f"Loaded {len(combined_df)} unique records")

        return combined_df

    def collect(
        self,
        use_consolidated: bool = True,
        use_all_consolidated: bool = False,
        days_back: int = 7,
    ) -> pd.DataFrame:
        """
        Collect and process LinkedIn job data.

        Args:
            use_consolidated: Use consolidated file(s) (recommended)
            use_all_consolidated: If True, fetch ALL consolidated files (more data).
                                 If False, fetch only latest consolidated file.
                                 Ignored if use_consolidated=False.
            days_back: If not using consolidated, fetch raw data from last N days

        Returns:
            Processed DataFrame with LinkedIn salary data
        """
        print("\n" + "=" * 60)
        print("LinkedIn Jobs Data Collection")
        print("=" * 60)

        # Fetch data
        if use_consolidated:
            if use_all_consolidated:
                df = self.fetch_all_consolidated()
            else:
                df = self.fetch_latest_consolidated()
        else:
            df = self.fetch_raw_data(days_back=days_back)

        if df.empty:
            print("No LinkedIn data available")
            return pd.DataFrame()

        # Standardize to common schema
        print("\nStandardizing LinkedIn data...")
        standardized_df = self._standardize_linkedin_data(df)

        # Filter out records without salary
        if 'annual_salary' in standardized_df.columns:
            before_count = len(standardized_df)
            standardized_df = standardized_df[standardized_df['annual_salary'].notna()]
            print(f"Filtered to {len(standardized_df)} records with salary data (from {before_count})")

        # Save processed data
        output_path = self.data_dir.parent / "processed" / "linkedin_ai_jobs.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        standardized_df.to_parquet(output_path, index=False)
        print(f"\nSaved {len(standardized_df)} records to {output_path}")

        return standardized_df

    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for LinkedIn data.

        Args:
            df: DataFrame with LinkedIn data

        Returns:
            Dictionary with summary statistics
        """
        if df.empty or 'annual_salary' not in df.columns:
            return {}

        # Filter valid salaries
        valid_salaries = df[
            (df['annual_salary'].notna()) &
            (df['annual_salary'] >= 50000) &
            (df['annual_salary'] <= 1000000)
        ]['annual_salary']

        stats = {
            'count': len(valid_salaries),
            'mean': valid_salaries.mean(),
            'median': valid_salaries.median(),
            'std': valid_salaries.std(),
            'min': valid_salaries.min(),
            'max': valid_salaries.max(),
            'percentile_25': valid_salaries.quantile(0.25),
            'percentile_75': valid_salaries.quantile(0.75),
            'percentile_90': valid_salaries.quantile(0.90),
        }

        return stats


if __name__ == "__main__":
    # Example usage
    collector = LinkedInJobsCollector()
    df = collector.collect(use_consolidated=True)

    if not df.empty:
        stats = collector.get_summary_stats(df)
        print("\nLinkedIn Salary Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: ${value:,.2f}")
            else:
                print(f"  {key}: {value}")
