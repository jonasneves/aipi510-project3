"""
Merge All LinkedIn Data Script

Combines ALL historical LinkedIn data from:
1. All consolidated files in s3://ai-salary-predictor/data/linkedin/processed/
2. Raw batch files in s3://ai-salary-predictor/data/linkedin/raw/ (optional)

Creates a comprehensive, deduplicated dataset for model training.

Usage:
    python scripts/merge_all_linkedin_data.py [--include-raw-batches]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Set, List, Dict

import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_collectors.linkedin_collector import LinkedInJobsCollector


class ComprehensiveLinkedInMerger:
    """Merges all historical LinkedIn data from S3."""

    def __init__(self, bucket: str = "ai-salary-predictor", data_dir: str = "data/raw"):
        """Initialize merger."""
        self.bucket = bucket
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.s3_client = boto3.client('s3')
        except NoCredentialsError:
            raise RuntimeError("AWS credentials not found. Configure AWS CLI first.")

        self.collector = LinkedInJobsCollector(data_dir=str(data_dir), bucket=bucket)

    def list_all_files(self, prefix: str, pattern: str = None) -> List[str]:
        """List all files in S3 prefix."""
        print(f"Listing files in s3://{self.bucket}/{prefix}...")

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

            files = []
            for page in pages:
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    if pattern and pattern not in key:
                        continue
                    files.append(key)

            return files

        except ClientError as e:
            print(f"Error listing files: {e}")
            return []

    def download_and_parse_jsonl(self, s3_key: str) -> List[Dict]:
        """Download and parse a JSONL file from S3."""
        filename = Path(s3_key).name
        local_path = self.data_dir / filename

        # Download if not cached
        if not local_path.exists():
            try:
                print(f"  Downloading {filename}...")
                self.s3_client.download_file(self.bucket, s3_key, str(local_path))
            except ClientError as e:
                print(f"  Error downloading {s3_key}: {e}")
                return []

        # Parse JSONL
        records = []
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Error parsing line {line_num} in {filename}: {e}")
                        continue
        except Exception as e:
            print(f"  Error reading {local_path}: {e}")
            return []

        return records

    def merge_all_sources(self, include_raw_batches: bool = False) -> pd.DataFrame:
        """
        Merge all LinkedIn data sources.

        Args:
            include_raw_batches: Whether to include raw batch files (slower)

        Returns:
            Merged and deduplicated DataFrame
        """
        print("=" * 70)
        print("COMPREHENSIVE LINKEDIN DATA MERGE")
        print("=" * 70)

        all_jobs = []
        seen_urls = set()

        # 1. Merge all consolidated files
        print("\n[1/3] Fetching ALL consolidated files...")
        consolidated_files = self.list_all_files(
            prefix="data/linkedin/processed/",
            pattern="consolidated-"
        )
        print(f"Found {len(consolidated_files)} consolidated files")

        for s3_key in consolidated_files:
            records = self.download_and_parse_jsonl(s3_key)
            print(f"  {Path(s3_key).name}: {len(records)} records")

            for record in records:
                job_url = record.get('job_url')
                if job_url and job_url not in seen_urls:
                    all_jobs.append(record)
                    seen_urls.add(job_url)

        print(f"  Unique jobs from consolidated: {len(all_jobs)}")

        # 2. Optionally merge raw batches
        if include_raw_batches:
            print("\n[2/2] Fetching raw batch files...")
            batch_files = self.list_all_files(
                prefix="data/linkedin/raw/",
                pattern="batch-"
            )
            print(f"Found {len(batch_files)} batch files")

            count_before = len(all_jobs)
            for s3_key in batch_files:
                records = self.download_and_parse_jsonl(s3_key)

                for record in records:
                    job_url = record.get('job_url')
                    if job_url and job_url not in seen_urls:
                        all_jobs.append(record)
                        seen_urls.add(job_url)

            print(f"  Added {len(all_jobs) - count_before} unique jobs from batch files")
        else:
            print("\n[2/2] Skipping raw batch files (use --include-raw-batches to include)")

        # Convert to DataFrame
        print(f"\n--- MERGE COMPLETE ---")
        print(f"Total unique jobs collected: {len(all_jobs)}")

        if not all_jobs:
            print("No data collected!")
            return pd.DataFrame()

        df = pd.DataFrame(all_jobs)

        # Standardize using existing collector logic
        print("\nStandardizing data...")
        standardized_df = self.collector._standardize_linkedin_data(df)

        # Filter to records with salary
        if 'annual_salary' in standardized_df.columns:
            before_count = len(standardized_df)
            standardized_df = standardized_df[standardized_df['annual_salary'].notna()]

            # Additional filtering for reasonable salaries
            standardized_df = standardized_df[
                (standardized_df['annual_salary'] >= 30000) &
                (standardized_df['annual_salary'] <= 1000000)
            ]

            print(f"Filtered to {len(standardized_df)} records with valid salary data")
            print(f"  (removed {before_count - len(standardized_df)} records without/invalid salary)")

        return standardized_df

    def save_merged_data(self, df: pd.DataFrame) -> Path:
        """Save merged data to local and S3."""
        if df.empty:
            print("No data to save!")
            return None

        # Save locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_path = self.data_dir / f"linkedin_merged_all_{timestamp}.parquet"
        df.to_parquet(local_path, index=False)
        print(f"\n✓ Saved merged data locally: {local_path}")

        # Also save as "latest" version
        latest_path = self.data_dir / "linkedin_ai_jobs.parquet"
        df.to_parquet(latest_path, index=False)
        print(f"✓ Updated latest version: {latest_path}")

        # Upload to S3
        s3_key = f"data/raw/parquet/linkedin_merged_all_{timestamp}.parquet"
        try:
            print(f"\nUploading to S3: s3://{self.bucket}/{s3_key}...")
            self.s3_client.upload_file(str(local_path), self.bucket, s3_key)
            print("✓ Uploaded to S3")

            # Also update the "latest" version in S3
            s3_latest_key = "data/raw/parquet/linkedin_ai_jobs.parquet"
            self.s3_client.upload_file(str(latest_path), self.bucket, s3_latest_key)
            print(f"✓ Updated S3 latest version: s3://{self.bucket}/{s3_latest_key}")

        except ClientError as e:
            print(f"Warning: Failed to upload to S3: {e}")

        return local_path

    def print_statistics(self, df: pd.DataFrame) -> None:
        """Print summary statistics."""
        if df.empty:
            return

        print("\n" + "=" * 70)
        print("MERGED DATA STATISTICS")
        print("=" * 70)

        print(f"\nTotal records: {len(df)}")

        if 'worksite_state' in df.columns:
            print(f"\nTop 10 states:")
            top_states = df['worksite_state'].value_counts().head(10)
            for state, count in top_states.items():
                print(f"  {state}: {count}")

        if 'job_title' in df.columns:
            print(f"\nTop 10 job titles:")
            top_titles = df['job_title'].value_counts().head(10)
            for title, count in top_titles.items():
                print(f"  {title}: {count}")

        if 'annual_salary' in df.columns:
            salaries = df['annual_salary'].dropna()
            if len(salaries) > 0:
                print(f"\nSalary statistics:")
                print(f"  Count: {len(salaries)}")
                print(f"  Mean: ${salaries.mean():,.0f}")
                print(f"  Median: ${salaries.median():,.0f}")
                print(f"  Min: ${salaries.min():,.0f}")
                print(f"  Max: ${salaries.max():,.0f}")
                print(f"  25th percentile: ${salaries.quantile(0.25):,.0f}")
                print(f"  75th percentile: ${salaries.quantile(0.75):,.0f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge all historical LinkedIn data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--include-raw-batches",
        action="store_true",
        help="Include raw batch files (slower, usually not needed)"
    )
    parser.add_argument(
        "--bucket",
        default="ai-salary-predictor",
        help="S3 bucket name"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Local data directory"
    )

    args = parser.parse_args()

    # Create merger and run
    merger = ComprehensiveLinkedInMerger(
        bucket=args.bucket,
        data_dir=args.data_dir
    )

    # Merge all data
    df = merger.merge_all_sources(include_raw_batches=args.include_raw_batches)

    # Save results
    if not df.empty:
        merger.save_merged_data(df)
        merger.print_statistics(df)

        print("\n" + "=" * 70)
        print("✓ MERGE COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run: python -m src.main merge")
        print("  2. Run: python -m src.main train --tune")
        print("  3. Your model will now use ALL historical LinkedIn data!")
    else:
        print("\n✗ No data was merged. Check your S3 bucket and credentials.")
        sys.exit(1)


if __name__ == "__main__":
    main()
