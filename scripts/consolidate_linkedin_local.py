"""
Consolidate LinkedIn Data Locally

Processes all raw LinkedIn batch files from local storage and creates
a consolidated processed file.

Usage:
    python scripts/consolidate_linkedin_local.py
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd


def parse_salary_text(salary_text: str) -> Optional[float]:
    """
    Parse LinkedIn salary text to annual numeric value.

    Examples:
        "$120K-$150K/yr" -> 135000
        "$50/hr" -> 104000 (50 * 40 * 52)
        "Not specified" -> None
    """
    if not salary_text or pd.isna(salary_text) or salary_text == "Not specified":
        return None

    text = str(salary_text).lower()

    # Check if hourly
    is_hourly = '/hr' in text or 'per hour' in text or 'hourly' in text

    # Extract numbers (handle K for thousands)
    numbers = re.findall(r'(\d+(?:\.\d+)?)\s*k?', text)

    if not numbers:
        return None

    # Convert K to thousands
    values = []
    for num_str in numbers:
        num = float(num_str)
        # If followed by 'k' or part of "$120K" format
        if 'k' in text.lower():
            num = num * 1000
        values.append(num)

    if not values:
        return None

    # Use midpoint if range, otherwise single value
    salary = sum(values) / len(values)

    # Convert hourly to annual if needed
    if is_hourly:
        salary = salary * 40 * 52
    elif salary < 300:
        # Likely hourly but not explicitly marked
        annual_estimate = salary * 40 * 52
        if 30000 <= annual_estimate <= 700000:
            salary = annual_estimate

    return salary


def parse_experience_years(exp_text: str) -> Optional[int]:
    """
    Parse experience requirement into years.

    Examples:
        "5+ years" -> 5
        "3-5 years" -> 4 (midpoint)
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


def extract_state(location: str) -> Optional[str]:
    """Extract state abbreviation from location string."""
    if not location or pd.isna(location):
        return None

    parts = str(location).split(',')
    if len(parts) >= 2:
        state = parts[-1].strip()
        # Handle state abbreviations (e.g., "CA")
        if len(state) == 2:
            return state.upper()

    return None


def standardize_linkedin_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize LinkedIn data to match H1B schema."""
    if df.empty:
        return df

    standardized = pd.DataFrame()

    # Extract state from location
    if 'location' in df.columns:
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
        standardized['annual_salary'] = df['salary'].apply(parse_salary_text)

    # Parse experience years
    if 'experience_years' in df.columns:
        standardized['estimated_yoe'] = df['experience_years'].apply(parse_experience_years)

    # Skills - keep as JSON array
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

    return standardized


def load_local_jsonl(file_path: Path) -> List[Dict]:
    """Load records from a local JSONL file."""
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Error parsing line {line_num} in {file_path.name}: {e}")
                    continue
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return []

    return records


def consolidate_local_linkedin_data():
    """Consolidate all local LinkedIn raw files."""
    print("=" * 70)
    print("CONSOLIDATING LOCAL LINKEDIN DATA")
    print("=" * 70)

    raw_dir = Path("data/linkedin/raw")
    processed_dir = Path("data/linkedin/processed")

    # Collect all files
    all_files = []

    if raw_dir.exists():
        all_files.extend(sorted(raw_dir.glob("batch-*.jsonl")))
        all_files.extend(sorted(raw_dir.glob("ai-jobs-*.jsonl")))

    if processed_dir.exists():
        all_files.extend(sorted(processed_dir.glob("consolidated-*.jsonl")))

    print(f"\nFound {len(all_files)} total files")

    if not all_files:
        print("No files to process!")
        return

    # Load and deduplicate all records
    all_jobs = []
    seen_urls = set()

    print("\nProcessing files...")
    for file_path in all_files:
        records = load_local_jsonl(file_path)
        if not records:
            continue

        added = 0
        for record in records:
            job_url = record.get('job_url')
            if job_url and job_url not in seen_urls:
                all_jobs.append(record)
                seen_urls.add(job_url)
                added += 1

        if added > 0:
            print(f"  {file_path.name}: {len(records)} records ({added} unique)")

    print(f"\nTotal unique jobs collected: {len(all_jobs):,}")

    if not all_jobs:
        print("No data collected!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_jobs)

    # Standardize
    print("\nStandardizing data...")
    standardized_df = standardize_linkedin_data(df)

    # Filter to records with salary
    if 'annual_salary' in standardized_df.columns:
        before_count = len(standardized_df)
        standardized_df = standardized_df[standardized_df['annual_salary'].notna()]

        # Additional filtering for reasonable salaries
        standardized_df = standardized_df[
            (standardized_df['annual_salary'] >= 30000) &
            (standardized_df['annual_salary'] <= 1000000)
        ]

        print(f"Filtered to {len(standardized_df):,} records with valid salary data")
        print(f"  (removed {before_count - len(standardized_df):,} records without/invalid salary)")

    # Save processed data
    output_path = Path("data/linkedin/processed/linkedin_ai_jobs.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized_df.to_parquet(output_path, index=False)

    print(f"\n✓ Saved {len(standardized_df):,} records to {output_path}")

    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    if 'annual_salary' in standardized_df.columns:
        salaries = standardized_df['annual_salary'].dropna()
        if len(salaries) > 0:
            print(f"\nSalary statistics:")
            print(f"  Count: {len(salaries):,}")
            print(f"  Mean: ${salaries.mean():,.0f}")
            print(f"  Median: ${salaries.median():,.0f}")
            print(f"  Min: ${salaries.min():,.0f}")
            print(f"  Max: ${salaries.max():,.0f}")

    if 'worksite_state' in standardized_df.columns:
        print(f"\nTop 10 states:")
        top_states = standardized_df['worksite_state'].value_counts().head(10)
        for state, count in top_states.items():
            print(f"  {state}: {count:,}")

    print("\n" + "=" * 70)
    print("✓ CONSOLIDATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    consolidate_local_linkedin_data()
