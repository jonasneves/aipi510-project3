"""
Adzuna Job Postings API Collector

Collects job postings with salary estimates, rate limiting, and retry logic.

API Documentation: https://developer.adzuna.com/
"""

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class AdzunaJobsCollector:
    """Collector for job posting data from Adzuna API."""

    BASE_URL = "https://api.adzuna.com/v1/api/jobs"

    # AI/ML related search queries
    AI_SEARCH_QUERIES = [
        "machine learning engineer",
        "data scientist",
        "AI engineer",
        "deep learning engineer",
        "NLP engineer",
        "computer vision engineer",
        "MLOps engineer",
        "ML engineer",
        "artificial intelligence",
        "research scientist machine learning",
        "applied scientist",
        "data engineer",
        "LLM engineer",
        "prompt engineer",
    ]

    # Key skills to search for
    SKILL_QUERIES = [
        "pytorch",
        "tensorflow",
        "huggingface",
        "langchain",
        "kubernetes ml",
        "aws sagemaker",
        "azure ml",
        "spark ml",
    ]

    # US locations to search
    LOCATIONS = {
        "san francisco": "California",
        "new york": "New York",
        "seattle": "Washington",
        "austin": "Texas",
        "boston": "Massachusetts",
        "chicago": "Illinois",
        "denver": "Colorado",
        "atlanta": "Georgia",
        "los angeles": "California",
        "washington dc": "District of Columbia",
        "remote": "Remote",
    }

    def __init__(
        self,
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        data_dir: str = "data/raw",
    ):
        """
        Initialize the Adzuna jobs collector.

        Args:
            app_id: Adzuna application ID
            api_key: Adzuna API key
            data_dir: Directory to store downloaded data
        """
        self.app_id = app_id or os.getenv("ADZUNA_APP_ID")
        self.api_key = api_key or os.getenv("ADZUNA_API_KEY")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def _check_credentials(self) -> bool:
        """Check if API credentials are available."""
        if not self.app_id or not self.api_key:
            print("Warning: Adzuna API credentials not set.")
            print("Set ADZUNA_APP_ID and ADZUNA_API_KEY environment variables")
            print("Or register at: https://developer.adzuna.com/")
            return False
        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _search_jobs(
        self,
        query: str,
        location: Optional[str] = None,
        country: str = "us",
        page: int = 1,
        results_per_page: int = 50,
        salary_min: Optional[int] = None,
        salary_max: Optional[int] = None,
        full_time: bool = True,
    ) -> dict:
        """
        Search for jobs matching query.

        Args:
            query: Search query
            location: Location filter
            country: Country code
            page: Page number
            results_per_page: Results per page (max 50)
            salary_min: Minimum salary filter
            salary_max: Maximum salary filter
            full_time: Filter for full-time jobs only

        Returns:
            API response dictionary
        """
        url = f"{self.BASE_URL}/{country}/search/{page}"

        params = {
            "app_id": self.app_id,
            "app_key": self.api_key,
            "what": query,
            "results_per_page": min(results_per_page, 50),
            "content-type": "application/json",
        }

        if location:
            params["where"] = location

        if salary_min:
            params["salary_min"] = salary_min

        if salary_max:
            params["salary_max"] = salary_max

        if full_time:
            params["full_time"] = 1

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _get_salary_histogram(
        self,
        query: str,
        location: Optional[str] = None,
        country: str = "us",
    ) -> dict:
        """
        Get salary distribution histogram for a job query.

        Args:
            query: Job search query
            location: Location filter
            country: Country code

        Returns:
            Salary histogram data
        """
        url = f"{self.BASE_URL}/{country}/histogram"

        params = {
            "app_id": self.app_id,
            "app_key": self.api_key,
            "what": query,
        }

        if location:
            params["where"] = location

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()

    def fetch_jobs(
        self,
        queries: Optional[list[str]] = None,
        locations: Optional[list[str]] = None,
        max_pages: int = 10,
        max_queries: Optional[int] = None,
        max_locations: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch job postings for AI/ML roles.

        Args:
            queries: Search queries (default: AI_SEARCH_QUERIES)
            locations: Locations to search (default: all LOCATIONS)
            max_pages: Maximum pages to fetch per query (default: 10)
            max_queries: Limit number of queries (default: all queries)
            max_locations: Limit number of locations (default: all locations)

        Returns:
            DataFrame with job posting data
        """
        if not self._check_credentials():
            return self._create_sample_data()

        if queries is None:
            queries = self.AI_SEARCH_QUERIES
            if max_queries:
                queries = queries[:max_queries]
        if locations is None:
            locations = list(self.LOCATIONS.keys())
            if max_locations:
                locations = locations[:max_locations]

        all_jobs = []

        for query in queries:
            for location in locations:
                print(f"Searching: '{query}' in {location}")

                for page in range(1, max_pages + 1):
                    try:
                        result = self._search_jobs(
                            query=query,
                            location=location,
                            page=page,
                        )

                        jobs = result.get("results", [])
                        if not jobs:
                            break

                        for job in jobs:
                            all_jobs.append(self._parse_job(job, query, location))

                        # Check if more pages available
                        total = result.get("count", 0)
                        if page * 50 >= total:
                            break

                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 429:
                            print("Rate limit reached. Stopping.")
                            break
                        print(f"Error: {e}")
                        break

        df = pd.DataFrame(all_jobs)

        if not df.empty:
            # Clean and process
            df = self._process_jobs_data(df)

            # Save to parquet
            output_path = self.data_dir / "adzuna_jobs.parquet"
            df.to_parquet(output_path, index=False)
            print(f"Saved {len(df)} jobs to {output_path}")

        return df

    def _parse_job(self, job: dict, query: str, location: str) -> dict:
        """Parse a job posting into structured format."""
        return {
            "job_id": job.get("id"),
            "title": job.get("title"),
            "description": job.get("description", "")[:500],
            "company": job.get("company", {}).get("display_name"),
            "location": job.get("location", {}).get("display_name"),
            "location_area": ", ".join(job.get("location", {}).get("area", [])),
            "salary_min": job.get("salary_min"),
            "salary_max": job.get("salary_max"),
            "salary_predicted": job.get("salary_is_predicted", False),
            "contract_type": job.get("contract_type"),
            "contract_time": job.get("contract_time"),
            "category": job.get("category", {}).get("label"),
            "created": job.get("created"),
            "redirect_url": job.get("redirect_url"),
            "search_query": query,
            "search_location": location,
        }

    def _process_jobs_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean jobs data."""
        df = df.copy()

        # Calculate average salary where available
        df["salary_avg"] = df[["salary_min", "salary_max"]].mean(axis=1)

        # Parse dates
        df["created_date"] = pd.to_datetime(df["created"], errors="coerce")
        df["year"] = df["created_date"].dt.year
        df["month"] = df["created_date"].dt.month

        # Extract state from location
        df["state"] = df["location_area"].apply(self._extract_state)

        # Flag remote jobs
        df["is_remote"] = (
            df["title"].str.lower().str.contains("remote", na=False) |
            df["location"].str.lower().str.contains("remote", na=False)
        )

        # Extract seniority level
        df["seniority"] = df["title"].apply(self._extract_seniority)

        return df

    def _extract_state(self, location_area: str) -> str:
        """Extract US state from location string."""
        if pd.isna(location_area):
            return "Unknown"

        # Common state patterns
        states = {
            "california": "CA", "ca": "CA",
            "new york": "NY", "ny": "NY",
            "washington": "WA", "wa": "WA",
            "texas": "TX", "tx": "TX",
            "massachusetts": "MA", "ma": "MA",
            "illinois": "IL", "il": "IL",
            "georgia": "GA", "ga": "GA",
            "colorado": "CO", "co": "CO",
            "virginia": "VA", "va": "VA",
            "florida": "FL", "fl": "FL",
            "north carolina": "NC", "nc": "NC",
        }

        location_lower = location_area.lower()
        for state_name, abbrev in states.items():
            if state_name in location_lower:
                return abbrev

        return "Other"

    def _extract_seniority(self, title: str) -> str:
        """Extract seniority level from job title."""
        if pd.isna(title):
            return "Unknown"

        title_lower = title.lower()

        if any(term in title_lower for term in ["senior", "sr.", "sr ", "lead", "principal", "staff"]):
            return "Senior"
        elif any(term in title_lower for term in ["junior", "jr.", "jr ", "entry", "associate"]):
            return "Junior"
        elif any(term in title_lower for term in ["director", "head of", "vp ", "chief"]):
            return "Executive"
        elif any(term in title_lower for term in ["manager", "mgr"]):
            return "Manager"
        else:
            return "Mid-level"

    def fetch_salary_distributions(
        self,
        queries: Optional[list[str]] = None,
        max_queries: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch salary histogram data for different job types.

        Args:
            queries: Job queries to get salary distributions for
            max_queries: Limit number of queries (default: all queries)

        Returns:
            DataFrame with salary distribution data
        """
        if not self._check_credentials():
            return pd.DataFrame()

        if queries is None:
            queries = self.AI_SEARCH_QUERIES
            if max_queries:
                queries = queries[:max_queries]

        all_histograms = []

        for query in queries:
            print(f"Fetching salary histogram for: {query}")

            try:
                result = self._get_salary_histogram(query)
                histogram = result.get("histogram", {})

                for salary_range, count in histogram.items():
                    all_histograms.append({
                        "query": query,
                        "salary_range": salary_range,
                        "job_count": count,
                    })

            except Exception as e:
                print(f"Error fetching histogram for {query}: {e}")

        return pd.DataFrame(all_histograms)

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data when API credentials are not available."""
        print("Creating sample data for demonstration...")

        # Sample job posting data based on typical AI/ML market
        sample_jobs = [
            {"title": "Senior Machine Learning Engineer", "company": "Tech Corp", "location": "San Francisco, CA", "salary_min": 180000, "salary_max": 250000, "seniority": "Senior"},
            {"title": "Data Scientist", "company": "Analytics Inc", "location": "New York, NY", "salary_min": 130000, "salary_max": 170000, "seniority": "Mid-level"},
            {"title": "ML Engineer", "company": "AI Startup", "location": "Seattle, WA", "salary_min": 150000, "salary_max": 200000, "seniority": "Mid-level"},
            {"title": "Junior Data Scientist", "company": "Data Co", "location": "Austin, TX", "salary_min": 90000, "salary_max": 120000, "seniority": "Junior"},
            {"title": "AI Research Scientist", "company": "Research Lab", "location": "Boston, MA", "salary_min": 160000, "salary_max": 220000, "seniority": "Senior"},
            {"title": "MLOps Engineer", "company": "Cloud Inc", "location": "Denver, CO", "salary_min": 140000, "salary_max": 180000, "seniority": "Mid-level"},
            {"title": "NLP Engineer", "company": "Language AI", "location": "San Francisco, CA", "salary_min": 170000, "salary_max": 230000, "seniority": "Mid-level"},
            {"title": "Computer Vision Engineer", "company": "Vision Tech", "location": "Los Angeles, CA", "salary_min": 155000, "salary_max": 210000, "seniority": "Mid-level"},
            {"title": "Lead Data Scientist", "company": "Finance Corp", "location": "Chicago, IL", "salary_min": 175000, "salary_max": 240000, "seniority": "Senior"},
            {"title": "Deep Learning Engineer", "company": "Neural Networks Inc", "location": "Remote", "salary_min": 160000, "salary_max": 220000, "seniority": "Mid-level"},
        ]

        df = pd.DataFrame(sample_jobs)
        df["salary_avg"] = (df["salary_min"] + df["salary_max"]) / 2
        df["is_remote"] = df["location"].str.contains("Remote")
        df["state"] = df["location"].apply(lambda x: x.split(", ")[-1] if ", " in x else "Remote")
        df["year"] = 2024
        df["month"] = 11
        df["search_query"] = "sample"
        df["salary_predicted"] = False

        return df

    def collect(
        self,
        include_histograms: bool = True,
        max_queries: Optional[int] = None,
        max_locations: Optional[int] = None,
        max_pages: int = 10,
    ) -> dict[str, pd.DataFrame]:
        """
        Collect comprehensive job posting data.

        Args:
            include_histograms: Whether to fetch salary histograms
            max_queries: Limit number of queries (default: all 14 queries)
            max_locations: Limit number of locations (default: all 11 locations)
            max_pages: Maximum pages per query (default: 10)

        Returns:
            Dictionary with jobs and histogram DataFrames
        """
        results = {}

        n_queries = len(self.AI_SEARCH_QUERIES) if not max_queries else max_queries
        n_locations = len(self.LOCATIONS) if not max_locations else max_locations
        print(f"Collecting job postings ({n_queries} queries, {n_locations} locations)...")

        jobs_df = self.fetch_jobs(
            max_queries=max_queries,
            max_locations=max_locations,
            max_pages=max_pages,
        )
        results["jobs"] = jobs_df

        if include_histograms and self._check_credentials():
            print("Collecting salary distributions...")
            histogram_df = self.fetch_salary_distributions(max_queries=max_queries)
            results["salary_histograms"] = histogram_df

        return results

    def get_salary_stats_by_role(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate salary statistics by job role.

        Args:
            df: DataFrame with job data

        Returns:
            Summary statistics by role
        """
        if df.empty or "salary_avg" not in df.columns:
            return pd.DataFrame()

        # Filter valid salaries
        valid_df = df[df["salary_avg"].notna() & (df["salary_avg"] > 30000)]

        stats = valid_df.groupby("search_query").agg({
            "salary_avg": ["mean", "median", "std", "min", "max", "count"],
            "salary_min": "mean",
            "salary_max": "mean",
        }).round(0)

        stats.columns = ["_".join(col).strip() for col in stats.columns.values]

        return stats.reset_index()


if __name__ == "__main__":
    # Example usage
    collector = AdzunaJobsCollector()
    results = collector.collect()

    for key, df in results.items():
        print(f"\n{key}: {len(df)} records")
        if not df.empty:
            print(df.head())
