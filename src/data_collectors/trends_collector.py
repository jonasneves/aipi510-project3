"""
Google Trends Collector

Fetches regional interest and emerging terms for AI/ML jobs.
"""

import time
from pathlib import Path

import pandas as pd


class GoogleTrendsCollector:
    """Collector for Google Trends data."""

    SEARCH_TERMS = [
        "machine learning engineer salary",
        "data scientist salary",
    ]

    SEED_KEYWORDS = [
        "AI engineer",
        "machine learning jobs",
        "LLM developer",
    ]

    def __init__(self, data_dir: str = "data/raw"):
        """Initialize collector."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._pytrends = None

    def _get_pytrends(self):
        """Initialize pytrends client."""
        if self._pytrends is None:
            try:
                from pytrends.request import TrendReq
                self._pytrends = TrendReq(hl="en-US", tz=360)
            except ImportError:
                raise ImportError("pytrends required: pip install pytrends")
            except TypeError as e:
                print(f"pytrends init error: {e}")
                print("Fix: pip install 'urllib3<2.0'")
                raise
        return self._pytrends

    def _is_available(self) -> bool:
        """Check if Google Trends API is available."""
        try:
            self._get_pytrends()
            return True
        except Exception:
            return False

    def fetch_regional_interest(self) -> pd.DataFrame:
        """Fetch regional interest data (US states)."""
        try:
            pytrends = self._get_pytrends()
            pytrends.build_payload(
                self.SEARCH_TERMS,
                timeframe="today 12-m",
                geo="US",
            )
            df = pytrends.interest_by_region(
                resolution="REGION",
                inc_low_vol=True,
                inc_geo_code=True,
            )
            time.sleep(2)

            if not df.empty:
                df = df.reset_index()
                df = df.melt(
                    id_vars=["geoName", "geoCode"],
                    var_name="search_term",
                    value_name="interest",
                )
            return df

        except Exception as e:
            print(f"Error fetching regional data: {e}")
            return pd.DataFrame()

    def fetch_emerging_terms(self) -> pd.DataFrame:
        """Find emerging/rising AI job search terms."""
        all_rising = []

        for keyword in self.SEED_KEYWORDS:
            print(f"Finding emerging terms for: {keyword}")
            try:
                pytrends = self._get_pytrends()
                pytrends.build_payload([keyword], timeframe="today 12-m", geo="US")
                related = pytrends.related_queries()
                time.sleep(2)

                if keyword in related and related[keyword]["rising"] is not None:
                    rising_df = related[keyword]["rising"]
                    rising_df["seed_keyword"] = keyword
                    all_rising.append(rising_df)

            except Exception as e:
                print(f"Error for '{keyword}': {e}")

        if not all_rising:
            return pd.DataFrame()

        return pd.concat(all_rising, ignore_index=True)

    def collect(self) -> dict[str, pd.DataFrame]:
        """Collect Google Trends data."""
        results = {}

        if not self._is_available():
            print("Google Trends unavailable. Fix: pip install 'urllib3<2.0'")
            return results

        print("Fetching regional interest...")
        regional_df = self.fetch_regional_interest()
        if not regional_df.empty:
            results["regional_interest"] = regional_df
            output_path = self.data_dir / "google_trends_regional.parquet"
            regional_df.to_parquet(output_path, index=False)
            print(f"Saved {len(regional_df)} regional records")

        print("Finding emerging terms...")
        emerging_df = self.fetch_emerging_terms()
        if not emerging_df.empty:
            results["emerging_terms"] = emerging_df
            output_path = self.data_dir / "google_trends_emerging.parquet"
            emerging_df.to_parquet(output_path, index=False)
            print(f"Saved {len(emerging_df)} emerging terms")

        return results


if __name__ == "__main__":
    collector = GoogleTrendsCollector()
    results = collector.collect()
    for key, df in results.items():
        print(f"{key}: {len(df)} records")
