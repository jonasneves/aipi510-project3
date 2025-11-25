"""
BLS Data Collector

Fetches national wage benchmarks for AI/ML occupations.
API: https://www.bls.gov/developers/
"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Implemented BLS API integration for occupational wage benchmarks with SOC code mapping
## and time series data collection for AI/ML related occupations.

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


class BLSDataCollector:
    """Collector for BLS national wage data."""

    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    # Key AI/ML occupations with their series IDs for national mean wage
    # Format: OEUN + area(0000000) + industry(000000) + SOC + datatype(04=mean wage)
    OCCUPATION_SERIES = {
        "OEUN0000000000000015122104": {
            "soc_code": "15-1221",
            "title": "Computer and Information Research Scientists",
        },
        "OEUN0000000000000015205104": {
            "soc_code": "15-2051",
            "title": "Data Scientists",
        },
        "OEUN0000000000000015125204": {
            "soc_code": "15-1252",
            "title": "Software Developers",
        },
        "OEUN0000000000000015204104": {
            "soc_code": "15-2041",
            "title": "Statisticians",
        },
        "OEUN0000000000000015203104": {
            "soc_code": "15-2031",
            "title": "Operations Research Analysts",
        },
        "OEUN0000000000000015121104": {
            "soc_code": "15-1211",
            "title": "Computer Systems Analysts",
        },
        "OEUN0000000000000015124304": {
            "soc_code": "15-1243",
            "title": "Database Architects",
        },
    }

    def __init__(self, api_key: Optional[str] = None, data_dir: str = "data/raw"):
        """Initialize BLS collector."""
        self.api_key = api_key or os.getenv("BLS_API_KEY")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect(self, start_year: int = 2022, end_year: int = 2024) -> pd.DataFrame:
        """
        Fetch national wage data for AI/ML occupations.

        Args:
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with occupation wage benchmarks
        """
        series_ids = list(self.OCCUPATION_SERIES.keys())

        payload = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(end_year),
            "annualaverage": True,
        }

        if self.api_key:
            payload["registrationkey"] = self.api_key

        try:
            response = requests.post(self.BASE_URL, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "REQUEST_SUCCEEDED":
                print(f"BLS API error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()

        except Exception as e:
            print(f"BLS API request failed: {e}")
            return pd.DataFrame()

        # Parse response
        records = []
        for series in data.get("Results", {}).get("series", []):
            series_id = series.get("seriesID", "")
            occ_info = self.OCCUPATION_SERIES.get(series_id, {})

            for item in series.get("data", []):
                # Only get annual averages
                if item.get("period") != "A01":
                    continue

                try:
                    wage = float(item.get("value", "0").replace(",", ""))
                except ValueError:
                    continue

                records.append({
                    "series_id": series_id,
                    "area_code": "0000000",
                    "area_name": "National",
                    "soc_code": occ_info.get("soc_code", ""),
                    "occupation": occ_info.get("title", ""),
                    "year": int(item.get("year", 0)),
                    "mean_annual_wage": wage,
                    "data_source": "BLS_OEWS",
                })

        df = pd.DataFrame(records)

        if not df.empty:
            # Keep only most recent year per occupation
            df = df.sort_values("year", ascending=False)
            df = df.drop_duplicates(subset=["soc_code"], keep="first")

            output_path = self.data_dir / "bls_wage_data.parquet"
            df.to_parquet(output_path, index=False)
            print(f"Saved {len(df)} records to {output_path}")

        return df

    def get_benchmark(self, soc_code: str) -> Optional[float]:
        """Get BLS benchmark salary for an occupation."""
        filepath = self.data_dir / "bls_wage_data.parquet"
        if not filepath.exists():
            return None

        df = pd.read_parquet(filepath)
        match = df[df["soc_code"] == soc_code]
        if not match.empty:
            return match.iloc[0]["mean_annual_wage"]
        return None


if __name__ == "__main__":
    collector = BLSDataCollector()
    df = collector.collect()
    print(df)
