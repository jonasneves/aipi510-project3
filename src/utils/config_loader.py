"""
Configuration Loader Utility

Centralized loading of YAML configuration files.
"""

from pathlib import Path
from typing import Any, Union

import yaml


class ConfigLoader:
    """Load and cache configuration from YAML files."""

    _cache: dict[str, dict] = {}
    _config_dir: Path = Path(__file__).parent.parent.parent / "configs"

    @classmethod
    def load(cls, config_name: str) -> dict[str, Any]:
        """
        Load a configuration file by name.

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Dictionary containing configuration

        Example:
            patterns = ConfigLoader.load("patterns")
            skills = patterns["skill_patterns"]
        """
        if config_name in cls._cache:
            return cls._cache[config_name]

        config_path = cls._config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        cls._cache[config_name] = config
        return config

    @classmethod
    def get_patterns(cls) -> dict[str, Any]:
        """Load resume parsing patterns configuration."""
        return cls.load("patterns")

    @classmethod
    def get_features(cls) -> dict[str, Any]:
        """Load feature engineering configuration."""
        return cls.load("features")

    @classmethod
    def get_data_sources(cls) -> dict[str, Any]:
        """Load data sources configuration."""
        return cls.load("data_sources")

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the configuration cache."""
        cls._cache = {}

    @classmethod
    def set_config_dir(cls, path: Union[str, Path]) -> None:
        """Set custom configuration directory."""
        cls._config_dir = Path(path)
        cls.clear_cache()


# Convenience functions for common operations
def get_skill_categories() -> dict[str, list[str]]:
    """Get skill category definitions."""
    return ConfigLoader.get_features()["skill_categories"]


def get_company_tiers() -> dict[str, list[str]]:
    """Get company tier definitions."""
    return ConfigLoader.get_features()["company_tiers"]


def get_location_multipliers() -> dict[str, float]:
    """Get location cost-of-living multipliers."""
    return ConfigLoader.get_features()["location_multipliers"]


def get_job_title_patterns() -> list[dict]:
    """Get job title extraction patterns."""
    return ConfigLoader.get_patterns()["job_title_patterns"]


def get_skill_patterns() -> dict[str, str]:
    """Get skill extraction patterns."""
    return ConfigLoader.get_patterns()["skill_patterns"]


def get_state_patterns() -> dict[str, str]:
    """Get state extraction patterns."""
    return ConfigLoader.get_patterns()["state_patterns"]


def get_encodings() -> dict[str, dict]:
    """Get encoding mappings for prediction."""
    return ConfigLoader.get_features()["encodings"]


if __name__ == "__main__":
    # Test loading configurations
    print("Testing configuration loader...\n")

    # Load patterns
    patterns = ConfigLoader.get_patterns()
    print(f"Loaded {len(patterns['job_title_patterns'])} job title patterns")
    print(f"Loaded {len(patterns['skill_patterns'])} skill patterns")
    print(f"Loaded {len(patterns['state_patterns'])} state patterns")

    # Load features
    features = ConfigLoader.get_features()
    print(f"\nLoaded {len(features['skill_categories'])} skill categories")
    print(f"Loaded {len(features['company_tiers'])} company tiers")
    print(f"Loaded {len(features['location_multipliers'])} location multipliers")

    # Load data sources
    sources = ConfigLoader.get_data_sources()
    print(f"\nLoaded {len(sources['sources'])} data sources")

    print("\nConfiguration loader test passed!")
