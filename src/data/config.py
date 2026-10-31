"""
Climate Dataset Configuration Module
------------------------------------
Centralized configuration for IMD (India Meteorological Department)
climate datasets including rainfall, maximum temperature, and minimum temperature.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# Project Base Directory Setup
# Computes absolute path to the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Spatial Bounding Box for the Indian Subcontinent Region
INDIA_BOUNDS = {
    "min_lat": 6.5,
    "max_lat": 38.5,
    "min_lon": 68.5,
    "max_lon": 97.5,
}

@dataclass
class DatasetConfig:
    """
    Stores configuration metadata for a single climate dataset.
    """
    name: str
    parameter: str
    spatial_resolution: str
    source_organization: str
    source_url: str
    expected_format: str
    raw_data_dir: Path


# --- Central Dataset Configurations ---

IMD_RAINFALL = DatasetConfig(
    name="IMD High-Resolution Gridded Rainfall",
    parameter="Precipitation / Daily Rainfall (mm/day)",
    spatial_resolution="0.25° x 0.25° lat/lon",
    source_organization="India Meteorological Department (IMD) / CMPG",
    source_url="https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html",
    expected_format="NetCDF (.nc) or Binary (.grd)",
    raw_data_dir=RAW_DATA_DIR / "rainfall"
)

IMD_MAX_TEMP = DatasetConfig(
    name="IMD High-Resolution Maximum Temperature",
    parameter="Maximum Surface Temperature - Tmax (°C)",
    spatial_resolution="1.0° x 1.0° lat/lon",
    source_organization="India Meteorological Department (IMD) / CMPG",
    source_url="https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html",
    expected_format="NetCDF (.nc) or Binary (.grd)",
    raw_data_dir=RAW_DATA_DIR / "max_temp"
)

IMD_MIN_TEMP = DatasetConfig(
    name="IMD High-Resolution Minimum Temperature",
    parameter="Minimum Surface Temperature - Tmin (°C)",
    spatial_resolution="1.0° x 1.0° lat/lon",
    source_organization="India Meteorological Department (IMD) / CMPG",
    source_url="https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html",
    expected_format="NetCDF (.nc) or Binary (.grd)",
    raw_data_dir=RAW_DATA_DIR / "min_temp"
)


# Registry mapping dataset keys to their configuration objects
DATASETS: Dict[str, DatasetConfig] = {
    "rainfall": IMD_RAINFALL,
    "max_temp": IMD_MAX_TEMP,
    "min_temp": IMD_MIN_TEMP,
}


def get_dataset_config(dataset_key: str) -> DatasetConfig:
    """
    Retrieve dataset configuration by dataset key name.

    Parameters:
        dataset_key (str): Name of dataset ('rainfall', 'max_temp', 'min_temp')

    Returns:
        DatasetConfig: The matching dataset configuration object.
    """
    key = dataset_key.lower().strip()
    if key not in DATASETS:
        available_keys = ", ".join(DATASETS.keys())
        raise KeyError(f"Dataset '{dataset_key}' not found. Available datasets: {available_keys}")
    return DATASETS[key]


if __name__ == "__main__":
    # Demonstration: Print all dataset configurations
    print("=" * 60)
    print("    IMD Climate Datasets Configuration Registry")
    print("=" * 60)
    for key, config in DATASETS.items():
        print(f"\n[Key: '{key}'] {config.name}")
        print(f"  Parameter:           {config.parameter}")
        print(f"  Spatial Resolution:  {config.spatial_resolution}")
        print(f"  Source Organization: {config.source_organization}")
        print(f"  Source URL:          {config.source_url}")
        print(f"  Expected Format:     {config.expected_format}")
        print(f"  Local Raw Directory: {config.raw_data_dir}")
