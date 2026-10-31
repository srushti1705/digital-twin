"""
Unit tests for src.data.validate_dataset module
------------------------------------------------
Verifies non-destructive dataset validation and diagnostic reporting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.validate_dataset import validate_climate_dataset


def test_validate_missing_file_error():
    """Verify FileNotFoundError when dataset does not exist."""
    with pytest.raises(FileNotFoundError):
        validate_climate_dataset("non_existent_climate_file.nc", verbose=False)


def test_validate_climate_dataset_structure(tmp_path):
    """Verify validation report dictionary generation for sample NetCDF dataset."""
    nc_file = tmp_path / "sample_rain.nc"

    times = pd.date_range("2023-06-01", periods=10)
    lats = np.arange(10.0, 15.0, 1.0)
    lons = np.arange(70.0, 75.0, 1.0)

    # 10 x 5 x 5 tensor
    rain_data = np.ones((len(times), len(lats), len(lons)), dtype=np.float32) * 5.0
    rain_data[0, 0, 0] = np.nan  # Introduce 1 missing value

    ds = xr.Dataset(
        data_vars={"rf": (["time", "lat", "lon"], rain_data, {"units": "mm/day"})},
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={"title": "Test IMD Rainfall NetCDF"}
    )
    ds.to_netcdf(nc_file)

    report = validate_climate_dataset(nc_file, verbose=True)

    assert report["file_size_mb"] > 0
    assert "time" in report["dimensions"]
    assert report["dimensions"]["time"] == 10
    assert report["spatial"]["latitude"]["min_lat"] == 10.0
    assert report["spatial"]["latitude"]["max_lat"] == 14.0
    assert report["spatial"]["longitude"]["min_lon"] == 70.0
    assert report["spatial"]["longitude"]["max_lon"] == 74.0
    assert report["temporal"]["total_timesteps"] == 10

    # Variable statistics checks
    rf_stats = report["variables"]["rf"]
    assert rf_stats["dtype"] == "float32"
    assert rf_stats["shape"] == [10, 5, 5]
    assert rf_stats["total_elements"] == 250
    assert rf_stats["missing_elements"] == 1
    assert rf_stats["min"] == 5.0
    assert rf_stats["max"] == 5.0
    assert rf_stats["mean"] == 5.0
