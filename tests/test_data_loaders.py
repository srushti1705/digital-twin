"""
Unit tests for src.data.loaders module
---------------------------------------
Verifies file existence checks, NetCDF loading via xarray, CSV loading via pandas,
and format auto-detection.
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.loaders import load_climate_data, load_csv, load_netcdf


def test_missing_file_raises_error():
    """Verify FileNotFoundError when target file does not exist."""
    missing_path = Path("data/raw/non_existent_file.nc")
    with pytest.raises(FileNotFoundError):
        load_netcdf(missing_path)

    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file.csv")


def test_load_csv_valid(tmp_path):
    """Verify successful CSV dataset loading."""
    csv_file = tmp_path / "test_climate.csv"
    df_sample = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5),
        "rainfall_mm": [0.0, 12.5, 4.2, 0.0, 25.1],
        "temperature_c": [28.5, 27.2, 26.8, 29.1, 24.5]
    })
    df_sample.to_csv(csv_file, index=False)

    df_loaded = load_csv(csv_file, parse_dates=["date"])
    assert isinstance(df_loaded, pd.DataFrame)
    assert len(df_loaded) == 5
    assert "rainfall_mm" in df_loaded.columns


def test_load_netcdf_valid(tmp_path):
    """Verify successful NetCDF dataset loading using xarray."""
    nc_file = tmp_path / "test_climate.nc"
    
    # Create sample NetCDF dataset
    times = pd.date_range("2023-01-01", periods=3)
    lats = [10.0, 20.0]
    lons = [70.0, 80.0]
    data = np.random.rand(len(times), len(lats), len(lons))

    ds_sample = xr.Dataset(
        data_vars={"rf": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons}
    )
    ds_sample.to_netcdf(nc_file)

    ds_loaded = load_netcdf(nc_file)
    assert isinstance(ds_loaded, xr.Dataset)
    assert "rf" in ds_loaded.data_vars
    assert len(ds_loaded.coords["time"]) == 3


def test_load_climate_data_autodetect(tmp_path):
    """Verify format auto-detection in load_climate_data."""
    csv_file = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv_file, index=False)

    df = load_climate_data(csv_file)
    assert isinstance(df, pd.DataFrame)
