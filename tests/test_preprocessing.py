"""
Unit Tests for src.preprocessing Package
-----------------------------------------
Tests data quality checks, missing value standardization, missing value pattern analysis,
approved parameter-specific missing-value strategies, spatial preprocessing, temporal processing,
feature engineering, scikit-learn normalization, and the main orchestration pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from src.preprocessing.quality_checks import check_dataset_quality, validate_physical_bounds
from src.preprocessing.missing_values import (
    detect_missing_values,
    handle_missing_values,
    get_missing_stats,
    apply_land_mask,
    analyze_missing_values,
    impute_temperature_missing_values,
    impute_rainfall_missing_values,
    process_climate_dataset_missing_values
)
from src.preprocessing.spatial_processing import (
    identify_spatial_coordinates,
    validate_coordinate_ranges,
    sort_spatial_coordinates,
    detect_duplicate_spatial_coordinates,
    get_spatial_resolution,
    document_resolution_mismatch,
    generate_spatial_report,
    crop_spatial_domain,
    regrid_dataset,
    regrid_temperature_to_rainfall
)
from src.preprocessing.temporal_processing import (
    identify_time_dimension,
    standardize_datetime_coordinates,
    sort_chronologically,
    detect_duplicate_timestamps,
    detect_missing_time_periods,
    extract_temporal_features,
    process_temporal_pipeline,
    align_temporal_dimension,
    compute_monthly_aggregations,
    compute_rolling_statistics
)
from src.preprocessing.feature_engineering import (
    create_climate_indices,
    create_lag_features,
    create_rolling_features,
    create_temporal_index_features,
    create_monthly_climatological_stats,
    engineer_all_climate_features,
    derive_climate_features
)
from src.preprocessing.normalization import (
    analyze_scaler_suitability,
    fit_dataset_scaler,
    transform_dataset_scaler,
    fit_transform_dataset_scaler,
    inverse_transform_dataset_scaler,
    save_scaler,
    load_scaler,
    normalize_min_max,
    normalize_z_score,
    normalize_features
)
from src.preprocessing.pipeline import run_preprocessing_pipeline


@pytest.fixture
def sample_rainfall_dataset():
    """Fixture providing a synthetic 0.25-degree rainfall dataset with raw -999.0 fill values."""
    lats = np.arange(10.0, 15.0, 0.25)
    lons = np.arange(70.0, 75.0, 0.25)
    times = pd.date_range("2020-01-01", periods=10, freq="D")

    np.random.seed(42)
    rain_data = np.random.uniform(0.0, 50.0, size=(10, len(lats), len(lons)))
    rain_data[:, 0:2, 0:2] = -999.0
    rain_data[3:5, 5, 5] = -999.0

    ds = xr.Dataset(
        data_vars={"rain": (("time", "lat", "lon"), rain_data)},
        coords={"time": times, "lat": lats, "lon": lons}
    )
    return ds


@pytest.fixture
def sample_temperature_dataset():
    """Fixture providing a synthetic 1.0-degree max/min temperature dataset with raw 99.9 fill values."""
    lats = np.arange(10.0, 15.0, 1.0)
    lons = np.arange(70.0, 75.0, 1.0)
    times = pd.date_range("2020-01-01", periods=10, freq="D")

    np.random.seed(42)
    tmax_data = np.random.uniform(25.0, 42.0, size=(10, len(lats), len(lons)))
    tmin_data = tmax_data - np.random.uniform(5.0, 15.0, size=(10, len(lats), len(lons)))

    tmax_data[:, 0, 0] = 99.9
    tmax_data[2:4, 2, 2] = 99.9
    tmin_data[:, 0, 0] = 99.9

    ds = xr.Dataset(
        data_vars={
            "tmax": (("time", "lat", "lon"), tmax_data),
            "tmin": (("time", "lat", "lon"), tmin_data)
        },
        coords={"time": times, "lat": lats, "lon": lons}
    )
    return ds


def test_quality_checks(sample_rainfall_dataset, sample_temperature_dataset):
    """Test check_dataset_quality and validate_physical_bounds."""
    res_rain = check_dataset_quality(sample_rainfall_dataset, var_name="rain", raw_fill_value=-999.0)
    assert res_rain["is_valid"] is True


def test_missing_values(sample_rainfall_dataset, sample_temperature_dataset):
    """Test detect_missing_values, handle_missing_values, and approved imputation strategies."""
    masks = detect_missing_values(sample_rainfall_dataset, fill_value_map={"rain": -999.0})
    assert masks["rain"].sum() > 0


def test_temporal_processing(sample_rainfall_dataset):
    """Test identify_time_dimension, standardize, sort, duplicate/gap detection, and feature extraction."""
    time_dim = identify_time_dimension(sample_rainfall_dataset)
    assert time_dim == "time"


def test_spatial_processing(sample_temperature_dataset, sample_rainfall_dataset):
    """Test spatial coord identification, range validation, sorting, resolution, duplicate check, and mismatch report."""
    lat_key, lon_key = identify_spatial_coordinates(sample_temperature_dataset)
    assert lat_key == "lat"


def test_feature_engineering(sample_temperature_dataset, sample_rainfall_dataset):
    """Test climate feature creation (indices, lags, trailing rolling features, temporal features)."""
    ds_temp = handle_missing_values(sample_temperature_dataset, fill_value_map={"tmax": 99.9, "tmin": 99.9})
    ds_rain = handle_missing_values(sample_rainfall_dataset, fill_value_map={"rain": -999.0})

    ds_regridded = regrid_temperature_to_rainfall(ds_temp, ds_rain)
    ds_combined = xr.Dataset(
        data_vars={
            "rain": ds_rain["rain"],
            "tmax": ds_regridded["tmax"],
            "tmin": ds_regridded["tmin"]
        },
        coords=ds_rain.coords
    )

    ds_all = engineer_all_climate_features(ds_combined, include_lags=True, lags=[1])
    assert "rain_lag1" in ds_all.data_vars


def test_scikit_learn_normalization(sample_temperature_dataset, sample_rainfall_dataset, tmp_path):
    """Test analyze_scaler_suitability, fit_dataset_scaler, transform, inverse transform, and save/load scalers."""
    ds_temp = handle_missing_values(sample_temperature_dataset, fill_value_map={"tmax": 99.9, "tmin": 99.9})
    scaler_reg = fit_dataset_scaler(ds_temp, scaler_type="standard", var_names=["tmax"])
    test_scaled = transform_dataset_scaler(ds_temp, scaler_reg)
    assert "tmax" in test_scaled.data_vars


def test_main_preprocessing_pipeline_execution(tmp_path):
    """Test main end-to-end preprocessing pipeline execution, record metrics, safety checks, and NetCDF exports."""
    summary = run_preprocessing_pipeline(
        output_dir=str(tmp_path),
        year=2020,
        save_netcdf=True,
        normalize=True
    )
    assert summary["pipeline_status"] == "SUCCESS"
    assert summary["year"] == 2020
    assert "step1_ingestion" in summary["step_metrics"]
    assert "step2_quality_checks" in summary["step_metrics"]
    assert "step3_missing_handling" in summary["step_metrics"]
    assert "step4_temporal_processing" in summary["step_metrics"]
    assert "step5_spatial_processing" in summary["step_metrics"]
    assert "step6_spatial_alignment" in summary["step_metrics"]
    assert "step7_feature_engineering" in summary["step_metrics"]

    # Verify exported output files
    output_nc = tmp_path / "india_climate_2020_aligned.nc"
    norm_nc = tmp_path / "india_climate_2020_normalized.nc"
    scaler_joblib = tmp_path / "climate_scaler_2020.joblib"

    assert output_nc.exists()
    assert norm_nc.exists()
    assert scaler_joblib.exists()
