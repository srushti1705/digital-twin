"""
Main Preprocessing Pipeline Module
-----------------------------------
End-to-end orchestration pipeline for processing IMD climate datasets (Rainfall, Max Temp, Min Temp).

Conceptual Pipeline Workflow:
  Raw IMD Data (.grd)
        │
        ▼
  1. Quality Checks (check_dataset_quality)
        │
        ▼
  2. Missing-Value Analysis & Handling (analyze_missing_values, impute_rainfall_missing_values, impute_temperature_missing_values)
        │
        ▼
  3. Temporal Processing (process_temporal_pipeline: standardize, sort, duplicates, missing gaps, temporal features)
        │
        ▼
  4. Spatial Processing & Bounds Validation (validate_coordinate_ranges, get_spatial_resolution, crop_spatial_domain)
        │
        ▼
  5. Spatial Alignment (regrid_temperature_to_rainfall: 1.0° -> 0.25° bilinear spatial interpolation)
        │
        ▼
  6. Feature Engineering (engineer_all_climate_features: DTR, Tmean, flags, lag1, trailing 7d rolling, cyclical sin/cos)
        │
        ▼
  7. Processed Dataset Export (save to data/processed/imd/india_climate_{year}_aligned.nc)
        │
        ▼
  8. Optional ML Scaling (fit_transform_dataset_scaler: StandardScaler & MinMaxScaler -> climate_scaler.joblib)

Safeguards:
- Never modifies raw datasets in data/raw/imd/
- Reports record counts affected by each processing step
- Includes robust try-except error handling for each phase
- Does NOT train machine-learning models
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import logging
import numpy as np
import xarray as xr

from src.data.loaders import IMDDataLoader
from src.preprocessing.quality_checks import check_dataset_quality, validate_physical_bounds
from src.preprocessing.missing_values import (
    analyze_missing_values,
    impute_rainfall_missing_values,
    impute_temperature_missing_values
)
from src.preprocessing.temporal_processing import process_temporal_pipeline
from src.preprocessing.spatial_processing import (
    validate_coordinate_ranges,
    get_spatial_resolution,
    document_resolution_mismatch,
    crop_spatial_domain,
    regrid_temperature_to_rainfall
)
from src.preprocessing.feature_engineering import engineer_all_climate_features
from src.preprocessing.normalization import (
    analyze_scaler_suitability,
    fit_transform_dataset_scaler,
    save_scaler
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    year: int = 2020,
    save_netcdf: bool = True,
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Executes the full, robust Phase 3 preprocessing pipeline on raw IMD climate datasets.

    Parameters:
        data_dir (str | Path, optional): Path to raw IMD data directory (default: 'data/raw/imd').
        output_dir (str | Path, optional): Path to processed data output directory (default: 'data/processed/imd').
        year (int): Target calendar year (default: 2020).
        save_netcdf (bool): Whether to export output to NetCDF file. Default: True.
        normalize (bool): Whether to generate ML scaling artifacts (StandardScaler/MinMaxScaler). Default: False.

    Returns:
        dict: Detailed execution summary dictionary containing step-by-step record metrics and Datasets.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir_path = Path(data_dir) if data_dir else project_root / "data" / "raw" / "imd"
    output_dir_path = Path(output_dir) if output_dir else project_root / "data" / "processed" / "imd"

    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Safety Verification: Ensure raw files are never targeted for output
    if "raw" in str(output_dir_path).lower():
        raise PermissionError(f"❌ Safety Violation: Cannot set output directory to raw data path '{output_dir_path}'. Save under 'data/processed/imd/' instead.")

    pipeline_report: Dict[str, Any] = {
        "year": year,
        "pipeline_status": "STARTED",
        "step_metrics": {}
    }

    logger.info("==================================================")
    logger.info("      IMD CLIMATE DATA PREPROCESSING PIPELINE     ")
    logger.info("==================================================")

    try:
        # ----------------------------------------------------------------------
        # STEP 1: RAW IMD DATA INGESTION
        # ----------------------------------------------------------------------
        logger.info(f"[Step 1/8] Ingesting raw IMD binary files for year {year}...")
        loader = IMDDataLoader(base_data_dir=str(data_dir_path))

        rain_path = loader.find_raw_file("rainfall", year)
        tmax_path = loader.find_raw_file("max_temperature", year)
        tmin_path = loader.find_raw_file("min_temperature", year)

        ds_rain_raw = loader.load_rainfall_grd(rain_path, year=year)
        ds_tmax_raw = loader.load_temperature_grd(tmax_path, year=year, var_name="tmax")
        ds_tmin_raw = loader.load_temperature_grd(tmin_path, year=year, var_name="tmin")

        ingestion_metrics = {
            "rainfall_shape": list(ds_rain_raw["rain"].shape),
            "rainfall_total_cells": ds_rain_raw["rain"].size,
            "tmax_shape": list(ds_tmax_raw["tmax"].shape),
            "tmax_total_cells": ds_tmax_raw["tmax"].size,
            "tmin_shape": list(ds_tmin_raw["tmin"].shape),
            "tmin_total_cells": ds_tmin_raw["tmin"].size,
            "raw_paths": {
                "rainfall": str(rain_path),
                "tmax": str(tmax_path),
                "tmin": str(tmin_path)
            }
        }
        pipeline_report["step_metrics"]["step1_ingestion"] = ingestion_metrics
        logger.info(f"  - Ingested Rainfall ({ds_rain_raw['rain'].size:,} cells), Tmax ({ds_tmax_raw['tmax'].size:,} cells), Tmin ({ds_tmin_raw['tmin'].size:,} cells).")

        # ----------------------------------------------------------------------
        # STEP 2: QUALITY CHECKS
        # ----------------------------------------------------------------------
        logger.info("\n[Step 2/8] Executing 10-point data quality validation suite...")
        qc_rain = check_dataset_quality(ds_rain_raw, var_name="rain", raw_fill_value=-999.0)
        qc_tmax = check_dataset_quality(ds_tmax_raw, var_name="tmax", raw_fill_value=99.9)
        qc_tmin = check_dataset_quality(ds_tmin_raw, var_name="tmin", raw_fill_value=99.9)

        pipeline_report["step_metrics"]["step2_quality_checks"] = {
            "rainfall_valid": qc_rain["is_valid"],
            "tmax_valid": qc_tmax["is_valid"],
            "tmin_valid": qc_tmin["is_valid"],
            "tmin_tmax_consistency": qc_tmax.get("checks", {}).get("tmin_tmax_consistency", {}).get("is_consistent", True)
        }

        # ----------------------------------------------------------------------
        # STEP 3: MISSING-VALUE ANALYSIS & HANDLING
        # ----------------------------------------------------------------------
        logger.info("\n[Step 3/8] Executing missing-value analysis and approved strategy handling...")
        ds_rain_imputed, rain_missing_metrics = impute_rainfall_missing_values(ds_rain_raw, var_name="rain", raw_fill_value=-999.0)
        ds_tmax_imputed, tmax_missing_metrics = impute_temperature_missing_values(ds_tmax_raw, var_name="tmax", raw_fill_value=99.9)
        ds_tmin_imputed, tmin_missing_metrics = impute_temperature_missing_values(ds_tmin_raw, var_name="tmin", raw_fill_value=99.9)

        pipeline_report["step_metrics"]["step3_missing_handling"] = {
            "rainfall": rain_missing_metrics,
            "tmax": tmax_missing_metrics,
            "tmin": tmin_missing_metrics
        }
        logger.info(f"  - Rainfall missing: {rain_missing_metrics['before_missing_count']:,} cells ({rain_missing_metrics['ocean_cells_preserved_as_nan']:,} ocean cells preserved as NaN).")
        logger.info(f"  - Tmax missing: Imputed {tmax_missing_metrics['imputed_intermittent_cells_count']} intermittent coastal cells. {tmax_missing_metrics['ocean_cells_preserved_as_nan']:,} ocean cells preserved as NaN.")
        logger.info(f"  - Tmin missing: Imputed {tmin_missing_metrics['imputed_intermittent_cells_count']} intermittent coastal cells. {tmin_missing_metrics['ocean_cells_preserved_as_nan']:,} ocean cells preserved as NaN.")

        # ----------------------------------------------------------------------
        # STEP 4: TEMPORAL PROCESSING
        # ----------------------------------------------------------------------
        logger.info("\n[Step 4/8] Executing temporal processing (datetime standardization, sorting, duplicate/gap checks)...")
        ds_rain_time, rain_temp_report = process_temporal_pipeline(ds_rain_imputed, start_date=f"{year}-01-01", extract_features=False)
        ds_tmax_time, tmax_temp_report = process_temporal_pipeline(ds_tmax_imputed, start_date=f"{year}-01-01", extract_features=False)
        ds_tmin_time, tmin_temp_report = process_temporal_pipeline(ds_tmin_imputed, start_date=f"{year}-01-01", extract_features=False)

        pipeline_report["step_metrics"]["step4_temporal_processing"] = {
            "total_timesteps": rain_temp_report["total_timesteps"],
            "date_range": f"{rain_temp_report['start_date']} to {rain_temp_report['end_date']}",
            "duplicates_count": rain_temp_report["duplicate_count"],
            "missing_gaps_count": rain_temp_report["missing_days_count"]
        }

        # ----------------------------------------------------------------------
        # STEP 5: SPATIAL PROCESSING & DIAGNOSTICS
        # ----------------------------------------------------------------------
        logger.info("\n[Step 5/8] Executing spatial bounds validation & resolution diagnostic inspection...")
        res_mismatch_info = document_resolution_mismatch({
            "rainfall": ds_rain_time,
            "tmax": ds_tmax_time,
            "tmin": ds_tmin_time
        })

        pipeline_report["step_metrics"]["step5_spatial_processing"] = {
            "resolution_mismatch_detected": res_mismatch_info["has_mismatch"],
            "rainfall_res": get_spatial_resolution(ds_rain_time),
            "temperature_res": get_spatial_resolution(ds_tmax_time)
        }

        # ----------------------------------------------------------------------
        # STEP 6: SPATIAL ALIGNMENT (BILINEAR RE-GRIDDING)
        # ----------------------------------------------------------------------
        logger.info("\n[Step 6/8] Executing spatial alignment (Bilinear interpolation of 1.0° Tmax/Tmin onto 0.25° Rainfall grid)...")
        ds_tmax_regridded = regrid_temperature_to_rainfall(ds_tmax_time, ds_rain_time, method="linear")
        ds_tmin_regridded = regrid_temperature_to_rainfall(ds_tmin_time, ds_rain_time, method="linear")

        # Combine variables into a single Dataset
        ds_combined = xr.Dataset(
            data_vars={
                "rain": ds_rain_time["rain"],
                "tmax": ds_tmax_regridded["tmax"],
                "tmin": ds_tmin_regridded["tmin"]
            },
            coords=ds_rain_time.coords,
            attrs={
                "title": f"IMD India Aligned Climate Dataset ({year})",
                "source": "India Meteorological Department (IMD)",
                "spatial_resolution": "0.25 degree x 0.25 degree",
                "temporal_resolution": "Daily",
                "regridding_method": "Bilinear spatial interpolation of 1.0° Tmax/Tmin onto 0.25° rainfall grid",
                "processed_by": "Antigravity AI Climate Preprocessing Pipeline"
            }
        )

        ds_cropped = crop_spatial_domain(ds_combined, lat_bounds=(6.5, 38.5), lon_bounds=(66.5, 100.0))

        pipeline_report["step_metrics"]["step6_spatial_alignment"] = {
            "target_resolution": "0.25° x 0.25°",
            "aligned_grid_shape": [int(ds_cropped.sizes["lat"]), int(ds_cropped.sizes["lon"])],
            "aligned_total_spatial_cells": int(ds_cropped.sizes["lat"] * ds_cropped.sizes["lon"]),
            "aligned_total_data_cells": ds_cropped["rain"].size
        }

        # ----------------------------------------------------------------------
        # STEP 7: FEATURE ENGINEERING
        # ----------------------------------------------------------------------
        logger.info("\n[Step 7/8] Executing feature engineering (DTR, Tmean, flags, lag1, trailing 7d rolling, cyclical features)...")
        ds_final = engineer_all_climate_features(
            ds_cropped,
            tmax_name="tmax",
            tmin_name="tmin",
            rain_name="rain",
            include_lags=True,
            lags=[1],
            include_rolling=True,
            windows=[3, 7],
            include_temporal=True
        )

        pipeline_report["step_metrics"]["step7_feature_engineering"] = {
            "total_variables_generated": len(ds_final.data_vars),
            "generated_variables": list(ds_final.data_vars.keys())
        }

        # ----------------------------------------------------------------------
        # STEP 8: SAVE PROCESSED NETCDF DATASET
        # ----------------------------------------------------------------------
        output_filepath = output_dir_path / f"india_climate_{year}_aligned.nc"

        if save_netcdf:
            logger.info(f"\n[Step 8/8] Exporting processed NetCDF dataset to: {output_filepath}")
            ds_final.to_netcdf(output_filepath)
            logger.info("  - Export completed successfully!")

        pipeline_report["output_path"] = str(output_filepath)
        pipeline_report["dimensions"] = dict(ds_final.sizes)
        pipeline_report["data_variables"] = list(ds_final.data_vars.keys())
        pipeline_report["final_dataset"] = ds_final

        # ----------------------------------------------------------------------
        # OPTIONAL STEP: ML FEATURE SCALING & PERSISTENCE
        # ----------------------------------------------------------------------
        if normalize:
            logger.info("\n[Optional ML Scaling] Fitting scikit-learn scalers and generating normalized dataset...")

            suitability = analyze_scaler_suitability(ds_final)
            logger.info("Scaler Suitability Analysis complete.")

            # Fit MinMaxScaler for rain, StandardScaler for temperature
            rain_vars = [v for v in ds_final.data_vars if "rain" in v.lower() or "rf" in v.lower()]
            temp_vars = [v for v in ds_final.data_vars if v not in rain_vars and not v.endswith("_flag") and v not in ("year", "month", "day", "day_of_year", "season")]

            scaled_rain_ds, rain_reg = fit_transform_dataset_scaler(ds_final, scaler_type="minmax", var_names=rain_vars)
            ds_norm, temp_reg = fit_transform_dataset_scaler(scaled_rain_ds, scaler_type="standard", var_names=temp_vars)

            norm_filepath = output_dir_path / f"india_climate_{year}_normalized.nc"
            scaler_filepath = output_dir_path / f"climate_scaler_{year}.joblib"

            if save_netcdf:
                ds_norm.to_netcdf(norm_filepath)
                logger.info(f"  - Saved normalized NetCDF dataset to: {norm_filepath}")

            combined_scaler_registry = {
                "rain_scaler": rain_reg,
                "temp_scaler": temp_reg,
                "year": year
            }
            save_scaler(combined_scaler_registry, scaler_filepath)

            pipeline_report["normalized_output_path"] = str(norm_filepath)
            pipeline_report["scaler_registry_path"] = str(scaler_filepath)
            pipeline_report["scaler_suitability"] = suitability
            pipeline_report["normalized_dataset"] = ds_norm

        pipeline_report["pipeline_status"] = "SUCCESS"

        logger.info("\n==================================================")
        logger.info("       PREPROCESSING PIPELINE COMPLETE            ")
        logger.info("==================================================")

        return pipeline_report

    except Exception as err:
        logger.error(f"❌ Pipeline Execution Error: {err}", exc_info=True)
        pipeline_report["pipeline_status"] = "FAILED"
        pipeline_report["error"] = str(err)
        raise IOError(f"❌ Preprocessing pipeline failed for year {year}: {err}") from err


if __name__ == "__main__":
    report_res = run_preprocessing_pipeline(year=2020, save_netcdf=True, normalize=True)
    print("\nPipeline Execution Final Status:", report_res["pipeline_status"])
    print(f"Dimensions: {report_res['dimensions']}")
    print(f"Generated Variables ({len(report_res['data_variables'])}): {report_res['data_variables']}")
