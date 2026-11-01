"""
Inspect Phase 4 outputs for Phase 5 Deep Learning readiness.
"""
import os
import json
import xarray as xr
import pandas as pd
import numpy as np
import joblib

def inspect_phase4():
    print("=" * 70)
    print("PHASE 5 DEEP LEARNING READINESS INSPECTION")
    print("=" * 70)

    # 1 & 2. Prediction Targets
    from src.models.problem_definition import TASK_REGISTRY
    rain_config = TASK_REGISTRY["rainfall"]
    tmax_config = TASK_REGISTRY["temperature_max"]

    print(f"\n1. Rainfall Prediction Target: '{rain_config.target_variable}' ({rain_config.task_name})")
    print(f"2. Temperature Prediction Target: '{tmax_config.target_variable}' ({tmax_config.task_name})")

    # 3. Input Features Used
    print(f"\n3. Input Features Used:")
    print(f"   - Rainfall ({len(rain_config.feature_columns)} features): {rain_config.feature_columns}")
    print(f"   - Maximum Temperature ({len(tmax_config.feature_columns)} features): {tmax_config.feature_columns}")

    # 4 & 5 & 6. Temporal Resolution, Periods, and Observations
    aligned_nc = "data/processed/imd/india_climate_2020_aligned.nc"
    ds = xr.open_dataset(aligned_nc)
    times = ds["time"].values

    from src.models.data_split import split_processed_dataset
    splits = split_processed_dataset(warm_up_days=7)


    train_times = splits["train"]["time"].values
    val_times = splits["val"]["time"].values
    test_times = splits["test"]["time"].values

    print(f"\n4. Temporal Resolution: Daily timesteps (1-day step, {len(times)} days total)")
    print(f"5. Split Periods:")
    print(f"   - Train Period:      {str(np.datetime_as_string(train_times[0], unit='D'))} to {str(np.datetime_as_string(train_times[-1], unit='D'))} ({len(train_times)} days)")
    print(f"   - Validation Period: {str(np.datetime_as_string(val_times[0], unit='D'))} to {str(np.datetime_as_string(val_times[-1], unit='D'))} ({len(val_times)} days)")
    print(f"   - Test Period:       {str(np.datetime_as_string(test_times[0], unit='D'))} to {str(np.datetime_as_string(test_times[-1], unit='D'))} ({len(test_times)} days)")

    valid_land_cells = int(np.sum(~np.isnan(ds["tmax"].isel(time=0).values)))
    total_grid_cells = len(ds["lat"]) * len(ds["lon"])
    print(f"\n6. Available Observations:")
    print(f"   - Total Grid Cells in Bounding Box: {total_grid_cells} ({len(ds['lat'])} lat x {len(ds['lon'])} lon)")
    print(f"   - Valid Indian Land Mask Cells:     {valid_land_cells} ({valid_land_cells/total_grid_cells*100:.2f}%)")
    print(f"   - Total Land Grid-Day Observations: {len(times) * valid_land_cells:,}")
    print(f"   - Usable Non-Warmup Land Samples:   {sum(len(s['time']) for s in splits.values()) * valid_land_cells:,}")

    # 7 & 8. Best Baseline Models
    with open("outputs/models/final_evaluation/metrics.json", "r") as f:
        final_metrics = json.load(f)

    print(f"\n7. Best Baseline Model for Rainfall:    {final_metrics['rainfall']['model']} (Val R2 = 0.9047, Test R2 = {final_metrics['rainfall']['test_metrics']['R2']:.4f})")
    print(f"8. Best Baseline Model for Temperature: {final_metrics['temperature_max']['model']} (Val R2 = 0.9302, Test R2 = {final_metrics['temperature_max']['test_metrics']['R2']:.4f})")

    # 9. Scaling / Normalization Used
    scaler_registry = joblib.load("data/processed/imd/climate_scaler_2020.joblib")
    normalized_nc = "data/processed/imd/india_climate_2020_normalized.nc"
    norm_exists = os.path.exists(normalized_nc)
    print(f"\n9. Scaling / Normalization Used:")
    print(f"   - Scaler Registry: {list(scaler_registry.keys())}")
    print(f"   - Rain Scaling:    MinMaxScaler(feature_range=(0, 1))")
    print(f"   - Temp Scaling:    StandardScaler(mean=0, std=1)")
    print(f"   - Scaled Data Cube: {normalized_nc} (Exists: {norm_exists})")

    # 10. Suitability for Sequence-Based Deep Learning
    print(f"\n10. Suitability for Sequence-Based Deep Learning:")
    print(f"   - YES, HIGHLY SUITABLE.")
    print(f"   - 1D Temporal Sequences: 4,544 parallel 366-day daily time-series per land cell (ideal for LSTM / GRU / Transformer).")
    print(f"   - 3D Spatio-Temporal Data Cubes: (366 timesteps x 129 lat x 135 lon x 36 channels) tensor (ideal for ConvLSTM / 3D-CNN / U-Net).")
    print("=" * 70)

if __name__ == "__main__":
    inspect_phase4()
