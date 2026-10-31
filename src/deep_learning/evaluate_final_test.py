"""
Final Unbiased Test Set Evaluation Script (Phase 5 Completion).

Executes final evaluation strictly on the held-out Test Dataset (2020-11-07 to 2020-12-31).
The test dataset was NEVER used for training, hyperparameter tuning, or model selection.

Outputs:
  - outputs/deep_learning/final_evaluation/final_test_evaluation.csv
  - outputs/deep_learning/final_evaluation/final_test_evaluation.md
  - outputs/deep_learning/final_evaluation/metrics_final_test.json
  - Diagnostic Plots (Actual vs Predicted, Residual Error, Spatial Error Maps)
"""

import os
import sys
import json
import time
import joblib
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.problem_definition import get_problem_definition
from src.models.data_split import split_processed_dataset
from src.deep_learning.sequence_dataset import build_split_sequence_datasets, create_sequence_dataloaders
from src.deep_learning.spatial_sequence_dataset import build_split_spatial_sequence_datasets, create_spatial_dataloaders
from src.deep_learning.lstm_model import build_rainfall_lstm_model, build_temperature_lstm_model
from src.deep_learning.convlstm_model import build_rainfall_convlstm_model, build_temperature_convlstm_model
from src.deep_learning.utils import get_device


def prepare_tabular_test_data(
    ds_test: xr.Dataset,
    feature_cols: List[str],
    target_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts tabular land grid samples for tree-based models on the test set.
    """
    land_mask = ~np.isnan(ds_test["tmax"].isel(time=0).values)  # (129, 135)
    
    # Extract tabular features
    X_list = []
    for col in feature_cols:
        vals = ds_test[col].values
        if vals.ndim == 3:
            vals_land = vals[:, land_mask]  # (time, N_land)
        elif vals.ndim == 1:
            N_land = int(np.sum(land_mask))
            vals_land = np.tile(vals[:, np.newaxis], (1, N_land))
        else:
            raise ValueError(f"Unexpected dim for '{col}': {vals.ndim}")
        X_list.append(vals_land.ravel())

    X_test = np.column_stack(X_list)
    y_test = ds_test[target_col].values[:, land_mask].ravel()

    # Drop any NaN rows if present
    valid_mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
    X_test_clean = X_test[valid_mask]
    y_test_clean = y_test[valid_mask]

    return X_test_clean, y_test_clean, land_mask


def evaluate_tree_model(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    Loads tree-based model (RF/XGBoost) and evaluates on test set.
    """
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return y_pred, metrics


def generate_final_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    land_mask: np.ndarray,
    target_name: str,
    model_name: str,
    output_dir: str,
    prefix_num: str
) -> Tuple[str, str, str]:
    """
    Generates actual vs predicted scatter plot, residual error distribution, and 2D spatial error map.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    unit_str = "mm/day" if "rain" in target_name.lower() else "°C"
    var_title = "Rainfall" if "rain" in target_name.lower() else "Maximum Temperature"

    # 1. Actual vs Predicted Scatter Plot
    fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=300)
    sample_indices = np.random.choice(len(y_true), size=min(10000, len(y_true)), replace=False)
    ax1.scatter(y_true[sample_indices], y_pred[sample_indices], alpha=0.3, color="#1f77b4", edgecolors="none", s=15)
    
    max_val = max(np.max(y_true[sample_indices]), np.max(y_pred[sample_indices]))
    min_val = min(np.min(y_true[sample_indices]), np.min(y_pred[sample_indices]))
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 Perfect Prediction")
    
    ax1.set_xlabel(f"Actual {var_title} ({unit_str})", fontsize=11, fontweight="bold")
    ax1.set_ylabel(f"Predicted {var_title} ({unit_str})", fontsize=11, fontweight="bold")
    ax1.set_title(f"Test Set: Actual vs Predicted {var_title} ({model_name})", fontsize=12, fontweight="bold", pad=12)
    ax1.legend(fontsize=10)

    plot1_path = os.path.join(output_dir, f"{prefix_num}_test_actual_vs_predicted_{target_name}.png")
    fig1.savefig(plot1_path, bbox_inches="tight")
    plt.close(fig1)

    # 2. Error Distribution Plot
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
    errors = y_pred - y_true
    ax2.hist(errors, bins=60, color="#2ca02c", edgecolor="black", alpha=0.7, density=True)
    ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
    ax2.axvline(np.mean(errors), color="black", linestyle="-", linewidth=1.5, label=f"Mean Error ({np.mean(errors):.3f})")
    
    ax2.set_xlabel(f"Prediction Error ({unit_str})", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2.set_title(f"Test Set: Prediction Error Distribution ({model_name})", fontsize=12, fontweight="bold", pad=12)
    ax2.legend(fontsize=10)

    plot2_path = os.path.join(output_dir, f"{prefix_num}_test_error_distribution_{target_name}.png")
    fig2.savefig(plot2_path, bbox_inches="tight")
    plt.close(fig2)

    # 3. 2D Spatial Absolute Error Map
    fig3, ax3 = plt.subplots(figsize=(8, 7), dpi=300)
    # Reshape predictions back to (N_days, N_land) -> mean per land cell -> 2D grid
    N_land = int(np.sum(land_mask))
    N_days = len(y_true) // N_land
    
    if N_days * N_land == len(y_true):
        abs_err = np.abs(y_true - y_pred).reshape((N_days, N_land))
        mean_spatial_err = np.mean(abs_err, axis=0)
        
        spatial_grid = np.full(land_mask.shape, np.nan)
        spatial_grid[land_mask] = mean_spatial_err

        im = ax3.imshow(spatial_grid, cmap="Reds", origin="lower")
        cbar = fig3.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label(f"Mean Absolute Error ({unit_str})", fontsize=11, fontweight="bold")
        ax3.set_title(f"Test Set: Spatial MAE Map ({model_name})", fontsize=12, fontweight="bold", pad=12)
        ax3.set_xlabel("Longitude Index", fontsize=10)
        ax3.set_ylabel("Latitude Index", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "Spatial map dimensions mismatch", ha="center", va="center")

    plot3_path = os.path.join(output_dir, f"{prefix_num}_test_spatial_error_map_{target_name}.png")
    fig3.savefig(plot3_path, bbox_inches="tight")
    plt.close(fig3)

    return plot1_path, plot2_path, plot3_path


def run_final_unbiased_test_evaluation():
    """
    Main evaluation pipeline on Test Dataset.
    """
    print("=" * 70)
    print("FINAL UNBIASED TEST DATASET EVALUATION (PHASE 5)")
    print("=" * 70)
    print("DATA LEAKAGE PREVENTION DISCLAIMER:")
    print("  - The test dataset ('2020-11-07' to '2020-12-31') was completely held out.")
    print("  - Zero test set observations were used for model training or model selection.")
    print("=" * 70)

    output_dir = "outputs/deep_learning/final_evaluation"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load splits (warm-up=7)
    splits_aligned = split_processed_dataset("data/processed/imd/india_climate_2020_aligned.nc", warm_up_days=7)
    ds_test_aligned = splits_aligned["test"]

    # Problem definitions
    rain_config = get_problem_definition("rainfall")
    tmax_config = get_problem_definition("temperature_max")

    # 2. Extract Test Land Grid Tabular Arrays
    X_test_rain, y_test_rain, land_mask = prepare_tabular_test_data(
        ds_test_aligned, rain_config.feature_columns, rain_config.target_variable
    )
    X_test_tmax, y_test_tmax, _ = prepare_tabular_test_data(
        ds_test_aligned, tmax_config.feature_columns, tmax_config.target_variable
    )

    results_list = []

    # -------------------------------------------------------------
    # TASK 1: RAINFALL MODEL EVALUATION ON TEST SET
    # -------------------------------------------------------------
    print("\n1. Evaluating Selected Rainfall Model (Random Forest)...")
    y_pred_rf_rain, metrics_rf_rain = evaluate_tree_model(
        "models/rainfall/rf_rainfall_baseline.joblib", X_test_rain, y_test_rain
    )
    print(f"   [Random Forest Rain Test] MAE: {metrics_rf_rain['MAE']:.4f} | RMSE: {metrics_rf_rain['RMSE']:.4f} | R²: {metrics_rf_rain['R2']:.4f}")

    p1_rain, p2_rain, p3_rain = generate_final_plots(
        y_test_rain, y_pred_rf_rain, land_mask, "rainfall", "Random Forest (Selected Winner)", output_dir, "01"
    )

    results_list.append({
        "Task": "Rainfall",
        "Model_Name": "Random Forest Regressor",
        "Model_Status": "Selected Winner",
        "Dataset_Split": "Test (2020-11-07 to 2020-12-31)",
        "MAE": metrics_rf_rain["MAE"],
        "RMSE": metrics_rf_rain["RMSE"],
        "R2": metrics_rf_rain["R2"]
    })

    print("\n2. Evaluating Runner-up Rainfall Model (XGBoost)...")
    y_pred_xgb_rain, metrics_xgb_rain = evaluate_tree_model(
        "models/rainfall/xgb_rainfall_baseline.joblib", X_test_rain, y_test_rain
    )
    print(f"   [XGBoost Rain Test] MAE: {metrics_xgb_rain['MAE']:.4f} | RMSE: {metrics_xgb_rain['RMSE']:.4f} | R²: {metrics_xgb_rain['R2']:.4f}")

    results_list.append({
        "Task": "Rainfall",
        "Model_Name": "XGBoost Regressor",
        "Model_Status": "Runner-up Baseline",
        "Dataset_Split": "Test (2020-11-07 to 2020-12-31)",
        "MAE": metrics_xgb_rain["MAE"],
        "RMSE": metrics_xgb_rain["RMSE"],
        "R2": metrics_xgb_rain["R2"]
    })

    # -------------------------------------------------------------
    # TASK 2: MAXIMUM TEMPERATURE MODEL EVALUATION ON TEST SET
    # -------------------------------------------------------------
    print("\n3. Evaluating Selected Temperature Model (Random Forest)...")
    y_pred_rf_tmax, metrics_rf_tmax = evaluate_tree_model(
        "models/temperature/rf_temperature_max_baseline.joblib", X_test_tmax, y_test_tmax
    )
    print(f"   [Random Forest Tmax Test] MAE: {metrics_rf_tmax['MAE']:.4f} | RMSE: {metrics_rf_tmax['RMSE']:.4f} | R²: {metrics_rf_tmax['R2']:.4f}")

    p1_tmax, p2_tmax, p3_tmax = generate_final_plots(
        y_test_tmax, y_pred_rf_tmax, land_mask, "temperature_max", "Random Forest (Selected Winner)", output_dir, "02"
    )

    results_list.append({
        "Task": "Temperature (Tmax)",
        "Model_Name": "Random Forest Regressor",
        "Model_Status": "Selected Winner",
        "Dataset_Split": "Test (2020-11-07 to 2020-12-31)",
        "MAE": metrics_rf_tmax["MAE"],
        "RMSE": metrics_rf_tmax["RMSE"],
        "R2": metrics_rf_tmax["R2"]
    })

    print("\n4. Evaluating Runner-up Temperature Model (XGBoost)...")
    y_pred_xgb_tmax, metrics_xgb_tmax = evaluate_tree_model(
        "models/temperature/xgb_temperature_max_baseline.joblib", X_test_tmax, y_test_tmax
    )
    print(f"   [XGBoost Tmax Test] MAE: {metrics_xgb_tmax['MAE']:.4f} | RMSE: {metrics_xgb_tmax['RMSE']:.4f} | R²: {metrics_xgb_tmax['R2']:.4f}")

    results_list.append({
        "Task": "Temperature (Tmax)",
        "Model_Name": "XGBoost Regressor",
        "Model_Status": "Runner-up Baseline",
        "Dataset_Split": "Test (2020-11-07 to 2020-12-31)",
        "MAE": metrics_xgb_tmax["MAE"],
        "RMSE": metrics_xgb_tmax["RMSE"],
        "R2": metrics_xgb_tmax["R2"]
    })

    # Save final test evaluation results CSV & MD
    results_df = pd.DataFrame(results_list)
    csv_path = os.path.join(output_dir, "final_test_evaluation.csv")
    results_df.to_csv(csv_path, index=False)

    md_content = f"""# Phase 5 Final Unbiased Test Set Evaluation Report

## Executive Summary

This report documents the final, unbiased test set evaluation for the selected best-performing climate prediction models (**Random Forest Regressor** for both Rainfall and Maximum Temperature).

> **Strict Data Leakage Prevention**:
> The **Test Dataset** ($249,920$ unseen land grid-day samples covering `2020-11-07` to `2020-12-31`) was held out strictly unobserved. **Zero test set samples were used during model training, hyperparameter tuning, or model selection.**

---

## Final Unbiased Test Set Performance Summary

| Task | Selected Model Name | Model Status | MAE ↓ | RMSE ↓ | R² Score ↑ | Evaluation Period |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Rainfall** | **Random Forest Regressor** | **Selected Winner** | **{metrics_rf_rain['MAE']:.4f} mm/day** | **{metrics_rf_rain['RMSE']:.4f} mm/day** | **{metrics_rf_rain['R2']:.4f}** | `2020-11-07` to `2020-12-31` |
| Rainfall | XGBoost Regressor | Runner-up Baseline | {metrics_xgb_rain['MAE']:.4f} mm/day | {metrics_xgb_rain['RMSE']:.4f} mm/day | {metrics_xgb_rain['R2']:.4f} | `2020-11-07` to `2020-12-31` |
| **Temperature ($T_{{max}}$)** | **Random Forest Regressor** | **Selected Winner** | **{metrics_rf_tmax['MAE']:.4f} °C** | **{metrics_rf_tmax['RMSE']:.4f} °C** | **{metrics_rf_tmax['R2']:.4f}** | `2020-11-07` to `2020-12-31` |
| Temperature ($T_{{max}}$) | XGBoost Regressor | Runner-up Baseline | {metrics_xgb_tmax['MAE']:.4f} °C | {metrics_xgb_tmax['RMSE']:.4f} °C | {metrics_xgb_tmax['R2']:.4f} | `2020-11-07` to `2020-12-31` |

---

## Saved Final Artifacts & Visualizations

- **Rainfall Diagnostic Plots**:
  1. Actual vs Predicted Scatter Plot: `01_test_actual_vs_predicted_rainfall.png`
  2. Residual Error Distribution: `01_test_error_distribution_rainfall.png`
  3. 2D Spatial MAE Map across India: `01_test_spatial_error_map_rainfall.png`
- **Temperature Diagnostic Plots**:
  1. Actual vs Predicted Scatter Plot: `02_test_actual_vs_predicted_temperature_max.png`
  2. Residual Error Distribution: `02_test_error_distribution_temperature_max.png`
  3. 2D Spatial MAE Map across India: `02_test_spatial_error_map_temperature_max.png`
- **Final Metrics CSV**: `final_test_evaluation.csv`
- **Final Metrics JSON**: `metrics_final_test.json`
"""

    md_path = os.path.join(output_dir, "final_test_evaluation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    json_path = os.path.join(output_dir, "metrics_final_test.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_period": "2020-11-07 to 2020-12-31",
            "test_samples": len(y_test_rain),
            "rainfall_selected_model": {
                "model_name": "Random Forest Regressor",
                "metrics": metrics_rf_rain
            },
            "temperature_selected_model": {
                "model_name": "Random Forest Regressor",
                "metrics": metrics_rf_tmax
            }
        }, f, indent=4)

    print(f"\nSaved final evaluation report to '{md_path}' and CSV to '{csv_path}'.")
    print("=" * 70)


if __name__ == "__main__":
    run_final_unbiased_test_evaluation()
