"""
Train Baseline Random Forest Model for Short-Term Rainfall Prediction.

Task: Next-day daily precipitation forecasting (mm/day).
Dataset: Processed 2020 IMD NetCDF dataset (data/processed/imd/india_climate_2020_aligned.nc).
Split: Chronological train (70%), validation (15%), test (15%).
Target: 'rain'
Features: Approved leak-free lag1, rolling 3d/7d summaries, physical flags, and cyclical temporal encodings.
"""

import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.problem_definition import get_problem_definition
from src.models.data_split import ChronologicalSplitter, split_processed_dataset
from src.models.baseline_models import create_baseline_model


def extract_tabular_features(
    ds_split: xr.Dataset,
    feature_cols: list,
    target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extracts tabular 2D DataFrame (samples x features) and Series (target)
    for valid land grid cells from a 3D xarray Dataset.
    """
    # Create land mask from day 0 non-NaN tmax
    land_mask_2d = ~np.isnan(ds_split["tmax"].isel(time=0).values)

    # Subset dataset to feature columns + target column
    vars_to_extract = feature_cols + [target_col]
    ds_sub = ds_split[vars_to_extract]

    # Convert to DataFrame
    df = ds_sub.to_dataframe().reset_index()

    # Filter out ocean grid cells using 2D land mask lookup
    # Map (lat, lon) to land_mask_2d
    lat_coords = ds_split["lat"].values
    lon_coords = ds_split["lon"].values
    
    # Filter rows where target column is not NaN
    df_land = df.dropna(subset=[target_col] + feature_cols).copy()

    X = df_land[feature_cols]
    y = df_land[target_col]

    return X, y


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculates MAE, RMSE, and R2 regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def generate_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str
) -> tuple[str, str]:
    """Generates Actual vs Predicted scatter plot and Residual error distribution plot."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    residuals = y_pred - y_true

    # 1. Actual vs Predicted Plot
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    # Subsample points for clear visualization if N is large
    idx = np.random.choice(len(y_true), min(20000, len(y_true)), replace=False) if len(y_true) > 20000 else np.arange(len(y_true))
    
    hb = ax1.hexbin(y_true[idx], y_pred[idx], gridsize=50, cmap="Blues", mincnt=1, bins="log")
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="1:1 Identity Line")
    
    cb = fig1.colorbar(hb, ax=ax1)
    cb.set_label("Log10 Sample Count")
    
    ax1.set_xlabel("Actual Daily Rainfall (mm/day)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Predicted Daily Rainfall (mm/day)", fontsize=12, fontweight="bold")
    ax1.set_title("Random Forest Baseline: Actual vs Predicted Rainfall (Validation Set)", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper left")
    
    fig1_path = os.path.join(output_dir, "01_actual_vs_predicted_rainfall.png")
    fig1.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig1)

    # 2. Residual Error Distribution Plot
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Residual Histogram
    ax2a.hist(residuals, bins=60, color="#2b5c8f", edgecolor="black", alpha=0.85, density=True)
    ax2a.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    ax2a.set_xlabel("Prediction Residual Error (y_pred - y_true) [mm]", fontsize=11, fontweight="bold")
    ax2a.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2a.set_title("Residual Error Distribution", fontsize=12, fontweight="bold")
    ax2a.legend()

    # Residual vs Actual Scatter
    ax2b.hexbin(y_true[idx], residuals[idx], gridsize=50, cmap="Reds", mincnt=1, bins="log")
    ax2b.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax2b.set_xlabel("Actual Daily Rainfall (mm/day)", fontsize=11, fontweight="bold")
    ax2b.set_ylabel("Residual Error [mm]", fontsize=11, fontweight="bold")
    ax2b.set_title("Residual Error vs Actual Value", fontsize=12, fontweight="bold")

    fig2.suptitle("Random Forest Baseline: Rainfall Prediction Error Diagnostics", fontsize=14, fontweight="bold", y=1.02)
    fig2_path = os.path.join(output_dir, "02_prediction_error_distribution_rainfall.png")
    fig2.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig2)

    return fig1_path, fig2_path


def train_rainfall_baseline():
    """Main training workflow for baseline Random Forest Rainfall model."""
    print("=" * 70)
    print("PHASE 4: RANDOM FOREST BASELINE RAINFALL MODEL TRAINING")
    print("=" * 70)

    # 1. Load Problem Specification
    config = get_problem_definition("rainfall")
    print(f"Task Name: {config.task_name}")
    print(f"Target Variable: '{config.target_variable}'")
    print(f"Feature Columns ({len(config.feature_columns)}): {config.feature_columns}")

    # 2. Load and Chronologically Split Dataset
    print("\nLoading and splitting dataset chronologically (70% Train / 15% Val / 15% Test)...")
    splits = split_processed_dataset(warm_up_days=7)
    ds_train = splits["train"]
    ds_val = splits["val"]

    # 3. Extract Tabular Features
    print("\nExtracting land tabular samples...")
    X_train, y_train = extract_tabular_features(ds_train, config.feature_columns, config.target_variable)
    X_val, y_val = extract_tabular_features(ds_val, config.feature_columns, config.target_variable)

    print(f"  - Training Set: {X_train.shape[0]:,} land grid-day samples, {X_train.shape[1]} features")
    print(f"  - Validation Set: {X_val.shape[0]:,} land grid-day samples, {X_val.shape[1]} features")

    # 4. Instantiate and Train Baseline Model
    hyperparams = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }
    print(f"\nInstantiating Random Forest Regressor with hyperparameters:\n  {hyperparams}")
    rf_model = create_baseline_model("random_forest", task_name="rainfall", hyperparameters=hyperparams)

    print("\nFitting Random Forest Regressor ONLY on training data...")
    rf_model.fit(X_train, y_train)

    # 5. Evaluate on Validation Set First
    print("\nEvaluating model on Validation Set...")
    val_preds = rf_model.predict(X_val)
    val_metrics = evaluate_predictions(y_val.values, val_preds)

    print("\n" + "=" * 45)
    print("VALIDATION SET PERFORMANCE METRICS:")
    print("=" * 45)
    print(f"  - Mean Absolute Error (MAE):     {val_metrics['MAE']:.4f} mm/day")
    print(f"  - Root Mean Squared Error (RMSE): {val_metrics['RMSE']:.4f} mm/day")
    print(f"  - R² Score (Variance Explained): {val_metrics['R2']:.4f}")
    print("=" * 45)

    # 6. Save Artifacts and Plots
    model_save_dir = "models/rainfall"
    output_save_dir = "outputs/models/rainfall"
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(output_save_dir, exist_ok=True)

    model_filepath = os.path.join(model_save_dir, "rf_rainfall_baseline.joblib")
    rf_model.save_model(model_filepath)

    fig1_path, fig2_path = generate_evaluation_plots(y_val.values, val_preds, output_save_dir)
    print(f"\nDiagnostic plots saved:")
    print(f"  - Actual vs Predicted: {fig1_path}")
    print(f"  - Error Distribution:  {fig2_path}")

    # Save metrics JSON
    metrics_summary = {
        "model_type": "RandomForestRegressor",
        "task_name": config.task_name,
        "target_variable": config.target_variable,
        "hyperparameters": hyperparams,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "validation_metrics": val_metrics,
        "plots": [fig1_path, fig2_path]
    }
    metrics_json_path = os.path.join(output_save_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"  - Metrics Summary:    {metrics_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    train_rainfall_baseline()
