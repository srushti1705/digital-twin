"""
Train Baseline PyTorch LSTM Model for Short-Term Maximum Temperature Prediction.

Task: Next-day daily maximum surface air temperature forecasting (°C).
Dataset: Processed 2020 IMD NetCDF dataset (data/processed/imd/india_climate_2020_aligned.nc).
Sequence Length: 7 days.
Target: 'tmax'
Features: Approved 21 leak-free maximum temperature features.
Evaluation: Evaluated strictly on the Validation set (2020-09-15 to 2020-11-06).
"""

import os
import json
import time
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
from src.deep_learning.lstm_model import build_temperature_lstm_model
from src.deep_learning.train import DeepLearningTrainer
from src.deep_learning.utils import set_seed, get_device, count_parameters


def generate_lstm_temperature_plots(
    history: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str
) -> tuple[str, str, str]:
    """
    Generates:
    1. Training vs Validation Loss Curve
    2. Actual vs Predicted Temperature Scatter Plot
    3. Residual Error Distribution Plot
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Training vs Validation Loss Curve
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "o-", color="#d35400", linewidth=2, label="Train Loss (MSE)")
    ax1.plot(epochs, history["val_loss"], "s-", color="#c0392b", linewidth=2, label="Validation Loss (MSE)")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Squared Error (MSE Loss)", fontsize=11, fontweight="bold")
    ax1.set_title("LSTM Maximum Temperature Model: Training vs Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(fontsize=11)
    
    fig1_path = os.path.join(output_dir, "01_temperature_lstm_loss_curve.png")
    fig1.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig1)

    # 2. Actual vs Predicted Scatter Plot
    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=300)
    idx = np.random.choice(len(y_true), min(20000, len(y_true)), replace=False) if len(y_true) > 20000 else np.arange(len(y_true))
    
    hb = ax2.hexbin(y_true[idx], y_pred[idx], gridsize=50, cmap="Oranges", mincnt=1, bins="log")
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], "b--", linewidth=2, label="1:1 Identity Line")
    
    cb = fig2.colorbar(hb, ax=ax2)
    cb.set_label("Log10 Sample Count")
    
    ax2.set_xlabel("Actual Daily Tmax (°C)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Predicted Daily Tmax (°C)", fontsize=12, fontweight="bold")
    ax2.set_title("LSTM Baseline: Actual vs Predicted Maximum Temperature (Validation Set)", fontsize=13, fontweight="bold", pad=12)
    ax2.legend(loc="upper left")
    
    fig2_path = os.path.join(output_dir, "02_temperature_lstm_actual_vs_predicted.png")
    fig2.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig2)

    # 3. Residual Error Distribution Plot
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    residuals = y_pred - y_true
    
    # Residual Histogram
    ax3a.hist(residuals, bins=60, color="#d35400", edgecolor="black", alpha=0.85, density=True)
    ax3a.axvline(0, color="blue", linestyle="--", linewidth=1.5, label="Zero Error")
    ax3a.set_xlabel("Prediction Residual Error (y_pred - y_true) [°C]", fontsize=11, fontweight="bold")
    ax3a.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax3a.set_title("LSTM Temperature Residual Error Distribution", fontsize=12, fontweight="bold")
    ax3a.legend()

    # Residual vs Actual Scatter
    ax3b.hexbin(y_true[idx], residuals[idx], gridsize=50, cmap="PuRd", mincnt=1, bins="log")
    ax3b.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax3b.set_xlabel("Actual Daily Tmax (°C)", fontsize=11, fontweight="bold")
    ax3b.set_ylabel("Residual Error [°C]", fontsize=11, fontweight="bold")
    ax3b.set_title("Residual Error vs Actual Value", fontsize=12, fontweight="bold")

    fig3.suptitle("LSTM Baseline: Temperature Prediction Error Diagnostics", fontsize=14, fontweight="bold", y=1.02)
    fig3_path = os.path.join(output_dir, "03_temperature_lstm_error_distribution.png")
    fig3.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig3)

    return fig1_path, fig2_path, fig3_path


def train_temperature_lstm_pipeline():
    """Main training workflow for PyTorch LSTM Maximum Temperature model."""
    set_seed(42)
    device = get_device()

    print("=" * 70)
    print("PHASE 5: PYTORCH LSTM MAXIMUM TEMPERATURE MODEL TRAINING")
    print("=" * 70)

    # 1. Load Problem Spec
    config = get_problem_definition("temperature_max")
    print(f"Task Name: {config.task_name}")
    print(f"Target Variable: '{config.target_variable}'")
    print(f"Feature Columns ({len(config.feature_columns)}): {config.feature_columns}")

    # 2. Load Dataset Splits
    print("\nLoading and splitting dataset chronologically (70% Train / 15% Val / 15% Test)...")
    splits = split_processed_dataset(warm_up_days=7)

    # 3. Build 7-Day Windowed Sequence Datasets
    seq_len = 7
    batch_size = 512
    epochs = 15
    patience = 5

    print(f"\nBuilding {seq_len}-day windowed PyTorch sequence datasets...")
    seq_datasets = build_split_sequence_datasets(
        ds_splits=splits,
        feature_cols=config.feature_columns,
        target_col=config.target_variable,
        sequence_length=seq_len,
        prediction_horizon=1
    )

    dataloaders = create_sequence_dataloaders(seq_datasets, batch_size=batch_size)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 4. Instantiate PyTorch LSTM Architecture
    input_dim = len(config.feature_columns)
    model = build_temperature_lstm_model(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    
    model_save_dir = "models/deep_learning/temperature"
    output_save_dir = "outputs/deep_learning/temperature"
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(output_save_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_save_dir, "lstm_temperature_max_best.pt")

    print(f"\nInstantiated PyTorch LSTM Architecture ({count_parameters(model):,} trainable parameters):")
    print(model)

    # 5. Define Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 6. Fit Model via DeepLearningTrainer
    trainer = DeepLearningTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path,
        patience=patience
    )

    print("\nTraining PyTorch LSTM ONLY on training set (Validation monitoring active)...")
    history = trainer.fit(train_loader, val_loader, epochs=epochs)

    # 7. Load Best Checkpoint for Final Validation Evaluation
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Collect predictions over Validation Set
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b = x_b.to(device)
            preds_b = model(x_b)
            val_preds_list.append(preds_b.cpu().numpy())
            val_targets_list.append(y_b.numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    # Calculate Regression Metrics
    mae = float(mean_absolute_error(val_targets, val_preds))
    rmse = float(np.sqrt(mean_squared_error(val_targets, val_preds)))
    r2 = float(r2_score(val_targets, val_preds))

    val_metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print("\n" + "=" * 50)
    print("LSTM MAXIMUM TEMPERATURE VALIDATION SET PERFORMANCE METRICS:")
    print("=" * 50)
    print(f"  - Mean Absolute Error (MAE):     {mae:.4f} °C")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f} °C")
    print(f"  - R² Score (Variance Explained): {r2:.4f}")
    print("=" * 50)

    # 8. Generate Visual Diagnostic Plots
    fig1_p, fig2_p, fig3_p = generate_lstm_temperature_plots(history, val_targets, val_preds, output_save_dir)
    print(f"\nDiagnostic plots generated:")
    print(f"  - Loss Curve Plot:      {fig1_p}")
    print(f"  - Actual vs Predicted:   {fig2_p}")
    print(f"  - Error Distribution:    {fig3_p}")

    # 9. Save Metrics Summary JSON
    metrics_summary = {
        "model_architecture": "PyTorch Stacked ClimateLSTMModel",
        "task_name": config.task_name,
        "target_variable": config.target_variable,
        "sequence_length": seq_len,
        "input_dim": input_dim,
        "hidden_dim": 64,
        "num_layers": 2,
        "trainable_parameters": count_parameters(model),
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 1e-3,
            "patience": patience,
            "loss_function": "MSELoss",
            "optimizer": "AdamW"
        },
        "validation_metrics": val_metrics,
        "checkpoint_saved": checkpoint_path,
        "plots": [fig1_p, fig2_p, fig3_p]
    }

    json_path = os.path.join(output_save_dir, "metrics_lstm_tmax.json")
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"  - Summary JSON Saved:   {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    train_temperature_lstm_pipeline()
