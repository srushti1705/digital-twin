"""
ConvLSTM Model Training and Evaluation Pipeline (Phase 5).

Trains PyTorch ConvLSTM on 5D spatio-temporal sequences:
  - Input shape: (Batch, Sequence_Length, Channels, Lat, Lon)
  - Output shape: (Batch, Lat, Lon)

Strictly enforces chronological boundaries (Train / Val / Test).
Calculates MAE, RMSE, and R2 on unseen land grid points of the validation split.
Generates diagnostic spatial maps and loss curve plots.
"""

import os
import sys
import json
import time
import joblib
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.problem_definition import get_problem_definition
from src.models.data_split import split_processed_dataset
from src.deep_learning.spatial_sequence_dataset import (
    build_split_spatial_sequence_datasets,
    create_spatial_dataloaders
)
from src.deep_learning.convlstm_model import build_rainfall_convlstm_model, build_temperature_convlstm_model
from src.deep_learning.utils import set_seed, get_device, count_parameters


class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error Loss computed strictly over valid land grid cells.
    Ignored ocean/non-land cells (masked as 0).
    """

    def __init__(self, land_mask: torch.Tensor):
        super().__init__()
        # land_mask shape: (H, W) or (1, H, W)
        if land_mask.ndim == 2:
            land_mask = land_mask.unsqueeze(0)
        self.register_buffer("land_mask", land_mask.float())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target shape: (Batch, H, W)
        mask = self.land_mask.to(pred.device)
        diff = (pred - target) * mask
        loss = torch.sum(diff ** 2) / (torch.sum(mask) * pred.size(0) + 1e-8)
        return loss


def unscale_target_predictions(
    y_scaled: np.ndarray,
    target_var: str,
    scaler_info: dict
) -> np.ndarray:
    """
    Unscales target predictions back to physical units (mm/day for rain, °C for tmax).
    """
    if target_var == "rain":
        max_val = scaler_info["rain_scaler"]["variable_stats"]["rain"]["max"]
        min_val = scaler_info["rain_scaler"]["variable_stats"]["rain"]["min"]
        return y_scaled * (max_val - min_val) + min_val
    elif target_var == "tmax":
        mean_val = scaler_info["temp_scaler"]["variable_stats"]["tmax"]["mean"]
        std_val = scaler_info["temp_scaler"]["variable_stats"]["tmax"]["std"]
        return y_scaled * std_val + mean_val
    else:
        return y_scaled


def generate_convlstm_diagnostic_plots(
    history: dict,
    y_true_spatial: np.ndarray,   # (H, W) mean validation spatial map in physical units
    y_pred_spatial: np.ndarray,   # (H, W) mean validation spatial map in physical units
    land_mask: np.ndarray,        # (H, W) boolean mask
    target_var: str,
    output_dir: str
) -> Tuple[str, str, str, str]:
    """
    Generates:
    1. Training vs Validation Loss Curve
    2. Predicted Climate Spatial Map
    3. Actual Climate Spatial Map
    4. Prediction Error Spatial Map
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    unit_str = "mm/day" if target_var == "rain" else "°C"
    var_title = "Rainfall" if target_var == "rain" else "Maximum Temperature"

    # Mask non-land grid cells with NaN for plotting cleanly
    y_true_plot = np.where(land_mask, y_true_spatial, np.nan)
    y_pred_plot = np.where(land_mask, y_pred_spatial, np.nan)
    error_plot = np.where(land_mask, np.abs(y_true_spatial - y_pred_spatial), np.nan)

    # 1. Training vs Validation Loss Curve
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", linewidth=2, label="Train Loss (Masked MSE)")
    ax1.plot(epochs, history["val_loss"], "s-", color="#ff7f0e", linewidth=2, label="Validation Loss (Masked MSE)")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Masked MSE Loss", fontsize=11, fontweight="bold")
    ax1.set_title(f"ConvLSTM {var_title} Model: Training vs Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(fontsize=11)

    fig1_path = os.path.join(output_dir, "01_convlstm_loss_curve.png")
    fig1.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig1)

    cmap = "Blues" if target_var == "rain" else "YlOrRd"

    # 2. Predicted Climate Map
    fig2, ax2 = plt.subplots(figsize=(8, 7), dpi=300)
    im2 = ax2.imshow(y_pred_plot, cmap=cmap, origin="lower")
    cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label(f"Predicted {var_title} ({unit_str})", fontsize=11, fontweight="bold")
    ax2.set_title(f"ConvLSTM Validation Mean Predicted {var_title} Map", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Longitude Index", fontsize=10)
    ax2.set_ylabel("Latitude Index", fontsize=10)

    fig2_path = os.path.join(output_dir, "02_convlstm_predicted_spatial_map.png")
    fig2.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig2)

    # 3. Actual Climate Map
    fig3, ax3 = plt.subplots(figsize=(8, 7), dpi=300)
    im3 = ax3.imshow(y_true_plot, cmap=cmap, origin="lower")
    cbar3 = fig3.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label(f"Actual {var_title} ({unit_str})", fontsize=11, fontweight="bold")
    ax3.set_title(f"IMD Ground-Truth Mean Validation {var_title} Map", fontsize=13, fontweight="bold", pad=12)
    ax3.set_xlabel("Longitude Index", fontsize=10)
    ax3.set_ylabel("Latitude Index", fontsize=10)

    fig3_path = os.path.join(output_dir, "03_convlstm_actual_spatial_map.png")
    fig3.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig3)

    # 4. Prediction Error Spatial Map
    fig4, ax4 = plt.subplots(figsize=(8, 7), dpi=300)
    im4 = ax4.imshow(error_plot, cmap="Reds", origin="lower")
    cbar4 = fig4.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label(f"Absolute Prediction Error ({unit_str})", fontsize=11, fontweight="bold")
    ax4.set_title(f"ConvLSTM Spatial Absolute Prediction Error Map", fontsize=13, fontweight="bold", pad=12)
    ax4.set_xlabel("Longitude Index", fontsize=10)
    ax4.set_ylabel("Latitude Index", fontsize=10)

    fig4_path = os.path.join(output_dir, "04_convlstm_error_spatial_map.png")
    fig4.savefig(fig4_path, bbox_inches="tight")
    plt.close(fig4)

    return fig1_path, fig2_path, fig3_path, fig4_path


def train_convlstm_model(
    task_type: str = "rainfall",  # 'rainfall' or 'temperature_max'
    epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    patience: int = 5
) -> dict:
    """
    Main training function for ConvLSTM model.
    """
    set_seed(42)
    device = get_device()

    task_key = "rainfall" if "rain" in task_type.lower() else "temperature_max"
    config = get_problem_definition(task_key)

    print("=" * 70)
    print(f"TRAINING CONVLSTM MODEL: {config.task_name.upper()}")
    print("=" * 70)
    print(f"Target Variable:  {config.target_variable}")
    print(f"Device:           {device}")

    # Load Scaler metadata for physical unscaling
    scaler_info = joblib.load("data/processed/imd/climate_scaler_2020.joblib")

    # Load splits
    splits = split_processed_dataset("data/processed/imd/india_climate_2020_normalized.nc", warm_up_days=7)
    
    seq_len = 7
    print(f"\nBuilding 5D Spatial ConvLSTM datasets (Sequence length L={seq_len})...")
    spatial_datasets, land_mask = build_split_spatial_sequence_datasets(
        ds_splits=splits,
        feature_cols=config.feature_columns,
        target_col=config.target_variable,
        sequence_length=seq_len
    )

    dataloaders = create_spatial_dataloaders(spatial_datasets, batch_size=batch_size)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    input_dim = len(config.feature_columns)

    if config.target_variable == "rain":
        model = build_rainfall_convlstm_model(input_dim=input_dim, hidden_dims=[32, 32], num_layers=2, kernel_size=3)
    else:
        model = build_temperature_convlstm_model(input_dim=input_dim, hidden_dims=[32, 32], num_layers=2, kernel_size=3)

    model = model.to(device)

    land_mask_tensor = torch.tensor(land_mask, dtype=torch.float32, device=device)
    criterion = MaskedMSELoss(land_mask_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model_dir = "models/deep_learning"
    output_dir = "outputs/deep_learning"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    best_checkpoint_path = os.path.join(model_dir, f"convlstm_{config.target_variable}_best.pt")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)  # (Batch, H, W)
            loss = criterion(preds, batch_y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)

        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                preds = model(batch_x)
                loss = criterion(preds, batch_y)

                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "input_dim": input_dim,
                    "target_variable": config.target_variable,
                    "sequence_length": seq_len
                }
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to '{best_checkpoint_path}' (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Validation loss did not improve for {patience} consecutive epochs.")
                break

    training_time = time.time() - start_time

    # Final Validation Evaluation
    print(f"\nEvaluating Best Checkpoint on Validation Split...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_preds_list = []
    val_true_list = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            val_preds_list.append(preds.cpu().numpy())
            val_true_list.append(batch_y.numpy())

    y_pred_scaled = np.concatenate(val_preds_list, axis=0)  # (N_val_windows, 129, 135)
    y_true_scaled = np.concatenate(val_true_list, axis=0)   # (N_val_windows, 129, 135)

    # Unscale predictions and ground truth to physical units
    y_pred_physical = unscale_target_predictions(y_pred_scaled, config.target_variable, scaler_info)
    y_true_physical = unscale_target_predictions(y_true_scaled, config.target_variable, scaler_info)

    # Extract land grid cells across all validation time windows
    land_mask_3d = np.tile(land_mask[np.newaxis, :, :], (y_pred_physical.shape[0], 1, 1))
    y_pred_land = y_pred_physical[land_mask_3d]
    y_true_land = y_true_physical[land_mask_3d]

    # Calculate metrics strictly on valid land points
    mae = float(mean_absolute_error(y_true_land, y_pred_land))
    rmse = float(np.sqrt(mean_squared_error(y_true_land, y_pred_land)))
    r2 = float(r2_score(y_true_land, y_pred_land))

    print(f"\nValidation Performance Metrics ({config.target_variable.upper()}):")
    print(f"  - MAE:  {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R²:   {r2:.4f}")

    # Compute mean spatial maps for plotting
    y_true_mean_map = np.mean(y_true_physical, axis=0)  # (129, 135)
    y_pred_mean_map = np.mean(y_pred_physical, axis=0)  # (129, 135)

    # Generate diagnostic plots
    fig1, fig2, fig3, fig4 = generate_convlstm_diagnostic_plots(
        history=history,
        y_true_spatial=y_true_mean_map,
        y_pred_spatial=y_pred_mean_map,
        land_mask=land_mask,
        target_var=config.target_variable,
        output_dir=output_dir
    )

    metrics_summary = {
        "model_architecture": "PyTorch ClimateConvLSTMModel",
        "task_name": config.task_name,
        "target_variable": config.target_variable,
        "sequence_length": seq_len,
        "input_dim": input_dim,
        "trainable_parameters": count_parameters(model),
        "validation_metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        },
        "training_time_seconds": round(training_time, 2),
        "checkpoint_saved": best_checkpoint_path,
        "plots": [fig1, fig2, fig3, fig4]
    }

    metrics_json_path = os.path.join(output_dir, f"metrics_convlstm_{config.target_variable}.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"\nSaved evaluation results to '{metrics_json_path}'.")
    return metrics_summary


if __name__ == "__main__":
    train_convlstm_model(task_type="rainfall", epochs=10, batch_size=8)
