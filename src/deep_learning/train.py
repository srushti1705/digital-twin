"""
Deep Learning Training Pipeline for PyTorch Climate Prediction Models (Phase 5).

Supports:
- Sequence dataset creation and PyTorch DataLoader streaming
- Stacked LSTM architecture instantiation
- MSE / Smooth L1 Loss optimization via Adam/AdamW
- Validation loss monitoring after each epoch
- Early stopping regularization with best checkpoint saving
- Configurable hyperparameters (epochs, batch size, learning rate, patience)
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.problem_definition import get_problem_definition, PredictionProblemConfig
from src.models.data_split import split_processed_dataset
from src.deep_learning.sequence_dataset import build_split_sequence_datasets, create_sequence_dataloaders
from src.deep_learning.lstm_model import build_rainfall_lstm_model, build_temperature_lstm_model, ClimateLSTMModel
from src.deep_learning.utils import set_seed, get_device, count_parameters


class EarlyStopping:
    """
    Early Stopping monitor to halt training when validation loss stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, checkpoint_path: str) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"  [EarlyStopping] Validation loss improved from {self.best_loss:.5f} to {val_loss:.5f}. Saving checkpoint to '{checkpoint_path}'...")
            self.best_loss = val_loss
            self.counter = 0
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss
            }, checkpoint_path)
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs. (Best: {self.best_loss:.5f})")
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class DeepLearningTrainer:
    """
    PyTorch Trainer class for climate deep learning models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_path: str,
        patience: int = 10
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": []
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Runs one full training epoch over train_loader."""
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients in recurrent networks
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            batch_size = x_batch.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        return running_loss / total_samples

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Runs validation evaluation over val_loader without computing gradients."""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            preds = self.model(x_batch)
            loss = self.criterion(preds, y_batch)

            batch_size = x_batch.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

        val_loss = running_loss / total_samples
        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)

        mae = float(np.mean(np.abs(preds_arr - targets_arr)))
        rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))

        return val_loss, mae, rmse

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30
    ) -> Dict[str, List[float]]:
        """
        Executes the full training loop for specified epochs with early stopping.
        """
        print(f"Beginning training on device '{self.device}' for up to {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae, val_rmse = self.evaluate(val_loader)
            ep_dur = time.time() - ep_start

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_mae)
            self.history["val_rmse"].append(val_rmse)

            print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_dur:.1f}s] - Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

            # Early stopping check and model checkpointing
            improved = self.early_stopping(val_loss, self.model, self.checkpoint_path)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {self.early_stopping.best_loss:.5f}")
                break

        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes.")
        return self.history


def train_lstm_task(
    task_key: str = "rainfall",
    sequence_length: int = 7,
    batch_size: int = 256,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 8,
    hidden_dim: int = 64,
    num_layers: int = 2
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Main execution function for training PyTorch LSTM climate models.
    """
    set_seed(42)
    device = get_device()

    print("=" * 70)
    print(f"TRAINING PYTORCH LSTM MODEL FOR TASK: '{task_key.upper()}'")
    print("=" * 70)

    # 1. Load Problem Spec & Splits
    config = get_problem_definition(task_key)
    splits = split_processed_dataset(warm_up_days=7)

    # 2. Build Sequence Datasets
    print(f"\nBuilding {sequence_length}-day windowed sequence datasets...")
    seq_datasets = build_split_sequence_datasets(
        ds_splits=splits,
        feature_cols=config.feature_columns,
        target_col=config.target_variable,
        sequence_length=sequence_length,
        prediction_horizon=1
    )

    # 3. Create PyTorch DataLoaders
    dataloaders = create_sequence_dataloaders(seq_datasets, batch_size=batch_size)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 4. Instantiate Model Architecture
    input_dim = len(config.feature_columns)
    if task_key == "rainfall":
        model = build_rainfall_lstm_model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        checkpoint_path = "models/deep_learning/rainfall/lstm_rainfall_best.pt"
    else:
        model = build_temperature_lstm_model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        checkpoint_path = "models/deep_learning/temperature/lstm_temperature_max_best.pt"

    print(f"\nInstantiated LSTM Architecture ({count_parameters(model):,} trainable parameters):")
    print(model)

    # 5. Define Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 6. Fit Model via DeepLearningTrainer
    trainer = DeepLearningTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path,
        patience=patience
    )

    history = trainer.fit(train_loader, val_loader, epochs=epochs)

    # Load best checkpoint into model
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_val_loss = best_checkpoint["val_loss"]

    # Final validation metrics check
    final_val_loss, final_val_mae, final_val_rmse = trainer.evaluate(val_loader)
    print("\n" + "=" * 50)
    print(f"BEST {task_key.upper()} LSTM VALIDATION RESULTS:")
    print("=" * 50)
    print(f"  - Best Val Loss (MSE): {best_val_loss:.5f}")
    print(f"  - Best Val MAE:        {final_val_mae:.4f}")
    print(f"  - Best Val RMSE:       {final_val_rmse:.4f}")
    print(f"  - Checkpoint Saved To: {checkpoint_path}")
    print("=" * 50)

    # Save training history JSON
    history_save_path = checkpoint_path.replace(".pt", "_history.json")
    with open(history_save_path, "w") as f:
        json.dump(history, f, indent=4)

    return model, history


if __name__ == "__main__":
    train_lstm_task(task_key="rainfall", epochs=5, batch_size=512)
