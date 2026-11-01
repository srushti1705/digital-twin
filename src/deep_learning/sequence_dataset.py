"""
PyTorch Sequence Dataset Module for Climate Prediction (Phase 5).

Converts processed climate time-series into supervised sequence windows for deep learning models:
  Input Sequence: [Day t-L, ..., Day t-1] (length L)
  Target Value:   [Day t] (horizon = 1)

Guarantees zero temporal data leakage and strict split boundary separation.
"""

from typing import Tuple, List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.data_split import ChronologicalSplitter


class ClimateSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapper for sequence-based climate forecasting.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray
    ):
        """
        Parameters
        ----------
        sequences : np.ndarray
            3D numpy array of shape (num_samples, sequence_length, num_features).
        targets : np.ndarray
            1D or 2D numpy array of shape (num_samples,) or (num_samples, target_dim).
        """
        if len(sequences) != len(targets):
            raise ValueError(
                f"Mismatch between sequences ({len(sequences)}) and targets ({len(targets)}) count."
            )
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def create_sliding_window_sequences(
    X_data: np.ndarray,
    y_data: np.ndarray,
    sequence_length: int = 7,
    prediction_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D/3D time-series matrices into supervised sliding window sequence pairs.

    Parameters
    ----------
    X_data : np.ndarray
        Feature matrix of shape (T_timesteps, N_spatial, F_features) or (T_timesteps, F_features).
    y_data : np.ndarray
        Target matrix of shape (T_timesteps, N_spatial) or (T_timesteps,).
    sequence_length : int, default=7
        Number of historical past timesteps included in input sequence (L).
    prediction_horizon : int, default=1
        Number of timesteps ahead to predict (1 = next day).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X_seq: (num_samples, sequence_length, F_features)
        y_seq: (num_samples,)
    """
    if X_data.ndim == 3:  # (T_timesteps, N_spatial, F_features)
        T_timesteps, N_spatial, F_features = X_data.shape
        num_windows = T_timesteps - sequence_length - prediction_horizon + 1
        if num_windows <= 0:
            return np.empty((0, sequence_length, F_features)), np.empty((0,))

        X_seq_list = []
        y_seq_list = []

        # Vectorized window extraction across all spatial cells simultaneously
        for t in range(num_windows):
            X_win = X_data[t : t + sequence_length, :, :]  # (sequence_length, N_spatial, F_features)
            X_win = np.transpose(X_win, (1, 0, 2))          # (N_spatial, sequence_length, F_features)
            y_win = y_data[t + sequence_length + prediction_horizon - 1, :]  # (N_spatial,)

            # Filter valid land cells (no NaNs in input window or target)
            valid_mask = ~np.isnan(X_win).any(axis=(1, 2)) & ~np.isnan(y_win)
            if np.any(valid_mask):
                X_seq_list.append(X_win[valid_mask])
                y_seq_list.append(y_win[valid_mask])

        if not X_seq_list:
            return np.empty((0, sequence_length, F_features)), np.empty((0,))

        return np.concatenate(X_seq_list, axis=0).astype(np.float32), np.concatenate(y_seq_list, axis=0).astype(np.float32)


    elif X_data.ndim == 2:  # Single time series (T_timesteps, F_features)
        T_timesteps, F_features = X_data.shape
        num_samples = T_timesteps - sequence_length - prediction_horizon + 1
        if num_samples <= 0:
            return np.empty((0, sequence_length, F_features)), np.empty((0,))

        X_seq_list = []
        y_seq_list = []

        for t in range(num_samples):
            seq_x = X_data[t : t + sequence_length, :]
            seq_y = y_data[t + sequence_length + prediction_horizon - 1]

            if not np.isnan(seq_x).any() and not np.isnan(seq_y):
                X_seq_list.append(seq_x)
                y_seq_list.append(seq_y)

        return np.array(X_seq_list, dtype=np.float32), np.array(y_seq_list, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported feature matrix dimension: {X_data.ndim}")


def build_split_sequence_datasets(
    ds_splits: Dict[str, xr.Dataset],
    feature_cols: List[str],
    target_col: str,
    sequence_length: int = 7,
    prediction_horizon: int = 1
) -> Dict[str, ClimateSequenceDataset]:
    """
    Constructs PyTorch ClimateSequenceDataset objects for train, val, and test splits
    by windowing within each split boundary independently (preventing cross-split leakage).

    Parameters
    ----------
    ds_splits : Dict[str, xr.Dataset]
        Dictionary of datasets ('train', 'val', 'test').
    feature_cols : List[str]
        List of feature column names.
    target_col : str
        Target column name.
    sequence_length : int, default=7
        Length of historical input sequence window.
    prediction_horizon : int, default=1
        Target prediction horizon (1-day ahead).

    Returns
    -------
    Dict[str, ClimateSequenceDataset]
        Dictionary of PyTorch datasets ('train', 'val', 'test').
    """
    datasets = {}

    for split_name, ds in ds_splits.items():
        # Reshape xarray to (time, land_cell, features)
        land_mask = ~np.isnan(ds["tmax"].isel(time=0).values)
        
        # Extract features and target arrays
        X_arr_list = []
        for col in feature_cols:
            vals = ds[col].values  # (time, lat, lon) or (time,)
            if vals.ndim == 3:
                vals_land = vals[:, land_mask]  # (time, N_land)
            elif vals.ndim == 1:
                N_land = int(np.sum(land_mask))
                vals_land = np.tile(vals[:, np.newaxis], (1, N_land))  # (time, N_land)
            else:
                raise ValueError(f"Unexpected dimension for feature '{col}': {vals.ndim}")
            X_arr_list.append(vals_land)

        X_data = np.stack(X_arr_list, axis=-1)  # (time, N_land, F_features)
        y_data = ds[target_col].values
        if y_data.ndim == 3:
            y_data = y_data[:, land_mask]


        # Create windowed sequences strictly inside this split boundary
        X_seq, y_seq = create_sliding_window_sequences(
            X_data=X_data,
            y_data=y_data,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )

        datasets[split_name] = ClimateSequenceDataset(X_seq, y_seq)
        print(f"[OK] {split_name.capitalize()} Sequence Dataset created: {len(datasets[split_name]):,} sequences (shape: {X_seq.shape}).")

    return datasets


def create_sequence_dataloaders(
    datasets: Dict[str, ClimateSequenceDataset],
    batch_size: int = 256,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, val, and test sequence datasets.

    Parameters
    ----------
    datasets : Dict[str, ClimateSequenceDataset]
        Sequence dataset dictionary.
    batch_size : int, default=256
        Batch size.
    num_workers : int, default=0
        Subprocess worker count for data loading.

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary of PyTorch DataLoaders ('train', 'val', 'test').
    """
    dataloaders = {}
    for split_name, dataset in datasets.items():
        # Shuffle ONLY for training set; keep validation and test sequential
        shuffle = (split_name == "train")
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    return dataloaders


if __name__ == "__main__":
    print("=" * 60)
    print("SEQUENCE DATASET CREATION UNIT TEST")
    print("=" * 60)

    # Synthetic test: 100 days, 10 cells, 5 features
    T, N, F = 100, 10, 5
    seq_len = 7
    X_syn = np.random.randn(T, N, F)
    y_syn = np.random.randn(T, N)

    X_seq, y_seq = create_sliding_window_sequences(X_syn, y_syn, sequence_length=seq_len, prediction_horizon=1)
    
    print(f"Input Synthetic Shape: {X_syn.shape}")
    print(f"Generated Sequence X Shape: {X_seq.shape}  (Expected: ({N*(T-seq_len)}, {seq_len}, {F}))")
    print(f"Generated Target y Shape:   {y_seq.shape}  (Expected: ({N*(T-seq_len)},))")

    dataset = ClimateSequenceDataset(X_seq, y_seq)
    sample_x, sample_y = dataset[0]
    print(f"Dataset Sample 0 -> x: {sample_x.shape}, y: {sample_y.item():.4f}")
    print("=" * 60)
