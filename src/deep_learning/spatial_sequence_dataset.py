"""
PyTorch Spatial Sequence Dataset Module for Spatio-Temporal ConvLSTM (Phase 5).

Converts 3D gridded xarray datasets into 5D supervised spatial-temporal sequence windows:
  Input Sequence (X): (num_samples, sequence_length, num_channels, height, width)
  Target Grid (y):   (num_samples, height, width)

Guarantees:
- Temporal order preservation with zero data leakage across split boundaries.
- Replaces NaNs with 0.0 in spatial feature maps.
- Computes 2D Land Mask matrix (1 for valid land grid cells, 0 for ocean/non-land).
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.data_split import split_processed_dataset


class SpatialClimateSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapper for 5D Spatial ConvLSTM Sequence Forecasting.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        land_mask: np.ndarray
    ):
        """
        Parameters
        ----------
        sequences : np.ndarray
            5D array of shape (num_samples, sequence_length, num_channels, height, width).
        targets : np.ndarray
            3D array of shape (num_samples, height, width).
        land_mask : np.ndarray
            2D boolean/float array of shape (height, width) indicating valid land cells.
        """
        if len(sequences) != len(targets):
            raise ValueError(
                f"Mismatch between sequences ({len(sequences)}) and targets ({len(targets)}) count."
            )
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.land_mask = torch.tensor(land_mask, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def build_spatial_feature_tensor(
    ds: xr.Dataset,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a 4D spatial feature tensor (time, channels, lat, lon) from xarray variables,
    replacing NaNs with 0.0 and generating a binary land mask.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset split.
    feature_cols : List[str]
        List of feature variable names.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        feature_tensor: (time, channels, lat, lon)
        land_mask: (lat, lon) boolean array (True for land)
    """
    time_len = len(ds["time"])
    lat_len = len(ds["lat"])
    lon_len = len(ds["lon"])

    # Extract 2D land mask from first time step of rain / tmax
    first_var = "rain" if "rain" in ds.data_vars else list(ds.data_vars.keys())[0]
    land_mask = ~np.isnan(ds[first_var].isel(time=0).values)  # shape: (129, 135)

    channels_list = []
    for col in feature_cols:
        vals = ds[col].values
        if vals.ndim == 3:  # (time, lat, lon)
            # Replace NaNs with 0.0 for convolution safety
            vals_clean = np.nan_to_num(vals, nan=0.0).astype(np.float32)
            channels_list.append(vals_clean)
        elif vals.ndim == 1:  # (time,) -> broadcast spatially across lat x lon canvas
            vals_clean = np.nan_to_num(vals, nan=0.0).astype(np.float32)
            # Tile (time, 1, 1) -> (time, lat, lon)
            broadcasted = np.tile(vals_clean[:, np.newaxis, np.newaxis], (1, lat_len, lon_len))
            channels_list.append(broadcasted)
        else:
            raise ValueError(f"Unsupported array dimension for feature '{col}': {vals.ndim}")

    # Stack along channel axis (axis=1) -> shape: (time, num_channels, lat, lon)
    feature_tensor = np.stack(channels_list, axis=1)
    return feature_tensor, land_mask


def create_spatial_sliding_windows(
    feature_tensor: np.ndarray,
    target_array: np.ndarray,
    sequence_length: int = 7,
    prediction_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 4D spatial feature tensor into 5D sliding window input/target sequence pairs.

    Parameters
    ----------
    feature_tensor : np.ndarray
        Shape (T_timesteps, C_channels, H_height, W_width).
    target_array : np.ndarray
        Shape (T_timesteps, H_height, W_width).
    sequence_length : int, default=7
        Number of lookback time steps (L).
    prediction_horizon : int, default=1
        Number of steps ahead to predict (1 = next day).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X_seq: (num_windows, sequence_length, C_channels, H_height, W_width)
        y_seq: (num_windows, H_height, W_width)
    """
    T_timesteps, C_channels, H_height, W_width = feature_tensor.shape
    num_windows = T_timesteps - sequence_length - prediction_horizon + 1

    if num_windows <= 0:
        return np.empty((0, sequence_length, C_channels, H_height, W_width)), np.empty((0, H_height, W_width))

    X_list = []
    y_list = []

    # Clean target array NaNs for loss stability
    target_clean = np.nan_to_num(target_array, nan=0.0).astype(np.float32)

    for t in range(num_windows):
        win_x = feature_tensor[t : t + sequence_length]  # (sequence_length, C, H, W)
        win_y = target_clean[t + sequence_length + prediction_horizon - 1]  # (H, W)

        X_list.append(win_x)
        y_list.append(win_y)

    X_seq = np.array(X_list, dtype=np.float32)
    y_seq = np.array(y_list, dtype=np.float32)

    return X_seq, y_seq


def build_split_spatial_sequence_datasets(
    ds_splits: Dict[str, xr.Dataset],
    feature_cols: List[str],
    target_col: str,
    sequence_length: int = 7,
    prediction_horizon: int = 1
) -> Tuple[Dict[str, SpatialClimateSequenceDataset], np.ndarray]:
    """
    Constructs 5D Spatial PyTorch Datasets for train, val, and test splits.

    Returns
    -------
    Tuple[Dict[str, SpatialClimateSequenceDataset], np.ndarray]
        Dictionary of spatial sequence datasets, and 2D land mask array.
    """
    datasets = {}
    land_mask = None

    for split_name, ds in ds_splits.items():
        feature_tensor, mask = build_spatial_feature_tensor(ds, feature_cols)
        if land_mask is None:
            land_mask = mask

        target_array = ds[target_col].values  # (time, lat, lon)

        X_seq, y_seq = create_spatial_sliding_windows(
            feature_tensor=feature_tensor,
            target_array=target_array,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )

        datasets[split_name] = SpatialClimateSequenceDataset(X_seq, y_seq, land_mask)
        print(f"[OK] {split_name.capitalize()} Spatial ConvLSTM Dataset: {len(datasets[split_name])} window sequences (shape: {X_seq.shape}).")

    return datasets, land_mask


def create_spatial_dataloaders(
    datasets: Dict[str, SpatialClimateSequenceDataset],
    batch_size: int = 8,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for 5D Spatial ConvLSTM datasets.
    """
    dataloaders = {}
    for split_name, dataset in datasets.items():
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
    print("SPATIAL CONVLSTM DATASET CREATION UNIT TEST")
    print("=" * 60)

    # Test loading normalized NetCDF dataset splits
    splits = split_processed_dataset("data/processed/imd/india_climate_2020_normalized.nc", warm_up_days=7)
    feat_cols = ["rain", "tmax", "tmin", "dtr", "tmean", "month_sin", "month_cos"]
    
    spatial_datasets, mask = build_split_spatial_sequence_datasets(
        ds_splits=splits,
        feature_cols=feat_cols,
        target_col="rain",
        sequence_length=7
    )

    dataloaders = create_spatial_dataloaders(spatial_datasets, batch_size=8)
    sample_x, sample_y = next(iter(dataloaders["train"]))
    
    print(f"\nDataLoader Sample Batch Test:")
    print(f"  - Land Mask Shape: {mask.shape} (Valid Land Points: {mask.sum()})")
    print(f"  - Batch X Shape:   {sample_x.shape}  (Expected: (8, 7, 7, 129, 135))")
    print(f"  - Batch Y Shape:   {sample_y.shape}  (Expected: (8, 129, 135))")
    print("=" * 60)
