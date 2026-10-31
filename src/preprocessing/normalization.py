"""
Normalization Module
--------------------
Provides reusable feature scaling and standardization utilities for climate data using scikit-learn:
- StandardScaler (Zero mean, unit variance)
- MinMaxScaler ([0, 1] or custom range scaling)

Features:
1. `analyze_scaler_suitability`: Evaluates skewness, zero-inflation, and distribution stats to recommend
   whether StandardScaler or MinMaxScaler is appropriate per climate parameter.
2. `fit_dataset_scaler`: Fits scikit-learn scalers strictly on training data (preventing data leakage).
3. `transform_dataset_scaler`: Transforms validation and test datasets using pre-fitted scalers.
4. `inverse_transform_dataset_scaler`: Reverts scaled predictions back to original physical units (°C, mm/day).
5. `save_scaler` & `load_scaler`: Serializes fitted scalers using joblib for Phase 4 model inference.

Does NOT modify raw datasets.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List, Union
import logging
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_scaler_suitability(
    ds: xr.Dataset,
    var_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates distribution statistics (min, max, mean, std, skewness, zero percentage)
    for each climate variable to recommend whether StandardScaler or MinMaxScaler is more appropriate.

    Recommendation Principles:
    - Temperature parameters (Tmax, Tmin, Tmean, DTR): Symmetric, approximately Gaussian distribution -> StandardScaler recommended.
    - Precipitation parameters (Rainfall, Cumulative Rain): Heavily zero-inflated, right-skewed distribution -> MinMaxScaler (or Log + StandardScaler) recommended.
    - Binary Flags (heatwave_flag, dry_day_flag): Binary [0, 1] -> Scaling not required.

    Parameters:
        ds (xr.Dataset): Input climate dataset.
        var_names (list, optional): List of variable names to analyze. Defaults to all data variables.

    Returns:
        dict: Detailed distribution suitability report and recommendations per variable.
    """
    target_vars = var_names if var_names is not None else list(ds.data_vars.keys())
    analysis: Dict[str, Any] = {"variables": {}}

    for v_name in target_vars:
        arr = ds[v_name].values
        valid_vals = arr[~np.isnan(arr)]

        if len(valid_vals) == 0:
            continue

        v_min = float(np.min(valid_vals))
        v_max = float(np.max(valid_vals))
        v_mean = float(np.mean(valid_vals))
        v_std = float(np.std(valid_vals))

        # Calculate skewness and zero percentage
        zero_pct = float(round((np.isclose(valid_vals, 0.0, atol=1e-3).sum() / len(valid_vals)) * 100, 2))
        
        # Pearson's median skewness coefficient: 3 * (mean - median) / std
        v_median = float(np.median(valid_vals))
        skewness = float(round(3.0 * (v_mean - v_median) / v_std, 3)) if v_std > 1e-6 else 0.0

        # Recommendation logic
        if v_name.endswith("_flag") or (v_min >= 0.0 and v_max <= 1.0 and len(np.unique(valid_vals)) <= 2):
            recommendation = "None (Binary Indicator)"
            reason = "Variable is a binary flag [0, 1]. Feature scaling is not required."
        elif "rain" in v_name.lower() or "rf" in v_name.lower() or "precip" in v_name.lower() or zero_pct > 30.0:
            recommendation = "MinMaxScaler"
            reason = f"Right-skewed / zero-inflated distribution ({zero_pct}% zeros, skewness={skewness}). MinMaxScaler preserves non-negativity and bounds."
        else:
            recommendation = "StandardScaler"
            reason = f"Continuous symmetric climate field (skewness={skewness}). StandardScaler centers distribution at zero mean and unit variance."

        analysis["variables"][v_name] = {
            "min": round(v_min, 3),
            "max": round(v_max, 3),
            "mean": round(v_mean, 3),
            "std": round(v_std, 3),
            "median": round(v_median, 3),
            "zero_percentage": zero_pct,
            "skewness": skewness,
            "recommended_scaler": recommendation,
            "justification": reason
        }

    return analysis


def fit_dataset_scaler(
    train_ds: xr.Dataset,
    scaler_type: str = "standard",
    var_names: Optional[List[str]] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0)
) -> Dict[str, Any]:
    """
    Fits scikit-learn scalers strictly on training dataset variables (preventing data leakage).

    Parameters:
        train_ds (xr.Dataset): Training xarray Dataset.
        scaler_type (str): 'standard' (StandardScaler) or 'minmax' (MinMaxScaler).
        var_names (list, optional): Variables to scale. Defaults to continuous non-flag variables.
        feature_range (tuple): Target scale range for MinMaxScaler (default: (0, 1)).

    Returns:
        dict: Fitted scaler registry dictionary containing fitted sklearn scaler instances and metadata.
    """
    target_vars = var_names if var_names is not None else [v for v in train_ds.data_vars if not v.endswith("_flag")]
    scaler_registry: Dict[str, Any] = {
        "scaler_type": scaler_type.lower(),
        "feature_range": feature_range,
        "scalers": {},
        "variable_stats": {}
    }

    for v_name in target_vars:
        if v_name not in train_ds.data_vars:
            continue

        arr = train_ds[v_name].values
        valid_mask = ~np.isnan(arr)
        valid_data = arr[valid_mask].reshape(-1, 1)

        if len(valid_data) == 0:
            logger.warning(f"Skipping scaler fitting for '{v_name}': all values are NaN.")
            continue

        if scaler_type.lower() in ("standard", "standardscaler", "zscore"):
            scaler = StandardScaler()
            scaler.fit(valid_data)
            stats = {"mean": float(scaler.mean_[0]), "std": float(np.sqrt(scaler.var_[0]))}
        elif scaler_type.lower() in ("minmax", "minmaxscaler"):
            scaler = MinMaxScaler(feature_range=feature_range)
            scaler.fit(valid_data)
            stats = {"min": float(scaler.data_min_[0]), "max": float(scaler.data_max_[0])}
        else:
            raise ValueError(f"❌ Unsupported scaler_type '{scaler_type}'. Choose 'standard' or 'minmax'.")

        scaler_registry["scalers"][v_name] = scaler
        scaler_registry["variable_stats"][v_name] = stats

    logger.info(f"Fitted {scaler_type.upper()} scalers on training data for {len(scaler_registry['scalers'])} variables.")
    return scaler_registry


def transform_dataset_scaler(
    ds: xr.Dataset,
    scaler_registry: Dict[str, Any]
) -> xr.Dataset:
    """
    Transforms validation/test dataset variables using pre-fitted scalers.

    Guarantees Zero Data Leakage:
    - Does NOT compute statistics on the input dataset. Applies only fitted scalers.

    Parameters:
        ds (xr.Dataset): Input dataset (validation or test set).
        scaler_registry (dict): Pre-fitted scaler registry from `fit_dataset_scaler`.

    Returns:
        xr.Dataset: Transformed xarray Dataset.
    """
    ds_scaled = ds.copy(deep=True)
    scalers = scaler_registry.get("scalers", {})
    scaler_type = scaler_registry.get("scaler_type", "custom")

    for v_name, scaler in scalers.items():
        if v_name not in ds_scaled.data_vars:
            continue

        arr = ds_scaled[v_name].values
        orig_shape = arr.shape
        flat_arr = arr.reshape(-1, 1)

        valid_mask = ~np.isnan(flat_arr)
        scaled_flat = np.full_like(flat_arr, np.nan, dtype=np.float32)

        if valid_mask.any():
            scaled_valid = scaler.transform(flat_arr[valid_mask].reshape(-1, 1))
            scaled_flat[valid_mask] = scaled_valid.flatten()

        scaled_arr = scaled_flat.reshape(orig_shape)

        ds_scaled[v_name] = (ds_scaled[v_name].dims, scaled_arr, ds_scaled[v_name].attrs)
        ds_scaled[v_name].attrs["scaling_method"] = scaler_type

    return ds_scaled


def fit_transform_dataset_scaler(
    train_ds: xr.Dataset,
    scaler_type: str = "standard",
    var_names: Optional[List[str]] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[xr.Dataset, Dict[str, Any]]:
    """
    Convenience function: Fits scalers on training dataset and transforms it in one call.

    Parameters:
        train_ds (xr.Dataset): Training xarray Dataset.
        scaler_type (str): 'standard' or 'minmax'.
        var_names (list, optional): Target variables.
        feature_range (tuple): Target range for MinMaxScaler.

    Returns:
        tuple: (scaled_train_ds, scaler_registry)
    """
    registry = fit_dataset_scaler(train_ds, scaler_type=scaler_type, var_names=var_names, feature_range=feature_range)
    scaled_ds = transform_dataset_scaler(train_ds, registry)
    return scaled_ds, registry


def inverse_transform_dataset_scaler(
    scaled_ds: xr.Dataset,
    scaler_registry: Dict[str, Any]
) -> xr.Dataset:
    """
    Reverts scaled dataset variables back to original physical units (°C, mm/day).

    Parameters:
        scaled_ds (xr.Dataset): Scaled xarray Dataset.
        scaler_registry (dict): Pre-fitted scaler registry.

    Returns:
        xr.Dataset: Dataset restored to original physical units.
    """
    ds_orig = scaled_ds.copy(deep=True)
    scalers = scaler_registry.get("scalers", {})

    for v_name, scaler in scalers.items():
        if v_name not in ds_orig.data_vars:
            continue

        arr = ds_orig[v_name].values
        orig_shape = arr.shape
        flat_arr = arr.reshape(-1, 1)

        valid_mask = ~np.isnan(flat_arr)
        restored_flat = np.full_like(flat_arr, np.nan, dtype=np.float32)

        if valid_mask.any():
            restored_valid = scaler.inverse_transform(flat_arr[valid_mask].reshape(-1, 1))
            restored_flat[valid_mask] = restored_valid.flatten()

        restored_arr = restored_flat.reshape(orig_shape)
        ds_orig[v_name] = (ds_orig[v_name].dims, restored_arr, ds_orig[v_name].attrs)
        if "scaling_method" in ds_orig[v_name].attrs:
            del ds_orig[v_name].attrs["scaling_method"]

    return ds_orig


def save_scaler(scaler_registry: Dict[str, Any], file_path: Union[str, Path]) -> Path:
    """
    Saves fitted scaler registry to disk using joblib for later model inference.

    Parameters:
        scaler_registry (dict): Scaler registry to save.
        file_path (str | Path): Destination file path (e.g., 'data/processed/imd/climate_scaler.joblib').

    Returns:
        Path: Absolute path to saved scaler file.
    """
    out_path = Path(file_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler_registry, out_path)
    logger.info(f"Saved fitted climate scaler registry to: {out_path}")
    return out_path


def load_scaler(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a fitted scaler registry from disk.

    Parameters:
        file_path (str | Path): Path to saved scaler joblib file.

    Returns:
        dict: Fitted scaler registry.
    """
    in_path = Path(file_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"❌ Scaler file not found at: '{in_path}'")

    registry = joblib.load(in_path)
    logger.info(f"Loaded climate scaler registry from: {in_path}")
    return registry


# Backward compatibility wrappers
def normalize_min_max(
    ds: xr.Dataset,
    var_names: Optional[List[str]] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[xr.Dataset, Dict[str, Dict[str, float]]]:
    """Wrapper function preserving backward compatibility."""
    scaled_ds, registry = fit_transform_dataset_scaler(ds, scaler_type="minmax", var_names=var_names, feature_range=feature_range)
    return scaled_ds, registry["variable_stats"]


def normalize_z_score(
    ds: xr.Dataset,
    var_names: Optional[List[str]] = None,
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[xr.Dataset, Dict[str, Dict[str, float]]]:
    """Wrapper function preserving backward compatibility."""
    scaled_ds, registry = fit_transform_dataset_scaler(ds, scaler_type="standard", var_names=var_names)
    return scaled_ds, registry["variable_stats"]


def normalize_features(
    ds: xr.Dataset,
    method: str = "minmax",
    var_names: Optional[List[str]] = None,
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[xr.Dataset, Dict[str, Dict[str, float]]]:
    """Wrapper dispatcher preserving backward compatibility."""
    scaled_ds, registry = fit_transform_dataset_scaler(ds, scaler_type=method, var_names=var_names)
    return scaled_ds, registry["variable_stats"]


if __name__ == "__main__":
    print("=== Normalization Module Initialized ===")
    print("Available Functions:")
    print("  - analyze_scaler_suitability()")
    print("  - fit_dataset_scaler()")
    print("  - transform_dataset_scaler()")
    print("  - fit_transform_dataset_scaler()")
    print("  - inverse_transform_dataset_scaler()")
    print("  - save_scaler()")
    print("  - load_scaler()")
