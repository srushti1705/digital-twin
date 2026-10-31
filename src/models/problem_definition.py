"""
Problem Definition Module for Phase 4 AI/ML Prediction Models.

Defines formal problem formulations for:
1. Rainfall Prediction (Next-day precipitation regression/forecasting)
2. Temperature Prediction (Next-day daily maximum and minimum temperature forecasting)

Uses actual processed IMD dataset features without inventing non-existent variables.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import xarray as xr


# Exact feature column groups present in data/processed/imd/india_climate_2020_aligned.nc
BASE_CLIMATE_FEATURES: List[str] = [
    "rain", "tmax", "tmin", "dtr", "tmean"
]

PHYSICAL_FLAG_FEATURES: List[str] = [
    "heatwave_flag", "rainy_day_flag", "heavy_rain_flag", "dry_day_flag"
]

LAG1_FEATURES: List[str] = [
    "rain_lag1", "tmax_lag1", "tmin_lag1", "dtr_lag1", "tmean_lag1"
]

ROLLING_FEATURES: List[str] = [
    "rain_roll_3d_mean", "rain_roll_3d_sum", "rain_roll_7d_mean", "rain_roll_7d_sum",
    "tmax_roll_3d_mean", "tmax_roll_7d_mean",
    "tmin_roll_3d_mean", "tmin_roll_7d_mean",
    "dtr_roll_3d_mean", "dtr_roll_7d_mean",
    "tmean_roll_3d_mean", "tmean_roll_7d_mean"
]

TEMPORAL_ENCODING_FEATURES: List[str] = [
    "year", "month", "day", "day_of_year", "season",
    "month_sin", "month_cos", "doy_sin", "doy_cos"
]


@dataclass
class PredictionProblemConfig:
    """
    Configuration specification for a supervised learning prediction task.
    """
    task_name: str
    target_variable: str
    target_column: str
    prediction_horizon_days: int
    required_history_days: int
    feature_columns: List[str]
    task_type: str  # 'regression' or 'classification'
    primary_metric: str
    secondary_metrics: List[str]
    description: str

    def validate_features(self, available_variables: List[str]) -> bool:
        """
        Verify that all specified feature columns and target column exist in the dataset.
        """
        missing = [f for f in self.feature_columns if f not in available_variables]
        if missing:
            raise ValueError(f"Task '{self.task_name}' specifies features missing from dataset: {missing}")
        if self.target_variable not in available_variables:
            raise ValueError(f"Task '{self.task_name}' specifies target '{self.target_variable}' missing from dataset.")
        return True


# Task 1: Next-Day Rainfall Prediction
RAINFALL_PREDICTION_TASK = PredictionProblemConfig(
    task_name="rainfall_prediction_next_day",
    target_variable="rain",
    target_column="rain_target_t1",
    prediction_horizon_days=1,
    required_history_days=7,  # 7-day historical lookback to match rolling 7d features
    feature_columns=[
        # Lagged observations
        "rain_lag1", "tmax_lag1", "tmin_lag1", "dtr_lag1", "tmean_lag1",
        # Rolling historical context
        "rain_roll_3d_mean", "rain_roll_3d_sum", "rain_roll_7d_mean", "rain_roll_7d_sum",
        "tmax_roll_3d_mean", "tmax_roll_7d_mean", "tmin_roll_3d_mean", "tmin_roll_7d_mean",
        "dtr_roll_3d_mean", "dtr_roll_7d_mean", "tmean_roll_3d_mean", "tmean_roll_7d_mean",
        # Physical flag states
        "rainy_day_flag", "heavy_rain_flag", "dry_day_flag",
        # Temporal & harmonic encodings
        "month_sin", "month_cos", "doy_sin", "doy_cos", "season"
    ],
    task_type="regression",
    primary_metric="RMSE",
    secondary_metrics=["MAE", "R2", "Correlation"],
    description=(
        "Short-term 1-day ahead daily precipitation forecasting (mm/day) at 0.25° spatial grid resolution. "
        "Predicts rain at timestep (t+1) using historical climate predictors measured up to timestep t."
    )
)

# Task 2a: Next-Day Maximum Temperature Prediction
TEMPERATURE_MAX_PREDICTION_TASK = PredictionProblemConfig(
    task_name="temperature_max_prediction_next_day",
    target_variable="tmax",
    target_column="tmax_target_t1",
    prediction_horizon_days=1,
    required_history_days=7,
    feature_columns=[
        # Lagged observations
        "tmax_lag1", "tmin_lag1", "dtr_lag1", "tmean_lag1", "rain_lag1",
        # Rolling historical thermal context
        "tmax_roll_3d_mean", "tmax_roll_7d_mean",
        "tmin_roll_3d_mean", "tmin_roll_7d_mean",
        "dtr_roll_3d_mean", "dtr_roll_7d_mean",
        "tmean_roll_3d_mean", "tmean_roll_7d_mean",
        "rain_roll_3d_mean", "rain_roll_7d_mean",
        # Physical extreme flags
        "heatwave_flag",
        # Temporal & harmonic encodings
        "month_sin", "month_cos", "doy_sin", "doy_cos", "season"
    ],
    task_type="regression",
    primary_metric="RMSE",
    secondary_metrics=["MAE", "R2"],
    description=(
        "Short-term 1-day ahead daily maximum surface air temperature forecasting (°C) at 0.25° grid resolution. "
        "Predicts tmax at timestep (t+1) using thermal and precipitation history up to timestep t."
    )
)

# Task 2b: Next-Day Minimum Temperature Prediction
TEMPERATURE_MIN_PREDICTION_TASK = PredictionProblemConfig(
    task_name="temperature_min_prediction_next_day",
    target_variable="tmin",
    target_column="tmin_target_t1",
    prediction_horizon_days=1,
    required_history_days=7,
    feature_columns=[
        # Lagged observations
        "tmin_lag1", "tmax_lag1", "dtr_lag1", "tmean_lag1", "rain_lag1",
        # Rolling historical thermal context
        "tmin_roll_3d_mean", "tmin_roll_7d_mean",
        "tmax_roll_3d_mean", "tmax_roll_7d_mean",
        "dtr_roll_3d_mean", "dtr_roll_7d_mean",
        "tmean_roll_3d_mean", "tmean_roll_7d_mean",
        "rain_roll_3d_mean", "rain_roll_7d_mean",
        # Temporal & harmonic encodings
        "month_sin", "month_cos", "doy_sin", "doy_cos", "season"
    ],
    task_type="regression",
    primary_metric="RMSE",
    secondary_metrics=["MAE", "R2"],
    description=(
        "Short-term 1-day ahead daily minimum surface air temperature forecasting (°C) at 0.25° grid resolution. "
        "Predicts tmin at timestep (t+1) using thermal and precipitation history up to timestep t."
    )
)

# Global Task Registry
TASK_REGISTRY: Dict[str, PredictionProblemConfig] = {
    "rainfall": RAINFALL_PREDICTION_TASK,
    "temperature_max": TEMPERATURE_MAX_PREDICTION_TASK,
    "temperature_min": TEMPERATURE_MIN_PREDICTION_TASK
}


def get_problem_definition(task_name: str) -> PredictionProblemConfig:
    """
    Retrieve problem definition config by task key.
    
    Parameters
    ----------
    task_name : str
        One of 'rainfall', 'temperature_max', or 'temperature_min'.
        
    Returns
    -------
    PredictionProblemConfig
        Formal specification object for the requested task.
    """
    if task_name not in TASK_REGISTRY:
        raise KeyError(
            f"Task '{task_name}' not found in registry. "
            f"Available tasks: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_name]


def get_available_tasks() -> List[str]:
    """
    Returns list of registered task names.
    """
    return list(TASK_REGISTRY.keys())


def validate_all_tasks_against_dataset(nc_path: str = "data/processed/imd/india_climate_2020_aligned.nc") -> bool:
    """
    Validates all defined tasks against actual variables present in the NetCDF dataset.
    """
    ds = xr.open_dataset(nc_path)
    available_vars = list(ds.data_vars.keys()) + list(ds.coords.keys())
    
    for name, config in TASK_REGISTRY.items():
        config.validate_features(available_vars)
        print(f"[OK] Task '{name}' successfully validated against dataset ({len(config.feature_columns)} features).")
    
    return True


if __name__ == "__main__":
    validate_all_tasks_against_dataset()
