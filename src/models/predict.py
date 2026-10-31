"""
Reusable Prediction Engine for Climate Models (Phase 4).

Provides robust, type-checked prediction inference for:
1. Rainfall Prediction Models
2. Maximum Temperature Prediction Models

Supports input validation, feature presence checks, value range checks,
and returns predictions in a clean, structured PredictionResult format.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import os
import numpy as np
import pandas as pd

from src.models.baseline_models import BaseClimateModel
from src.models.problem_definition import get_problem_definition, PredictionProblemConfig


# Default model file locations
DEFAULT_RAINFALL_MODEL_PATH = "models/rainfall/rf_rainfall_baseline.joblib"
DEFAULT_TEMPERATURE_MODEL_PATH = "models/temperature/rf_temperature_max_baseline.joblib"


@dataclass
class PredictionResult:
    """Structured container for model prediction outputs."""
    task_name: str
    target_variable: str
    predictions: np.ndarray
    num_samples: int
    model_type: str
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts prediction array to a pandas DataFrame."""
        df = pd.DataFrame({
            f"predicted_{self.target_variable}": self.predictions
        })
        return df

    def to_dict(self) -> Dict[str, Any]:
        """Converts result object to a serializable dictionary."""
        return {
            "task_name": self.task_name,
            "target_variable": self.target_variable,
            "num_samples": self.num_samples,
            "model_type": self.model_type,
            "predictions_summary": {
                "mean": float(np.mean(self.predictions)),
                "std": float(np.std(self.predictions)),
                "min": float(np.min(self.predictions)),
                "max": float(np.max(self.predictions))
            },
            "predictions": self.predictions.tolist(),
            "metadata": self.metadata
        }


class ClimatePredictor:
    """
    Inference predictor wrapper for saved baseline climate models.
    """

    def __init__(self, model_path: str, task_key: str):
        """
        Parameters
        ----------
        model_path : str
            Path to serialized joblib model wrapper file.
        task_key : str
            Problem definition task key ('rainfall' or 'temperature_max').
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        self.model_path = model_path
        self.task_key = task_key
        self.config: PredictionProblemConfig = get_problem_definition(task_key)
        self.model_wrapper: BaseClimateModel = BaseClimateModel.load_model(model_path)
        self.required_features: List[str] = self.model_wrapper.feature_names or self.config.feature_columns

    def validate_input(self, input_data: Union[pd.DataFrame, Dict[str, Any], np.ndarray]) -> pd.DataFrame:
        """
        Validates presence of required features and numeric bounds on input data.

        Parameters
        ----------
        input_data : pd.DataFrame, dict, or np.ndarray
            Input payload containing feature measurements.

        Returns
        -------
        pd.DataFrame
            Validated and re-ordered DataFrame containing required feature columns.
        """
        if isinstance(input_data, dict):
            # Convert dictionary of lists or scalars to DataFrame
            df = pd.DataFrame(input_data if isinstance(next(iter(input_data.values())), list) else [input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            if input_data.shape[1] != len(self.required_features):
                raise ValueError(
                    f"Numpy array shape {input_data.shape} does not match required feature count ({len(self.required_features)})."
                )
            df = pd.DataFrame(input_data, columns=self.required_features)
        else:
            raise TypeError(f"Unsupported input type for prediction: {type(input_data)}")

        # 1. Feature presence check
        missing_cols = [c for c in self.required_features if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Input data is missing required feature columns for '{self.task_key}': {missing_cols}")

        # Re-order columns to match model training order
        df_feats = df[self.required_features].copy()

        # 2. Check for unexpected NaNs or Infs
        if df_feats.isna().any().any():
            nan_cols = df_feats.columns[df_feats.isna().any()].tolist()
            raise ValueError(f"Input features contain missing/NaN values in columns: {nan_cols}")

        # 3. Domain range bounds checks
        # Rainfall lags should be non-negative
        for col in ["rain_lag1", "rain_roll_3d_mean", "rain_roll_7d_mean", "rain_roll_3d_sum", "rain_roll_7d_sum"]:
            if col in df_feats.columns and (df_feats[col] < 0).any():
                raise ValueError(f"Input column '{col}' contains negative rainfall values.")

        # Temperature values should be within physical atmospheric range (-50°C to 60°C)
        temp_cols = [c for c in df_feats.columns if "tmax" in c or "tmin" in c or "tmean" in c]
        for col in temp_cols:
            if (df_feats[col] < -50.0).any() or (df_feats[col] > 60.0).any():
                raise ValueError(f"Input column '{col}' contains unphysical temperature values outside [-50°C, 60°C].")

        return df_feats

    def predict(
        self,
        input_data: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """
        Generates predictions for validated input features.

        Parameters
        ----------
        input_data : pd.DataFrame, dict, or np.ndarray
            Input feature matrix or dict.
        metadata : dict, optional
            Additional contextual metadata (e.g., coordinates, timestamps).

        Returns
        -------
        PredictionResult
            Structured prediction output container.
        """
        df_valid = self.validate_input(input_data)
        preds = self.model_wrapper.predict(df_valid)

        # Post-processing: rainfall predictions cannot be negative
        if self.config.target_variable == "rain":
            preds = np.clip(preds, 0.0, None)

        return PredictionResult(
            task_name=self.config.task_name,
            target_variable=self.config.target_variable,
            predictions=preds,
            num_samples=len(preds),
            model_type=self.model_wrapper.model_type,
            feature_names=self.required_features,
            metadata=metadata or {}
        )


# Global cached predictors for fast inference
_predictor_cache: Dict[str, ClimatePredictor] = {}


def load_rainfall_model(filepath: str = DEFAULT_RAINFALL_MODEL_PATH) -> ClimatePredictor:
    """Loads and caches the rainfall prediction model."""
    if "rainfall" not in _predictor_cache or _predictor_cache["rainfall"].model_path != filepath:
        _predictor_cache["rainfall"] = ClimatePredictor(model_path=filepath, task_key="rainfall")
    return _predictor_cache["rainfall"]


def load_temperature_model(filepath: str = DEFAULT_TEMPERATURE_MODEL_PATH) -> ClimatePredictor:
    """Loads and caches the maximum temperature prediction model."""
    if "temperature_max" not in _predictor_cache or _predictor_cache["temperature_max"].model_path != filepath:
        _predictor_cache["temperature_max"] = ClimatePredictor(model_path=filepath, task_key="temperature_max")
    return _predictor_cache["temperature_max"]


def predict_rainfall(
    input_data: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
    model_path: str = DEFAULT_RAINFALL_MODEL_PATH,
    metadata: Optional[Dict[str, Any]] = None
) -> PredictionResult:
    """
    Convenience function to predict short-term daily rainfall.
    """
    predictor = load_rainfall_model(filepath=model_path)
    return predictor.predict(input_data, metadata=metadata)


def predict_temperature(
    input_data: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
    model_path: str = DEFAULT_TEMPERATURE_MODEL_PATH,
    metadata: Optional[Dict[str, Any]] = None
) -> PredictionResult:
    """
    Convenience function to predict short-term daily maximum temperature.
    """
    predictor = load_temperature_model(filepath=model_path)
    return predictor.predict(input_data, metadata=metadata)


if __name__ == "__main__":
    print("=" * 60)
    print("REUSABLE PREDICTION ENGINE UNIT TEST")
    print("=" * 60)

    # 1. Test sample dictionary input for Rainfall
    sample_rain_dict = {
        "rain_lag1": [0.0, 12.5],
        "tmax_lag1": [32.0, 28.5],
        "tmin_lag1": [22.0, 20.0],
        "dtr_lag1": [10.0, 8.5],
        "tmean_lag1": [27.0, 24.25],
        "rain_roll_3d_mean": [0.0, 4.2],
        "rain_roll_3d_sum": [0.0, 12.6],
        "rain_roll_7d_mean": [1.2, 5.0],
        "rain_roll_7d_sum": [8.4, 35.0],
        "tmax_roll_3d_mean": [31.5, 29.0],
        "tmax_roll_7d_mean": [31.0, 29.5],
        "tmin_roll_3d_mean": [21.5, 19.5],
        "tmin_roll_7d_mean": [21.0, 19.0],
        "dtr_roll_3d_mean": [10.0, 9.5],
        "dtr_roll_7d_mean": [10.0, 10.5],
        "tmean_roll_3d_mean": [26.5, 24.25],
        "tmean_roll_7d_mean": [26.0, 24.25],
        "rainy_day_flag": [0.0, 1.0],
        "heavy_rain_flag": [0.0, 0.0],
        "dry_day_flag": [1.0, 0.0],
        "month_sin": [0.5, 0.5],
        "month_cos": [0.866, 0.866],
        "doy_sin": [0.2, 0.2],
        "doy_cos": [0.98, 0.98],
        "season": [1.0, 1.0]
    }

    res_rain = predict_rainfall(sample_rain_dict)
    print(f"[OK] Rainfall Prediction Result: {res_rain.predictions}")
    print(f"     Summary: {res_rain.to_dict()['predictions_summary']}")

    # 2. Test sample DataFrame input for Temperature
    sample_tmax_dict = {
        "tmax_lag1": [33.5, 34.0],
        "tmin_lag1": [22.0, 23.0],
        "dtr_lag1": [11.5, 11.0],
        "tmean_lag1": [27.75, 28.5],
        "rain_lag1": [0.0, 0.0],
        "tmax_roll_3d_mean": [33.0, 33.5],
        "tmax_roll_7d_mean": [32.5, 33.0],
        "tmin_roll_3d_mean": [21.5, 22.0],
        "tmin_roll_7d_mean": [21.0, 21.5],
        "dtr_roll_3d_mean": [11.5, 11.5],
        "dtr_roll_7d_mean": [11.5, 11.5],
        "tmean_roll_3d_mean": [27.25, 27.75],
        "tmean_roll_7d_mean": [26.75, 27.25],
        "rain_roll_3d_mean": [0.0, 0.0],
        "rain_roll_7d_mean": [0.0, 0.0],
        "heatwave_flag": [0.0, 0.0],
        "month_sin": [0.5, 0.5],
        "month_cos": [0.866, 0.866],
        "doy_sin": [0.2, 0.2],
        "doy_cos": [0.98, 0.98],
        "season": [1.0, 1.0]
    }

    df_tmax_sample = pd.DataFrame(sample_tmax_dict)
    res_tmax = predict_temperature(df_tmax_sample)
    print(f"[OK] Temperature Prediction Result: {res_tmax.predictions}")
    print(f"     Summary: {res_tmax.to_dict()['predictions_summary']}")
    print("=" * 60)
