"""
Reusable Prediction Engine for Deep Learning Climate Models (Phase 5).

Provides type-checked, validated prediction inference for:
1. PyTorch 1D Stacked LSTM Models (Rainfall & Temperature)
2. PyTorch 2D Spatio-Temporal ConvLSTM Models (Rainfall & Temperature)

Handles:
- Loading saved PyTorch checkpoints (.pt)
- Loading preprocessing/scaler metadata (climate_scaler_2020.joblib)
- Input sequence shape validation
- Running torch.no_grad() model forward pass
- Converting scaled model outputs back to physical units (mm/day, °C)
- Returning clean, structured DeepLearningPredictionResult objects
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.problem_definition import get_problem_definition
from src.deep_learning.lstm_model import build_rainfall_lstm_model, build_temperature_lstm_model, ClimateLSTMModel
from src.deep_learning.convlstm_model import build_rainfall_convlstm_model, build_temperature_convlstm_model, ClimateConvLSTMModel
from src.deep_learning.utils import get_device


# Default checkpoint file locations
DEFAULT_LSTM_RAINFALL_PATH = "models/deep_learning/rainfall/lstm_rainfall_best.pt"
DEFAULT_LSTM_TEMP_PATH = "models/deep_learning/temperature/lstm_temperature_max_best.pt"
DEFAULT_CONVLSTM_RAINFALL_PATH = "models/deep_learning/convlstm_rain_best.pt"
DEFAULT_CONVLSTM_TEMP_PATH = "models/deep_learning/convlstm_tmax_best.pt"

DEFAULT_SCALER_PATH = "data/processed/imd/climate_scaler_2020.joblib"


@dataclass
class DeepLearningPredictionResult:
    """Structured container for deep learning model prediction outputs."""
    task_name: str
    target_variable: str
    model_architecture: str
    unit: str
    predictions: np.ndarray  # Physical unscaled units
    predictions_scaled: np.ndarray  # Scaled model outputs
    sequence_length: int
    input_shape: Tuple[int, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts result object to a serializable dictionary."""
        return {
            "task_name": self.task_name,
            "target_variable": self.target_variable,
            "model_architecture": self.model_architecture,
            "unit": self.unit,
            "sequence_length": self.sequence_length,
            "input_shape": list(self.input_shape),
            "predictions_summary": {
                "mean": float(np.mean(self.predictions)),
                "std": float(np.std(self.predictions)),
                "min": float(np.min(self.predictions)),
                "max": float(np.max(self.predictions))
            },
            "predictions": self.predictions.tolist(),
            "metadata": self.metadata
        }


class DeepLearningPredictor:
    """
    Inference predictor wrapper for saved PyTorch deep learning climate models.
    """

    def __init__(
        self,
        checkpoint_path: str,
        task_key: str,
        architecture_type: str = "lstm",  # 'lstm' or 'convlstm'
        scaler_path: str = DEFAULT_SCALER_PATH
    ):
        """
        Parameters
        ----------
        checkpoint_path : str
            Path to saved PyTorch checkpoint (.pt).
        task_key : str
            Problem definition key ('rainfall' or 'temperature_max').
        architecture_type : str, default='lstm'
            Architecture family ('lstm' or 'convlstm').
        scaler_path : str, default=DEFAULT_SCALER_PATH
            Path to saved scaler metadata joblib file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler metadata file not found at: {scaler_path}")

        self.checkpoint_path = checkpoint_path
        self.task_key = task_key
        self.architecture_type = architecture_type.lower()
        self.device = get_device()

        self.config = get_problem_definition(task_key)
        self.scaler_info = joblib.load(scaler_path)

        # Reinstantiate model and load checkpoint
        self.model = self._load_model_checkpoint()
        self.model.eval()

    def _load_model_checkpoint(self) -> nn.Module:
        """Loads PyTorch model weights and config from checkpoint file."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Determine input_dim safely
        if "config" in checkpoint and "input_dim" in checkpoint["config"]:
            input_dim = checkpoint["config"]["input_dim"]
        else:
            input_dim = len(self.config.feature_columns)

        target_var = self.config.target_variable

        if self.architecture_type == "lstm":
            if target_var == "rain":
                model = build_rainfall_lstm_model(input_dim=input_dim, hidden_dim=64, num_layers=2)
            else:
                model = build_temperature_lstm_model(input_dim=input_dim, hidden_dim=64, num_layers=2)
        elif self.architecture_type == "convlstm":
            if target_var == "rain":
                model = build_rainfall_convlstm_model(input_dim=input_dim, hidden_dims=[32, 32], num_layers=2, kernel_size=3)
            else:
                model = build_temperature_convlstm_model(input_dim=input_dim, hidden_dims=[32, 32], num_layers=2, kernel_size=3)
        else:
            raise ValueError(f"Unsupported architecture type: {self.architecture_type}")

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        return model

    def unscale_predictions(self, y_scaled: np.ndarray) -> np.ndarray:
        """Converts normalized model predictions back to physical units."""
        target_var = self.config.target_variable

        if target_var == "rain":
            max_val = self.scaler_info["rain_scaler"]["variable_stats"]["rain"]["max"]
            min_val = self.scaler_info["rain_scaler"]["variable_stats"]["rain"]["min"]
            unscaled = y_scaled * (max_val - min_val) + min_val
            return np.clip(unscaled, 0.0, None)  # Rainfall cannot be negative
        elif target_var == "tmax":
            mean_val = self.scaler_info["temp_scaler"]["variable_stats"]["tmax"]["mean"]
            std_val = self.scaler_info["temp_scaler"]["variable_stats"]["tmax"]["std"]
            return y_scaled * std_val + mean_val
        else:
            return y_scaled

    def validate_input_sequence(self, input_sequence: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Validates input sequence tensor shape and data types.
        """
        if isinstance(input_sequence, np.ndarray):
            tensor_x = torch.tensor(input_sequence, dtype=torch.float32)
        elif isinstance(input_sequence, torch.Tensor):
            tensor_x = input_sequence.float()
        else:
            raise TypeError(f"Unsupported input sequence type: {type(input_sequence)}")

        # Add batch dimension if missing
        if self.architecture_type == "lstm" and tensor_x.ndim == 2:
            # (seq_len, input_dim) -> (1, seq_len, input_dim)
            tensor_x = tensor_x.unsqueeze(0)
        elif self.architecture_type == "convlstm" and tensor_x.ndim == 4:
            # (seq_len, C, H, W) -> (1, seq_len, C, H, W)
            tensor_x = tensor_x.unsqueeze(0)

        # Validate tensor dimensions
        if self.architecture_type == "lstm":
            if tensor_x.ndim != 3:
                raise ValueError(f"LSTM input sequence must be 3D (batch, seq_len, input_dim), got shape {tensor_x.shape}.")
            if tensor_x.size(-1) != len(self.config.feature_columns):
                raise ValueError(f"LSTM feature count mismatch: expected {len(self.config.feature_columns)}, got {tensor_x.size(-1)}.")
        elif self.architecture_type == "convlstm":
            if tensor_x.ndim != 5:
                raise ValueError(f"ConvLSTM input sequence must be 5D (batch, seq_len, channels, lat, lon), got shape {tensor_x.shape}.")
            if tensor_x.size(2) != len(self.config.feature_columns):
                raise ValueError(f"ConvLSTM channel count mismatch: expected {len(self.config.feature_columns)}, got {tensor_x.size(2)}.")

        return tensor_x

    def predict(
        self,
        input_sequence: Union[np.ndarray, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None
    ) -> DeepLearningPredictionResult:
        """
        Generates predictions for validated historical sequence input.
        """
        tensor_x = self.validate_input_sequence(input_sequence).to(self.device)

        with torch.no_grad():
            preds_scaled_tensor = self.model(tensor_x)
            preds_scaled = preds_scaled_tensor.cpu().numpy()

        preds_physical = self.unscale_predictions(preds_scaled)
        unit_str = "mm/day" if self.config.target_variable == "rain" else "°C"

        return DeepLearningPredictionResult(
            task_name=self.config.task_name,
            target_variable=self.config.target_variable,
            model_architecture=f"PyTorch_{self.architecture_type.upper()}_Model",
            unit=unit_str,
            predictions=preds_physical,
            predictions_scaled=preds_scaled,
            sequence_length=tensor_x.size(1),
            input_shape=tuple(tensor_x.shape),
            metadata=metadata or {}
        )


# Global cached deep learning predictor instances
_dl_predictor_cache: Dict[str, DeepLearningPredictor] = {}


def predict_lstm_rainfall(
    input_sequence: Union[np.ndarray, torch.Tensor],
    checkpoint_path: str = DEFAULT_LSTM_RAINFALL_PATH,
    metadata: Optional[Dict[str, Any]] = None
) -> DeepLearningPredictionResult:
    """
    Convenience function for 1D LSTM Rainfall prediction.
    """
    cache_key = f"lstm_rain_{checkpoint_path}"
    if cache_key not in _dl_predictor_cache:
        _dl_predictor_cache[cache_key] = DeepLearningPredictor(checkpoint_path, "rainfall", architecture_type="lstm")
    return _dl_predictor_cache[cache_key].predict(input_sequence, metadata=metadata)


def predict_lstm_temperature(
    input_sequence: Union[np.ndarray, torch.Tensor],
    checkpoint_path: str = DEFAULT_LSTM_TEMP_PATH,
    metadata: Optional[Dict[str, Any]] = None
) -> DeepLearningPredictionResult:
    """
    Convenience function for 1D LSTM Maximum Temperature prediction.
    """
    cache_key = f"lstm_tmax_{checkpoint_path}"
    if cache_key not in _dl_predictor_cache:
        _dl_predictor_cache[cache_key] = DeepLearningPredictor(checkpoint_path, "temperature_max", architecture_type="lstm")
    return _dl_predictor_cache[cache_key].predict(input_sequence, metadata=metadata)


def predict_convlstm_rainfall(
    input_sequence: Union[np.ndarray, torch.Tensor],
    checkpoint_path: str = DEFAULT_CONVLSTM_RAINFALL_PATH,
    metadata: Optional[Dict[str, Any]] = None
) -> DeepLearningPredictionResult:
    """
    Convenience function for 2D Spatio-Temporal ConvLSTM Rainfall prediction.
    """
    cache_key = f"convlstm_rain_{checkpoint_path}"
    if cache_key not in _dl_predictor_cache:
        _dl_predictor_cache[cache_key] = DeepLearningPredictor(checkpoint_path, "rainfall", architecture_type="convlstm")
    return _dl_predictor_cache[cache_key].predict(input_sequence, metadata=metadata)


def predict_convlstm_temperature(
    input_sequence: Union[np.ndarray, torch.Tensor],
    checkpoint_path: str = DEFAULT_CONVLSTM_TEMP_PATH,
    metadata: Optional[Dict[str, Any]] = None
) -> DeepLearningPredictionResult:
    """
    Convenience function for 2D Spatio-Temporal ConvLSTM Maximum Temperature prediction.
    """
    cache_key = f"convlstm_tmax_{checkpoint_path}"
    if cache_key not in _dl_predictor_cache:
        _dl_predictor_cache[cache_key] = DeepLearningPredictor(checkpoint_path, "temperature_max", architecture_type="convlstm")
    return _dl_predictor_cache[cache_key].predict(input_sequence, metadata=metadata)


if __name__ == "__main__":
    print("=" * 60)
    print("DEEP LEARNING PREDICTION ENGINE UNIT TEST")
    print("=" * 60)

    # 1. Test 1D LSTM Rainfall Inference (Batch=4, Seq=7, Features=25)
    dummy_lstm_rain_input = np.random.rand(4, 7, 25).astype(np.float32)
    res_lstm_rain = predict_lstm_rainfall(dummy_lstm_rain_input)
    print(f"[OK] LSTM Rainfall Prediction Success!")
    print(f"     Target Unit: {res_lstm_rain.unit}")
    print(f"     Predictions Shape: {res_lstm_rain.predictions.shape}")
    print(f"     Summary: {res_lstm_rain.to_dict()['predictions_summary']}")

    # 2. Test 1D LSTM Temperature Inference (Batch=4, Seq=7, Features=21)
    dummy_lstm_temp_input = np.random.rand(4, 7, 21).astype(np.float32)
    res_lstm_tmax = predict_lstm_temperature(dummy_lstm_temp_input)
    print(f"\n[OK] LSTM Temperature Prediction Success!")
    print(f"     Target Unit: {res_lstm_tmax.unit}")
    print(f"     Predictions Shape: {res_lstm_tmax.predictions.shape}")
    print(f"     Summary: {res_lstm_tmax.to_dict()['predictions_summary']}")

    # 3. Test 2D ConvLSTM Temperature Inference (Batch=1, Seq=7, Channels=21, H=129, W=135)
    dummy_conv_temp_input = np.random.rand(1, 7, 21, 129, 135).astype(np.float32)
    res_conv_tmax = predict_convlstm_temperature(dummy_conv_temp_input)
    print(f"\n[OK] ConvLSTM Temperature Prediction Success!")
    print(f"     Target Unit: {res_conv_tmax.unit}")
    print(f"     Predictions Shape: {res_conv_tmax.predictions.shape}")
    print(f"     Summary: {res_conv_tmax.to_dict()['predictions_summary']}")
    print("=" * 60)
