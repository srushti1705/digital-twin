"""
Baseline Machine Learning Framework for Climate Prediction (Phase 4).

Provides reusable, scikit-learn compatible wrappers for:
1. Random Forest Regressor
2. XGBoost Regressor (conditional on xgboost installation)

Supports model creation, configurable hyperparameterization, training,
prediction, serialization (save), and deserialization (load).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# Optional XGBoost import handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False


def is_xgboost_available() -> bool:
    """Returns True if xgboost package is installed."""
    return XGBOOST_AVAILABLE


class BaseClimateModel(ABC):
    """
    Abstract Base Class for Phase 4 Baseline Climate Models.
    """

    def __init__(
        self,
        model_type: str,
        task_name: str = "general",
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        self.model_type = model_type
        self.task_name = task_name
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None
        self.is_fitted: bool = False

    @abstractmethod
    def _build_model(self) -> None:
        """Instantiates the underlying regressor with configured hyperparameters."""
        pass

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None,
        **kwargs
    ) -> "BaseClimateModel":
        """
        Fits the underlying ML model to the training dataset.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input feature matrix.
        y : pd.Series or np.ndarray
            Target values.
        feature_names : List[str], optional
            Names of input feature columns.
        target_name : str, optional
            Name of target variable.

        Returns
        -------
        BaseClimateModel
            Self (fitted estimator).
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        elif feature_names is not None:
            self.feature_names = feature_names

        if isinstance(y, pd.Series):
            self.target_name = y.name
        elif target_name is not None:
            self.target_name = target_name

        if self.model is None:
            self._build_model()

        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generates predictions for input feature matrix X.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            1D array of continuous predictions.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"Model '{self.model_type}' is not fitted yet. Call fit() before predict().")
        return self.model.predict(X)

    def save_model(self, filepath: str) -> str:
        """
        Serializes the model wrapper and underlying estimator to disk using joblib.

        Parameters
        ----------
        filepath : str
            Destination path (e.g., 'models/baselines/rf_rainfall.joblib').

        Returns
        -------
        str
            Saved file path.
        """
        if not self.is_fitted:
            print(f"[Warning] Saving an unfitted model to '{filepath}'.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[OK] Model '{self.model_type}' for task '{self.task_name}' saved to: {filepath}")
        return filepath

    @classmethod
    def load_model(cls, filepath: str) -> "BaseClimateModel":
        """
        Loads a serialized model wrapper from disk.

        Parameters
        ----------
        filepath : str
            Path to serialized model file.

        Returns
        -------
        BaseClimateModel
            Deserialized model object.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        loaded_model = joblib.load(filepath)
        print(f"[OK] Loaded model '{loaded_model.model_type}' for task '{loaded_model.task_name}' from: {filepath}")
        return loaded_model


class RandomForestClimateModel(BaseClimateModel):
    """
    Random Forest Regressor wrapper for Climate Prediction.
    """

    DEFAULT_HYPERPARAMS: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }

    def __init__(
        self,
        task_name: str = "general",
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        params = {**self.DEFAULT_HYPERPARAMS, **(hyperparameters or {})}
        super().__init__(model_type="random_forest", task_name=task_name, hyperparameters=params)
        self._build_model()

    def _build_model(self) -> None:
        self.model = RandomForestRegressor(**self.hyperparameters)


class XGBoostClimateModel(BaseClimateModel):
    """
    XGBoost Regressor wrapper for Climate Prediction.
    """

    DEFAULT_HYPERPARAMS: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }

    def __init__(
        self,
        task_name: str = "general",
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed in the environment. "
                "Install via 'pip install xgboost' to use XGBoostClimateModel."
            )
        params = {**self.DEFAULT_HYPERPARAMS, **(hyperparameters or {})}
        super().__init__(model_type="xgboost", task_name=task_name, hyperparameters=params)
        self._build_model()

    def _build_model(self) -> None:
        self.model = xgb.XGBRegressor(**self.hyperparameters)


def create_baseline_model(
    model_type: str,
    task_name: str = "general",
    hyperparameters: Optional[Dict[str, Any]] = None
) -> BaseClimateModel:
    """
    Factory function to instantiate baseline climate models.

    Parameters
    ----------
    model_type : str
        Either 'random_forest' (or 'rf') or 'xgboost' (or 'xgb').
    task_name : str, default='general'
        Task identifier ('rainfall', 'temperature_max', 'temperature_min').
    hyperparameters : Dict[str, Any], optional
        Custom hyperparameter overrides.

    Returns
    -------
    BaseClimateModel
        Instantiated model wrapper ready for fitting.
    """
    m_type = model_type.lower()
    if m_type in ["random_forest", "rf", "randomforest"]:
        return RandomForestClimateModel(task_name=task_name, hyperparameters=hyperparameters)
    elif m_type in ["xgboost", "xgb"]:
        return XGBoostClimateModel(task_name=task_name, hyperparameters=hyperparameters)
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Supported model types: 'random_forest', 'xgboost'"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE ML FRAMEWORK UNIT TEST")
    print("=" * 60)
    print(f"XGBoost Available: {XGBOOST_AVAILABLE}")
    
    # Test RF instantiation
    rf_model = create_baseline_model("random_forest", task_name="rainfall", hyperparameters={"n_estimators": 10})
    print(f"Created model: {rf_model.model_type} for task: {rf_model.task_name}")
    print(f"Hyperparameters: {rf_model.hyperparameters}")
    
    # Synthetic quick fit/predict test
    X_dummy = np.random.randn(50, 5)
    y_dummy = np.random.randn(50)
    rf_model.fit(X_dummy, y_dummy)
    preds = rf_model.predict(X_dummy[:5])
    print(f"Fit & Predict successful. Predictions shape: {preds.shape}")
    print("=" * 60)
