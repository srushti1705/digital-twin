"""Reproducible one-day-ahead IMD grid forecasting and evaluation."""

from __future__ import annotations

import calendar
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    from . import rainfall_api, temperature_api
except ImportError:
    import rainfall_api
    import temperature_api

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = Path(
    os.environ.get(
        "CLIMATE_MODEL_DIR",
        str(PROJECT_ROOT / "models" / "climate_forecasts"),
    )
).expanduser().resolve()
METRICS_DIRECTORY = Path(
    os.environ.get(
        "CLIMATE_METRICS_DIR",
        str(PROJECT_ROOT / "outputs" / "climate_forecasts"),
    )
).expanduser().resolve()
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
ClimateVariable = Literal["rainfall", "tmax", "tmin"]

VARIABLES: dict[str, dict[str, Any]] = {
    "rainfall": {
        "label": "Daily IMD rainfall",
        "units": "mm/day",
        "resolution": 0.25,
        "source": "India Meteorological Department",
        "description": "Daily total rainfall",
        "rows": rainfall_api.IMD_GRID_SHAPE[0],
        "columns": rainfall_api.IMD_GRID_SHAPE[1],
        "latitude_origin": rainfall_api.IMD_LATITUDE_ORIGIN,
        "longitude_origin": rainfall_api.IMD_LONGITUDE_ORIGIN,
        "missing_value": rainfall_api.IMD_MISSING_VALUE,
    },
    "tmax": {
        "label": "Daily IMD maximum temperature",
        "units": "°C",
        "resolution": 1.0,
        "source": "India Meteorological Department",
        "description": "Daily maximum temperature",
        "rows": temperature_api.GRID_SHAPE[0],
        "columns": temperature_api.GRID_SHAPE[1],
        "latitude_origin": temperature_api.LATITUDE_ORIGIN,
        "longitude_origin": temperature_api.LONGITUDE_ORIGIN,
        "missing_value": temperature_api.MISSING_VALUE,
    },
    "tmin": {
        "label": "Daily IMD minimum temperature",
        "units": "°C",
        "resolution": 1.0,
        "source": "India Meteorological Department",
        "description": "Daily minimum temperature",
        "rows": temperature_api.GRID_SHAPE[0],
        "columns": temperature_api.GRID_SHAPE[1],
        "latitude_origin": temperature_api.LATITUDE_ORIGIN,
        "longitude_origin": temperature_api.LONGITUDE_ORIGIN,
        "missing_value": temperature_api.MISSING_VALUE,
    },
}
FEATURE_NAMES = (
    "latest_observation",
    "three_day_mean",
    "seven_day_mean",
    "day_of_year_sin",
    "day_of_year_cos",
    "latitude",
    "longitude",
)
MIN_VALIDATION_RMSE_IMPROVEMENT = 0.01
MAX_VALIDATION_MAE_DEGRADATION = 0.01


@dataclass(frozen=True)
class ForecastGrid:
    variable: str
    initialization_date: str
    forecast_date: str
    run_type: str
    model_name: str
    units: str
    spatial_resolution_degrees: float
    latitudes: list[float]
    longitudes: list[float]
    values: list[list[float | None]]
    valid_cell_count: int
    missing_cell_count: int
    source: str
    training_period: dict[str, str]
    metadata: dict[str, Any]


def available_observation_dates(variable: str) -> list[str]:
    if variable == "rainfall":
        return rainfall_api._imd_available_dates()
    if variable in {"tmax", "tmin"}:
        return temperature_api._dates_for_product(variable)
    raise ValueError(f"Unknown climate variable: {variable}")


@lru_cache(maxsize=32)
def _read_observation(variable: str, observation_date: date) -> np.ndarray:
    if variable == "rainfall":
        files = rainfall_api._matching_year_files(observation_date.year)
        path = next((candidate for candidate in files if candidate.suffix.lower() == ".grd"), None)
        if path is None:
            raise FileNotFoundError(
                f"No annual IMD rainfall GRD is available for {observation_date.year}."
            )
        days = 366 if calendar.isleap(observation_date.year) else 365
        expected_bytes = (
            days
            * rainfall_api.IMD_GRID_SHAPE[0]
            * rainfall_api.IMD_GRID_SHAPE[1]
            * np.dtype("<f4").itemsize
        )
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"IMD rainfall file {path.name} has an unexpected byte size.")
        daily_grids = np.memmap(
            path,
            dtype="<f4",
            mode="r",
            shape=(days, *rainfall_api.IMD_GRID_SHAPE),
        )
        grid = np.array(daily_grids[observation_date.timetuple().tm_yday - 1], copy=True)
        grid[~np.isfinite(grid) | (grid == rainfall_api.IMD_MISSING_VALUE)] = np.nan
        return grid
    if variable in {"tmax", "tmin"}:
        paths = temperature_api._year_files(variable, observation_date.year)
        path = next((candidate for candidate in paths if candidate.suffix.lower() == ".grd"), None)
        if path is None:
            raise FileNotFoundError(
                f"No annual IMD {variable} GRD is available for {observation_date.year}."
            )
        grid, _, _ = temperature_api._read_binary_day(path, observation_date)
        return np.asarray(grid, dtype=np.float32)
    raise ValueError(f"Unknown climate variable: {variable}")


def _coordinate_grids(variable: str) -> tuple[np.ndarray, np.ndarray]:
    config = VARIABLES[variable]
    latitudes = (
        config["latitude_origin"]
        + np.arange(config["rows"], dtype=np.float32) * config["resolution"]
    )
    longitudes = (
        config["longitude_origin"]
        + np.arange(config["columns"], dtype=np.float32) * config["resolution"]
    )
    return np.meshgrid(latitudes, longitudes, indexing="ij")


def _feature_grid(
    history: np.ndarray,
    target_date: date,
    variable: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    config = VARIABLES[variable]
    if history.shape != (7, config["rows"], config["columns"]):
        raise ValueError("A forecast requires exactly seven consecutive native-grid observations.")

    valid_history = np.isfinite(history).all(axis=0)
    latest = history[-1]
    three_day_mean = history[-3:].mean(axis=0)
    seven_day_mean = history.mean(axis=0)
    day_angle = 2 * math.pi * (target_date.timetuple().tm_yday - 1) / (
        366 if calendar.isleap(target_date.year) else 365
    )
    latitude_grid, longitude_grid = _coordinate_grids(variable)

    features = np.stack(
        (
            latest,
            three_day_mean,
            seven_day_mean,
            np.full(latest.shape, math.sin(day_angle), dtype=np.float32),
            np.full(latest.shape, math.cos(day_angle), dtype=np.float32),
            latitude_grid,
            longitude_grid,
        ),
        axis=-1,
    )
    return features, valid_history, latest.shape


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "sample_count": int(actual.size),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def _select_xgboost(
    candidate_metrics: dict[str, float | int],
    persistence_metrics: dict[str, float | int],
) -> bool:
    return (
        candidate_metrics["RMSE"]
        <= persistence_metrics["RMSE"] * (1 - MIN_VALIDATION_RMSE_IMPROVEMENT)
        and candidate_metrics["MAE"]
        <= persistence_metrics["MAE"] * (1 + MAX_VALIDATION_MAE_DEGRADATION)
    )


def _is_contiguous_day(previous: str, current: str) -> bool:
    return date.fromisoformat(previous) + timedelta(days=1) == date.fromisoformat(current)


def _collect_split(
    variable: str,
    dates: list[str],
    first_target_index: int,
    stop_target_index: int,
    max_samples_per_day: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features_by_day: list[np.ndarray] = []
    targets_by_day: list[np.ndarray] = []
    persistence_by_day: list[np.ndarray] = []
    truth_by_day: list[np.ndarray] = []
    random = np.random.default_rng(seed)

    for target_index in range(first_target_index, stop_target_index):
        history_dates = dates[target_index - 7 : target_index]
        if len(history_dates) != 7 or any(
            not _is_contiguous_day(previous, current)
            for previous, current in zip(history_dates, history_dates[1:])
        ) or not _is_contiguous_day(history_dates[-1], dates[target_index]):
            continue

        history = np.stack(
            [
                _read_observation(variable, date.fromisoformat(day))
                for day in history_dates
            ],
            axis=0,
        )
        target = _read_observation(variable, date.fromisoformat(dates[target_index]))
        feature_grid, valid_history, _ = _feature_grid(
            history,
            date.fromisoformat(dates[target_index]),
            variable,
        )
        valid = valid_history & np.isfinite(target)
        if variable == "rainfall":
            valid &= target >= 0
        row_indices, column_indices = np.where(valid)
        if row_indices.size == 0:
            continue

        if max_samples_per_day is not None and row_indices.size > max_samples_per_day:
            selected = random.choice(
                row_indices.size,
                size=max_samples_per_day,
                replace=False,
            )
            row_indices = row_indices[selected]
            column_indices = column_indices[selected]

        features_by_day.append(feature_grid[row_indices, column_indices])
        targets_by_day.append(target[row_indices, column_indices])
        persistence_by_day.append(history[-1, row_indices, column_indices])
        truth_by_day.append(target[row_indices, column_indices])

    if not targets_by_day:
        raise ValueError(
            f"No valid samples were found for {variable} between "
            f"{dates[first_target_index]} and {dates[stop_target_index - 1]}."
        )
    return (
        np.concatenate(features_by_day),
        np.concatenate(targets_by_day),
        np.concatenate(persistence_by_day),
        np.concatenate(truth_by_day),
    )


def _predict(estimator: Any, features: np.ndarray, variable: str) -> np.ndarray:
    if estimator is None:
        predicted = features[:, 0].copy()
    else:
        predicted = estimator.predict(features)
    if variable == "rainfall":
        predicted = np.maximum(predicted, 0)
    return np.asarray(predicted, dtype=np.float32)


def train_variable(
    variable: ClimateVariable,
    *,
    max_samples_per_day: int = 1500,
    estimator_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dates = available_observation_dates(variable)
    if len(dates) < 70:
        raise ValueError(
            f"{variable} has {len(dates)} available observations; at least 70 are required "
            "for chronological train/validation/test evaluation."
        )
    if any(not _is_contiguous_day(left, right) for left, right in zip(dates, dates[1:])):
        raise ValueError(
            f"{variable} observations contain date gaps. This baseline trainer requires "
            "one continuous daily series; missing grid cells are supported, missing dates are not."
        )

    train_end = math.floor(len(dates) * 0.70)
    validation_end = math.floor(len(dates) * 0.85)
    earliest_target_index = 7
    if train_end <= earliest_target_index or validation_end <= train_end:
        raise ValueError(f"Not enough dates to split {variable} chronologically.")

    x_train, y_train, _, _ = _collect_split(
        variable,
        dates,
        earliest_target_index,
        train_end,
        max_samples_per_day,
        seed=42,
    )
    x_validation, y_validation, baseline_validation, truth_validation = _collect_split(
        variable,
        dates,
        train_end,
        validation_end,
        None,
        seed=43,
    )
    validation_estimator = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=60,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.9,
        max_bin=128,
        tree_method="hist",
        n_jobs=min(4, os.cpu_count() or 1),
        random_state=42,
        **(estimator_parameters or {}),
    )
    validation_estimator.fit(x_train, y_train)
    candidate_validation = _predict(validation_estimator, x_validation, variable)
    baseline_validation = _predict(None, baseline_validation[:, None], variable)
    candidate_validation_metrics = _metrics(truth_validation, candidate_validation)
    persistence_validation_metrics = _metrics(truth_validation, baseline_validation)
    use_xgboost = _select_xgboost(
        candidate_validation_metrics,
        persistence_validation_metrics,
    )

    final_train_end = validation_end
    x_final, y_final, _, _ = _collect_split(
        variable,
        dates,
        earliest_target_index,
        final_train_end,
        max_samples_per_day,
        seed=44,
    )

    estimator = None
    model_name = "Persistence baseline"
    parameters: dict[str, Any] = {}
    if use_xgboost:
        estimator = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=60,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.9,
            max_bin=128,
            tree_method="hist",
            n_jobs=min(4, os.cpu_count() or 1),
            random_state=42,
            **(estimator_parameters or {}),
        )
        estimator.fit(x_final, y_final)
        model_name = "XGBoost"
        parameters = estimator.get_params()

    x_test, _, persistence_test, truth_test = _collect_split(
        variable,
        dates,
        validation_end,
        len(dates),
        None,
        seed=45,
    )
    test_prediction = _predict(estimator, x_test, variable)
    persistence_test = _predict(None, persistence_test[:, None], variable)
    test_metrics = _metrics(truth_test, test_prediction)
    persistence_test_metrics = _metrics(truth_test, persistence_test)

    config = VARIABLES[variable]
    training_metadata = {
        "variable": variable,
        "model_name": model_name,
        "source": config["source"],
        "product": config["label"],
        "units": config["units"],
        "spatial_resolution_degrees": config["resolution"],
        "grid_shape": [config["rows"], config["columns"]],
        "feature_names": list(FEATURE_NAMES),
        "prediction_horizon_days": 1,
        "missing_value_handling": "Missing source cells are excluded; no filling or interpolation.",
        "training_period": {
            "start": dates[earliest_target_index],
            "end": dates[final_train_end - 1],
        },
        "chronological_splits": {
            "training": {
                "start": dates[earliest_target_index],
                "end": dates[train_end - 1],
                "samples": int(y_train.size),
            },
            "validation": {
                "start": dates[train_end],
                "end": dates[validation_end - 1],
                "samples": int(y_validation.size),
            },
            "test": {
                "start": dates[validation_end],
                "end": dates[-1],
                "samples": int(truth_test.size),
            },
        },
        "candidate_xgboost_validation_metrics": candidate_validation_metrics,
        "persistence_validation_metrics": persistence_validation_metrics,
        "selection_criterion": {
            "minimum_validation_rmse_improvement_fraction": MIN_VALIDATION_RMSE_IMPROVEMENT,
            "maximum_validation_mae_degradation_fraction": MAX_VALIDATION_MAE_DEGRADATION,
            "selected_model": model_name,
        },
        "test_metrics": test_metrics,
        "persistence_test_metrics": persistence_test_metrics,
        "xgboost_parameters": parameters if use_xgboost else validation_estimator.get_params(),
        "estimator_training_sample_count": int(y_final.size),
        "evaluation_date": datetime.now().date().isoformat(),
        "caveat": (
            "Preliminary evaluation on a single annual IMD series; these results are not "
            "multi-year or operational forecast validation."
        ),
    }

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    METRICS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIRECTORY / f"{variable}.joblib"
    metrics_path = METRICS_DIRECTORY / f"{variable}.json"
    joblib.dump({"estimator": estimator, "metadata": training_metadata}, model_path)
    metrics_path.write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")
    return training_metadata


def load_trained_model(variable: str) -> tuple[Any, dict[str, Any]]:
    if variable not in VARIABLES:
        raise ValueError(f"Unknown climate variable: {variable}")
    model_path = MODEL_DIRECTORY / f"{variable}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained {variable} forecast model is installed. "
            "Run `python train_climate_models.py` from the backend directory."
        )
    artifact = joblib.load(model_path)
    if (
        not isinstance(artifact, dict)
        or "metadata" not in artifact
        or artifact["metadata"].get("variable") != variable
    ):
        raise ValueError(f"The saved {variable} forecast artifact has an invalid format.")
    return artifact.get("estimator"), artifact["metadata"]


def model_catalog() -> dict[str, Any]:
    result: dict[str, Any] = {"models": []}
    for variable, config in VARIABLES.items():
        model_path = MODEL_DIRECTORY / f"{variable}.joblib"
        entry: dict[str, Any] = {
            "variable": variable,
            "label": config["label"],
            "units": config["units"],
            "spatial_resolution_degrees": config["resolution"],
            "available": False,
            "available_initialization_dates": [],
        }
        if model_path.is_file():
            try:
                _, metadata = load_trained_model(variable)
                observation_dates = available_observation_dates(variable)
                entry.update(
                    {
                        "available": bool(observation_dates),
                        "model_name": metadata["model_name"],
                        "training_period": metadata["training_period"],
                        "test_metrics": metadata["test_metrics"],
                        "persistence_test_metrics": metadata["persistence_test_metrics"],
                        "latest_observation_date": observation_dates[-1] if observation_dates else None,
                        "available_initialization_dates": observation_dates[6:],
                    }
                )
            except (OSError, ValueError, KeyError) as error:
                entry["error"] = str(error)
        result["models"].append(entry)
    return result


def predict_grid(variable: ClimateVariable, initialization_date: date) -> ForecastGrid:
    estimator, metadata = load_trained_model(variable)
    dates = available_observation_dates(variable)
    initialized = initialization_date.isoformat()
    try:
        index = dates.index(initialized)
    except ValueError as error:
        raise ValueError(
            f"No {variable} observation is available for {initialized}."
        ) from error
    if index < 6:
        raise ValueError("Seven consecutive observations are required before the initialization date.")
    selected_dates = dates[index - 6 : index + 1]
    if any(not _is_contiguous_day(left, right) for left, right in zip(selected_dates, selected_dates[1:])):
        raise ValueError("The seven-day observation history contains a missing date.")

    history = np.stack(
        [_read_observation(variable, date.fromisoformat(day)) for day in selected_dates]
    )
    target_date = initialization_date + timedelta(days=1)
    feature_grid, valid_history, _ = _feature_grid(history, target_date, variable)
    flat_features = feature_grid.reshape(-1, len(FEATURE_NAMES))
    valid_flat = valid_history.reshape(-1)
    flat_values = np.full(valid_flat.shape, np.nan, dtype=np.float32)
    flat_values[valid_flat] = _predict(estimator, flat_features[valid_flat], variable)
    values = flat_values.reshape(feature_grid.shape[:2])
    valid_count = int(np.isfinite(values).sum())
    config = VARIABLES[variable]
    latitudes = (
        config["latitude_origin"] + np.arange(config["rows"]) * config["resolution"]
    ).tolist()
    longitudes = (
        config["longitude_origin"] + np.arange(config["columns"]) * config["resolution"]
    ).tolist()
    forecast_date = target_date.isoformat()
    latest_observation_date = dates[-1]
    return ForecastGrid(
        variable=variable,
        initialization_date=initialized,
        forecast_date=forecast_date,
        run_type=(
            "historical_hindcast"
            if forecast_date <= latest_observation_date
            else "out_of_sample_projection"
        ),
        model_name=metadata["model_name"],
        units=config["units"],
        spatial_resolution_degrees=config["resolution"],
        latitudes=latitudes,
        longitudes=longitudes,
        values=[
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in values
        ],
        valid_cell_count=valid_count,
        missing_cell_count=int(values.size - valid_count),
        source=config["source"],
        training_period=metadata["training_period"],
        metadata=metadata,
    )


def forecast_cell(
    variable: ClimateVariable,
    initialization_date: date,
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    forecast = predict_grid(variable, initialization_date)
    if (
        latitude < forecast.latitudes[0] - forecast.spatial_resolution_degrees / 2
        or latitude > forecast.latitudes[-1] + forecast.spatial_resolution_degrees / 2
        or longitude < forecast.longitudes[0] - forecast.spatial_resolution_degrees / 2
        or longitude > forecast.longitudes[-1] + forecast.spatial_resolution_degrees / 2
    ):
        raise ValueError("The requested location is outside the native forecast grid.")
    row = int(np.abs(np.asarray(forecast.latitudes) - latitude).argmin())
    column = int(np.abs(np.asarray(forecast.longitudes) - longitude).argmin())
    value = forecast.values[row][column]
    return {
        "variable": variable,
        "model_name": forecast.model_name,
        "initialization_date": forecast.initialization_date,
        "forecast_date": forecast.forecast_date,
        "run_type": forecast.run_type,
        "latitude": forecast.latitudes[row],
        "longitude": forecast.longitudes[column],
        "value": value,
        "units": forecast.units,
        "spatial_resolution_degrees": forecast.spatial_resolution_degrees,
        "missing": value is None,
    }
