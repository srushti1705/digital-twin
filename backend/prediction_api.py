"""API routes for evaluated one-day-ahead IMD grid forecasts."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

try:
    from .climate_forecast import (
        VARIABLES,
        available_observation_dates,
        load_trained_model,
        model_catalog,
        predict_grid,
    )
except ImportError:
    from climate_forecast import (
        VARIABLES,
        available_observation_dates,
        load_trained_model,
        model_catalog,
        predict_grid,
    )

router = APIRouter(prefix="/api/predictions", tags=["predictions"])
ClimateVariable = Literal["rainfall", "tmax", "tmin"]


@lru_cache(maxsize=6)
def _cached_forecast(variable: ClimateVariable, initialization_date: date):
    return predict_grid(variable, initialization_date)


@router.get("")
@router.get("/catalog")
def prediction_catalog():
    return model_catalog()


@router.get("/{variable}/dates")
def prediction_dates(variable: ClimateVariable):
    try:
        _, metadata = load_trained_model(variable)
        observation_dates = available_observation_dates(variable)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {
        "variable": variable,
        "model_name": metadata["model_name"],
        "available_initialization_dates": observation_dates[6:],
        "latest_observation_date": observation_dates[-1] if observation_dates else None,
        "prediction_horizon_days": 1,
        "units": VARIABLES[variable]["units"],
        "source": VARIABLES[variable]["source"],
    }


@router.get("/{variable}/cell")
def prediction_cell(
    variable: ClimateVariable,
    initialization_date: date = Query(),
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=180),
):
    try:
        grid = _cached_forecast(variable, initialization_date)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if (
        latitude < grid.latitudes[0] - grid.spatial_resolution_degrees / 2
        or latitude > grid.latitudes[-1] + grid.spatial_resolution_degrees / 2
        or longitude < grid.longitudes[0] - grid.spatial_resolution_degrees / 2
        or longitude > grid.longitudes[-1] + grid.spatial_resolution_degrees / 2
    ):
        raise HTTPException(
            status_code=422,
            detail="The requested location is outside the native forecast grid.",
        )
    row = min(
        range(len(grid.latitudes)),
        key=lambda index: abs(grid.latitudes[index] - latitude),
    )
    column = min(
        range(len(grid.longitudes)),
        key=lambda index: abs(grid.longitudes[index] - longitude),
    )
    value = grid.values[row][column]
    return {
        "variable": variable,
        "model_name": grid.model_name,
        "initialization_date": grid.initialization_date,
        "forecast_date": grid.forecast_date,
        "run_type": grid.run_type,
        "latitude": grid.latitudes[row],
        "longitude": grid.longitudes[column],
        "value": value,
        "units": grid.units,
        "spatial_resolution_degrees": grid.spatial_resolution_degrees,
        "missing": value is None,
    }


@router.get("/{variable}")
def prediction_grid(
    variable: ClimateVariable,
    initialization_date: date = Query(),
):
    try:
        grid = _cached_forecast(variable, initialization_date)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "source": grid.source,
        "variable": grid.variable,
        "model_name": grid.model_name,
        "initialization_date": grid.initialization_date,
        "forecast_date": grid.forecast_date,
        "run_type": grid.run_type,
        "units": grid.units,
        "spatial_resolution_degrees": grid.spatial_resolution_degrees,
        "grid_shape": [len(grid.latitudes), len(grid.longitudes)],
        "latitudes": grid.latitudes,
        "longitudes": grid.longitudes,
        "values": grid.values,
        "valid_cell_count": grid.valid_cell_count,
        "missing_cell_count": grid.missing_cell_count,
        "training_period": grid.training_period,
        "evaluation": {
            "test_metrics": grid.metadata["test_metrics"],
            "persistence_test_metrics": grid.metadata["persistence_test_metrics"],
            "selection_criterion": grid.metadata["selection_criterion"],
            "caveat": grid.metadata["caveat"],
        },
    }
