"""Adapters and endpoints for official IMD daily temperature grids."""

from __future__ import annotations

import calendar
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/climate/temperature", tags=["temperature"])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMD_MAX_TEMP_SOURCE = "https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html"
IMD_MIN_TEMP_SOURCE = "https://imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html"
GRID_SHAPE = (31, 31)
LATITUDE_ORIGIN = 7.5
LONGITUDE_ORIGIN = 67.5
GRID_RESOLUTION = 1.0
MISSING_VALUE = 99.9
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
TemperatureName = Literal["tmax", "tmin"]

PRODUCTS: dict[str, dict[str, str]] = {
    "tmax": {
        "directory": "max_temp",
        "filename": "maxtemp",
        "name": "Daily maximum temperature",
        "variable": "Tmax",
        "source_url": IMD_MAX_TEMP_SOURCE,
    },
    "tmin": {
        "directory": "min_temp",
        "filename": "mintemp",
        "name": "Daily minimum temperature",
        "variable": "Tmin",
        "source_url": IMD_MIN_TEMP_SOURCE,
    },
}


def _data_directory(product: TemperatureName) -> Path:
    configured = os.environ.get("IMD_TEMPERATURE_DIR")
    if configured:
        return Path(configured).expanduser().resolve() / PRODUCTS[product]["directory"]
    return PROJECT_ROOT / "data" / "raw" / PRODUCTS[product]["directory"]


def _year_files(product: TemperatureName, year: int) -> list[Path]:
    directory = _data_directory(product)
    if not directory.is_dir():
        return []
    result = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".grd":
            continue
        if year in {int(value) for value in YEAR_PATTERN.findall(path.name)}:
            result.append(path)
    return sorted(result, key=lambda path: (path.suffix.lower() != ".grd", str(path).lower()))


def _days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def _read_binary_day(
    path: Path,
    observation_date: date,
) -> tuple[np.ndarray, int, int]:
    expected_cells = _days_in_year(observation_date.year) * GRID_SHAPE[0] * GRID_SHAPE[1]
    expected_bytes = expected_cells * np.dtype("<f4").itemsize
    if path.stat().st_size != expected_bytes:
        raise HTTPException(
            status_code=500,
            detail=(
                f"IMD temperature file {path.name} has {path.stat().st_size} bytes; "
                f"expected {expected_bytes} for {observation_date.year}."
            ),
        )
    annual_data = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        shape=(_days_in_year(observation_date.year), *GRID_SHAPE),
    )
    grid = np.asarray(annual_data[observation_date.timetuple().tm_yday - 1], dtype=np.float64)
    missing = ~np.isfinite(grid) | np.isclose(
        grid, MISSING_VALUE, rtol=0, atol=1e-4
    )
    valid_count = int((~missing).sum())
    missing_count = int(missing.sum())
    grid[missing] = np.nan
    return grid, valid_count, missing_count


def _dates_for_product(product: TemperatureName) -> list[str]:
    directory = _data_directory(product)
    if not directory.is_dir():
        return []
    dates: set[str] = set()
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".grd":
            continue
        years = sorted({int(value) for value in YEAR_PATTERN.findall(path.name)})
        if not years:
            continue
        for year in years:
            days = _days_in_year(year)
            expected_bytes = days * GRID_SHAPE[0] * GRID_SHAPE[1] * np.dtype("<f4").itemsize
            if path.stat().st_size != expected_bytes:
                raise HTTPException(
                    status_code=500,
                    detail=f"IMD temperature file {path.name} has an unexpected byte size.",
                )
            annual_data = np.memmap(
                path,
                dtype="<f4",
                mode="r",
                shape=(days, *GRID_SHAPE),
            )
            valid_cells = np.isfinite(annual_data) & ~np.isclose(
                annual_data, MISSING_VALUE, rtol=0, atol=1e-4
            )
            days_with_observations = valid_cells.any(axis=(1, 2))
            dates.update(
                (date(year, 1, 1) + timedelta(days=index)).isoformat()
                for index, available in enumerate(days_with_observations)
                if available
            )
    return sorted(dates)


def _load_temperature_grid(
    product: TemperatureName,
    observation_date: date,
) -> tuple[dict[str, Any], np.ndarray]:
    for path in _year_files(product, observation_date.year):
        if path.suffix.lower() != ".grd":
            continue
        grid, valid_count, missing_count = _read_binary_day(path, observation_date)
        if valid_count == 0:
            continue
        metadata = {
            "source": "India Meteorological Department (IMD)",
            "product": PRODUCTS[product]["name"],
            "variable": PRODUCTS[product]["variable"],
            "observation_date": observation_date.isoformat(),
            "units": "°C",
            "spatial_resolution_degrees": GRID_RESOLUTION,
            "grid_shape": list(GRID_SHAPE),
            "latitudes": (LATITUDE_ORIGIN + np.arange(GRID_SHAPE[0]) * GRID_RESOLUTION).tolist(),
            "longitudes": (
                LONGITUDE_ORIGIN + np.arange(GRID_SHAPE[1]) * GRID_RESOLUTION
            ).tolist(),
            "valid_cell_count": valid_count,
            "missing_cell_count": missing_count,
            "missing_value_raw": MISSING_VALUE,
            "data_format": "IMD little-endian float32 GRD",
            "filename": path.name,
            "source_url": PRODUCTS[product]["source_url"],
        }
        return metadata, grid
    if not _year_files(product, observation_date.year):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No IMD {product} file is installed for {observation_date.year} in "
                f"{_data_directory(product)}."
            ),
        )
    raise HTTPException(
        status_code=404,
        detail=f"No valid IMD {product} observation exists for {observation_date.isoformat()}.",
    )


def _get_product(raw_product: str) -> TemperatureName:
    if raw_product == "tmax":
        return "tmax"
    if raw_product == "tmin":
        return "tmin"
    raise HTTPException(status_code=404, detail="Temperature variable must be 'tmax' or 'tmin'.")


def _json_grid(grid: np.ndarray) -> list[list[float | None]]:
    return [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in grid
    ]


@router.get("/{product}/dates")
def temperature_dates(product: str) -> dict[str, Any]:
    selected_product = _get_product(product)
    dates = _dates_for_product(selected_product)
    return {
        "source": "India Meteorological Department (IMD)",
        "product": PRODUCTS[selected_product]["name"],
        "variable": PRODUCTS[selected_product]["variable"],
        "units": "°C",
        "spatial_resolution_degrees": GRID_RESOLUTION,
        "available_dates": dates,
        "source_url": PRODUCTS[selected_product]["source_url"],
    }


@router.get("/{product}")
def temperature_grid(
    product: str,
    observation_date: date = Query(alias="date"),
) -> dict[str, Any]:
    selected_product = _get_product(product)
    metadata, grid = _load_temperature_grid(selected_product, observation_date)
    return {**metadata, "values": _json_grid(grid)}


@router.get("/{product}/cell")
def temperature_cell(
    product: str,
    observation_date: date = Query(alias="date"),
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=180),
) -> dict[str, Any]:
    selected_product = _get_product(product)
    metadata, grid = _load_temperature_grid(selected_product, observation_date)
    latitudes = np.asarray(metadata["latitudes"])
    longitudes = np.asarray(metadata["longitudes"])
    if (
        latitude < LATITUDE_ORIGIN - GRID_RESOLUTION / 2
        or latitude > latitudes[-1] + GRID_RESOLUTION / 2
        or longitude < LONGITUDE_ORIGIN - GRID_RESOLUTION / 2
        or longitude > longitudes[-1] + GRID_RESOLUTION / 2
    ):
        raise HTTPException(
            status_code=422,
            detail="The requested location is outside the installed IMD temperature grid.",
        )
    row = int(np.abs(latitudes - latitude).argmin())
    column = int(np.abs(longitudes - longitude).argmin())
    value = grid[row, column]
    return {
        "source": metadata["source"],
        "product": metadata["product"],
        "variable": metadata["variable"],
        "observation_date": metadata["observation_date"],
        "latitude": float(latitudes[row]),
        "longitude": float(longitudes[column]),
        "temperature": None if not np.isfinite(value) else float(value),
        "units": metadata["units"],
        "spatial_resolution_degrees": metadata["spatial_resolution_degrees"],
        "missing": not np.isfinite(value),
        "filename": metadata["filename"],
        "source_url": metadata["source_url"],
    }
