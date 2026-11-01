"""Adapters and API routes for independent IMD and NASA rainfall sources."""

from __future__ import annotations

import calendar
import os
import re
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/climate/rainfall", tags=["rainfall"])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMD_SOURCE_URL = "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html"
NASA_GIBS_CAPABILITIES_URL = (
    "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    "?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.1.1"
)
NASA_GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
NASA_DAILY_LAYER = "IMERG_Precipitation_Rate"
NASA_DAILY_CACHE_SECONDS = 1800
IMD_GRID_SHAPE = (129, 135)
IMD_LATITUDE_ORIGIN = 6.5
IMD_LONGITUDE_ORIGIN = 66.5
IMD_GRID_RESOLUTION = 0.25
IMD_MISSING_VALUE = -999.0

_year_pattern = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_nasa_cache_lock = threading.Lock()
_nasa_cache: tuple[float, dict[str, Any]] | None = None


@dataclass(frozen=True)
class RainfallGrid:
    observation_date: str
    latitudes: list[float]
    longitudes: list[float]
    values: list[list[float | None]]
    spatial_resolution_degrees: float | None
    valid_cell_count: int
    missing_cell_count: int
    data_format: str
    filename: str


def _imd_data_directory() -> Path:
    configured = os.environ.get("IMD_RAINFALL_DIR")
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT / "data" / "raw" / "rainfall"


def _matching_year_files(year: int) -> list[Path]:
    directory = _imd_data_directory()
    if not directory.is_dir():
        return []
    paths = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".grd", ".nc", ".nc4"}:
            continue
        years = {int(match) for match in _year_pattern.findall(path.name)}
        if year in years:
            paths.append(path)
    return sorted(paths, key=lambda path: (path.suffix.lower() != ".grd", str(path).lower()))


def _grid_values(values: np.ndarray, attributes: dict[str, Any]) -> tuple[np.ndarray, int, int]:
    grid = np.asarray(values, dtype=np.float64)
    missing = ~np.isfinite(grid)
    for attribute in ("_FillValue", "missing_value"):
        raw_missing = attributes.get(attribute)
        if raw_missing is not None:
            for value in np.asarray(raw_missing).reshape(-1):
                try:
                    sentinel = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(sentinel):
                    missing |= grid == sentinel
    missing |= grid == IMD_MISSING_VALUE
    valid_count = int((~missing).sum())
    missing_count = int(missing.sum())
    masked = grid.copy()
    masked[missing] = np.nan
    return masked, valid_count, missing_count


def _as_json_rows(grid: np.ndarray) -> list[list[float | None]]:
    return [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in grid
    ]


def _regular_resolution(coordinates: np.ndarray) -> float | None:
    if coordinates.size < 2:
        return None
    differences = np.diff(coordinates.astype(np.float64))
    step = float(np.median(np.abs(differences)))
    if step <= 0 or not np.allclose(np.abs(differences), step, rtol=0, atol=1e-5):
        return None
    return step


def _load_grd(path: Path, observation_date: date) -> RainfallGrid:
    days_in_year = 366 if calendar.isleap(observation_date.year) else 365
    expected_cells = days_in_year * IMD_GRID_SHAPE[0] * IMD_GRID_SHAPE[1]
    expected_bytes = expected_cells * np.dtype("<f4").itemsize
    if path.stat().st_size != expected_bytes:
        raise HTTPException(
            status_code=500,
            detail=(
                f"IMD rainfall file {path.name} has {path.stat().st_size} bytes; "
                f"expected {expected_bytes} for {observation_date.year}."
            ),
        )

    daily_grids = np.memmap(path, dtype="<f4", mode="r", shape=(days_in_year, *IMD_GRID_SHAPE))
    day_index = observation_date.timetuple().tm_yday - 1
    masked, valid_count, missing_count = _grid_values(
        daily_grids[day_index],
        {"missing_value": IMD_MISSING_VALUE},
    )
    latitudes = (
        IMD_LATITUDE_ORIGIN + np.arange(IMD_GRID_SHAPE[0]) * IMD_GRID_RESOLUTION
    ).tolist()
    longitudes = (
        IMD_LONGITUDE_ORIGIN + np.arange(IMD_GRID_SHAPE[1]) * IMD_GRID_RESOLUTION
    ).tolist()
    return RainfallGrid(
        observation_date=observation_date.isoformat(),
        latitudes=latitudes,
        longitudes=longitudes,
        values=_as_json_rows(masked),
        spatial_resolution_degrees=IMD_GRID_RESOLUTION,
        valid_cell_count=valid_count,
        missing_cell_count=missing_count,
        data_format="IMD little-endian float32 GRD",
        filename=path.name,
    )


def _coordinate_name(dataset: xr.Dataset, dimension: str, aliases: set[str]) -> str | None:
    candidates = [dimension, *dataset.coords.keys()]
    for candidate in candidates:
        if candidate in dataset.variables and candidate.lower() in aliases:
            return candidate
    for candidate in dataset.coords:
        if dimension in dataset[candidate].dims and candidate.lower() in aliases:
            return candidate
    return None


def _load_netcdf(path: Path, observation_date: date) -> RainfallGrid | None:
    try:
        with xr.open_dataset(path, decode_times=True) as dataset:
            dimensions = {dimension.lower(): dimension for dimension in dataset.dims}
            time_dimension = dimensions.get("time")
            latitude_dimension = dimensions.get("lat") or dimensions.get("latitude")
            longitude_dimension = dimensions.get("lon") or dimensions.get("longitude")
            if not time_dimension or not latitude_dimension or not longitude_dimension:
                raise ValueError("NetCDF rainfall file must have time, latitude, and longitude dimensions.")

            rainfall_variable = next(
                (
                    variable
                    for variable in dataset.data_vars.values()
                    if {time_dimension, latitude_dimension, longitude_dimension}
                    .issubset(set(variable.dims))
                ),
                None,
            )
            if rainfall_variable is None:
                raise ValueError("NetCDF file does not contain a rainfall variable on time/latitude/longitude.")

            times = np.asarray(dataset[time_dimension].values).astype("datetime64[D]")
            matches = np.flatnonzero(times == np.datetime64(observation_date.isoformat(), "D"))
            if matches.size == 0:
                return None
            time_index = int(matches[0])

            latitude_name = _coordinate_name(
                dataset, latitude_dimension, {"lat", "latitude"}
            )
            longitude_name = _coordinate_name(
                dataset, longitude_dimension, {"lon", "longitude"}
            )
            if latitude_name is None or longitude_name is None:
                raise ValueError("NetCDF file is missing latitude or longitude coordinate values.")

            latitudes_array = np.asarray(dataset[latitude_name].values, dtype=np.float64)
            longitudes_array = np.asarray(dataset[longitude_name].values, dtype=np.float64)
            values = rainfall_variable.isel({time_dimension: time_index}).transpose(
                latitude_dimension, longitude_dimension
            )
            masked, valid_count, missing_count = _grid_values(
                np.asarray(values.values),
                dict(rainfall_variable.attrs),
            )
            latitude_resolution = _regular_resolution(latitudes_array)
            longitude_resolution = _regular_resolution(longitudes_array)
            resolution = (
                latitude_resolution
                if latitude_resolution is not None and latitude_resolution == longitude_resolution
                else None
            )
            return RainfallGrid(
                observation_date=observation_date.isoformat(),
                latitudes=latitudes_array.tolist(),
                longitudes=longitudes_array.tolist(),
                values=_as_json_rows(masked),
                spatial_resolution_degrees=resolution,
                valid_cell_count=valid_count,
                missing_cell_count=missing_count,
                data_format="NetCDF",
                filename=path.name,
            )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read IMD rainfall NetCDF file {path.name}: {error}",
        ) from error


def _load_imd_grid(observation_date: date) -> RainfallGrid:
    files = _matching_year_files(observation_date.year)
    if not files:
        directory = _imd_data_directory()
        raise HTTPException(
            status_code=404,
            detail=(
                f"No IMD rainfall file is installed for {observation_date.year} in {directory}. "
                "Install the official annual GRD or NetCDF file to enable this date."
            ),
        )

    for path in files:
        if path.suffix.lower() == ".grd":
            return _load_grd(path, observation_date)
        grid = _load_netcdf(path, observation_date)
        if grid is not None:
            return grid
    raise HTTPException(
        status_code=404,
        detail=f"No observation for {observation_date.isoformat()} exists in the installed IMD rainfall files.",
    )


def _imd_available_dates() -> list[str]:
    dates: set[str] = set()
    for path in _imd_data_directory().rglob("*") if _imd_data_directory().is_dir() else ():
        if not path.is_file() or path.suffix.lower() not in {".grd", ".nc", ".nc4"}:
            continue
        years = sorted({int(match) for match in _year_pattern.findall(path.name)})
        if not years:
            continue
        for year in years:
            if path.suffix.lower() == ".grd":
                expected_size = (
                    (366 if calendar.isleap(year) else 365)
                    * IMD_GRID_SHAPE[0]
                    * IMD_GRID_SHAPE[1]
                    * np.dtype("<f4").itemsize
                )
                if path.stat().st_size != expected_size:
                    raise HTTPException(
                        status_code=500,
                        detail=f"IMD rainfall file {path.name} has an unexpected byte size.",
                    )
                dates.update(
                    (date(year, 1, 1) + timedelta(days=day - 1)).isoformat()
                    for day in range(1, (366 if calendar.isleap(year) else 365) + 1)
                )
                continue
            try:
                with xr.open_dataset(path, decode_times=True) as dataset:
                    if "time" not in dataset.coords:
                        raise ValueError("time coordinate is absent")
                    for value in np.asarray(dataset["time"].values).astype("datetime64[D]"):
                        dates.add(np.datetime_as_string(value, unit="D"))
            except Exception as error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not inspect dates in IMD NetCDF file {path.name}: {error}",
                ) from error
    return sorted(dates)


def _fetch_nasa_availability() -> dict[str, Any]:
    global _nasa_cache
    now = time.monotonic()
    with _nasa_cache_lock:
        if _nasa_cache is not None and now - _nasa_cache[0] < NASA_DAILY_CACHE_SECONDS:
            return _nasa_cache[1]

    request = urllib.request.Request(
        NASA_GIBS_CAPABILITIES_URL,
        headers={"User-Agent": "IndiaClimateTwin/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            document = ET.fromstring(response.read())
    except (urllib.error.URLError, TimeoutError, ET.ParseError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"NASA GIBS availability could not be retrieved: {error}",
        ) from error

    layer = next(
        (
            candidate
            for candidate in document.findall(".//Layer")
            if candidate.findtext("Name") == NASA_DAILY_LAYER
        ),
        None,
    )
    if layer is None:
        raise HTTPException(
            status_code=502,
            detail=f"NASA GIBS did not advertise the expected layer {NASA_DAILY_LAYER}.",
        )
    time_extent = layer.find("./Dimension[@name='time']")
    if time_extent is None or not time_extent.text:
        time_extent = layer.find("./Extent[@name='time']")
    if time_extent is None or not time_extent.text:
        raise HTTPException(
            status_code=502,
            detail="NASA GIBS did not advertise a time extent for its daily IMERG layer.",
        )

    intervals = [part.strip() for part in time_extent.text.split(",") if part.strip()]
    endpoints = [
        endpoint
        for interval in intervals
        for endpoint in interval.split("/")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", endpoint)
    ]
    if not endpoints:
        raise HTTPException(
            status_code=502,
            detail="NASA GIBS returned an unrecognized time extent for the daily IMERG layer.",
        )
    legend_resource = layer.find("./Style/LegendURL/OnlineResource")
    legend_url = (
        legend_resource.attrib.get("{http://www.w3.org/1999/xlink}href")
        if legend_resource is not None
        else None
    )

    result = {
        "source": "NASA GPM IMERG via NASA GIBS",
        "layer": NASA_DAILY_LAYER,
        "product": "Daily IMERG precipitation-rate imagery",
        "product_version": "Not identified by the GIBS layer metadata",
        "units": "mm/hr",
        "spatial_resolution": "Native grid values are not exposed by this rendered layer",
        "visualization_only": True,
        "queryable_measurements": False,
        "observation_date_start": min(endpoints),
        "observation_date_end": max(endpoints),
        "available_time_intervals": intervals,
        "imagery_service": NASA_GIBS_WMS_URL,
        "legend_url": legend_url,
        "attribution": "NASA Global Imagery Browse Services (GIBS); GPM IMERG",
    }
    with _nasa_cache_lock:
        _nasa_cache = (time.monotonic(), result)
    return result


@router.get("/sources")
def rainfall_sources() -> dict[str, Any]:
    available_dates = _imd_available_dates()
    return {
        "sources": [
            {
                "id": "imd",
                "name": "IMD Gridded Daily Rainfall",
                "product": "0.25-degree daily gridded rainfall",
                "units": "mm/day",
                "spatial_resolution": "0.25°",
                "available_dates": len(available_dates),
                "first_available_date": available_dates[0] if available_dates else None,
                "last_available_date": available_dates[-1] if available_dates else None,
                "source_url": IMD_SOURCE_URL,
                "queryable_measurements": True,
            },
            {
                "id": "nasa",
                "name": "NASA GPM IMERG via GIBS",
                "product": "Rendered precipitation imagery; date availability is provided separately",
                "source_url": NASA_GIBS_WMS_URL,
                "queryable_measurements": False,
                "earthdata_file_access": "GES DISC granule downloads require server-side Earthdata authentication.",
            },
        ]
    }


@router.get("/imd/dates")
def imd_rainfall_dates() -> dict[str, Any]:
    dates = _imd_available_dates()
    return {
        "source": "India Meteorological Department",
        "product": "0.25-degree daily gridded rainfall",
        "units": "mm/day",
        "spatial_resolution_degrees": IMD_GRID_RESOLUTION,
        "available_dates": dates,
        "source_url": IMD_SOURCE_URL,
    }


@router.get("/imd")
def imd_rainfall_grid(observation_date: date = Query(alias="date")) -> dict[str, Any]:
    grid = _load_imd_grid(observation_date)
    return {
        "source": "India Meteorological Department (IMD)",
        "product": "Daily gridded rainfall",
        "product_version": "Official annual archive file",
        "observation_date": grid.observation_date,
        "units": "mm/day",
        "spatial_resolution_degrees": grid.spatial_resolution_degrees,
        "grid_shape": [len(grid.latitudes), len(grid.longitudes)],
        "latitudes": grid.latitudes,
        "longitudes": grid.longitudes,
        "values": grid.values,
        "missing_value_raw": IMD_MISSING_VALUE if grid.data_format.endswith("GRD") else None,
        "valid_cell_count": grid.valid_cell_count,
        "missing_cell_count": grid.missing_cell_count,
        "data_format": grid.data_format,
        "filename": grid.filename,
        "source_url": IMD_SOURCE_URL,
    }


@router.get("/imd/cell")
def imd_rainfall_cell(
    observation_date: date = Query(alias="date"),
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=180),
) -> dict[str, Any]:
    grid = _load_imd_grid(observation_date)
    latitudes = np.asarray(grid.latitudes)
    longitudes = np.asarray(grid.longitudes)
    if (
        latitude < min(grid.latitudes) - 0.125
        or latitude > max(grid.latitudes) + 0.125
        or longitude < min(grid.longitudes) - 0.125
        or longitude > max(grid.longitudes) + 0.125
    ):
        raise HTTPException(
            status_code=422,
            detail="The requested location is outside the installed IMD rainfall grid.",
        )
    row = int(np.abs(latitudes - latitude).argmin())
    column = int(np.abs(longitudes - longitude).argmin())
    value = grid.values[row][column]
    return {
        "source": "India Meteorological Department (IMD)",
        "product": "Daily gridded rainfall",
        "observation_date": grid.observation_date,
        "latitude": grid.latitudes[row],
        "longitude": grid.longitudes[column],
        "rainfall": value,
        "units": "mm/day",
        "spatial_resolution_degrees": grid.spatial_resolution_degrees,
        "missing": value is None,
        "data_format": grid.data_format,
        "filename": grid.filename,
        "source_url": IMD_SOURCE_URL,
    }


@router.get("/nasa/dates")
def nasa_rainfall_dates() -> dict[str, Any]:
    return _fetch_nasa_availability()
