"""Spatial masking and geometry operations for What-If scenario bounds."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.path as mpath
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEOJSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "geospatial"
    / "india_states"
    / "india_states_processed.geojson"
)

# Rainfall coordinates
RAINFALL_LATS = 6.5 + np.arange(129) * 0.25
RAINFALL_LONS = 66.5 + np.arange(135) * 0.25

# Temperature coordinates
TEMP_LATS = 7.5 + np.arange(31) * 1.0
TEMP_LONS = 67.5 + np.arange(31) * 1.0


@lru_cache(maxsize=1)
def _load_features() -> dict[str, Any]:
    """Load and index state boundary features by state code."""
    path = GEOJSON_PATH
    if not path.is_file():
        # Fallback to frontend public copy
        alt = PROJECT_ROOT / "frontend" / "public" / "india-states.geojson"
        if alt.is_file():
            path = alt
        else:
            return {}

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    indexed: dict[str, Any] = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        code = props.get("iso_code") or props.get("STATE_CODE") or feature.get("id")
        if code:
            indexed[str(code).upper()] = feature
    return indexed


def get_supported_states() -> list[dict[str, str]]:
    """Return available state options for geographic selection."""
    features = _load_features()
    result = [{"code": "all_india", "name": "All India (Nationwide Grid)"}]
    for code, feature in sorted(features.items()):
        name = (
            feature.get("properties", {}).get("state_name")
            or feature.get("properties", {}).get("ST_NM")
            or code
        )
        result.append({"code": code, "name": name})
    return result


@lru_cache(maxsize=128)
def get_spatial_mask(scope_code: str, grid_type: str) -> np.ndarray:
    """Return a boolean 2D mask matching the grid shape.

    True where cells are inside the selected region; False otherwise.
    """
    if grid_type == "rainfall":
        lats, lons = RAINFALL_LATS, RAINFALL_LONS
        shape = (129, 135)
    elif grid_type == "temperature":
        lats, lons = TEMP_LATS, TEMP_LONS
        shape = (31, 31)
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")

    code_clean = scope_code.strip().upper()
    if code_clean in {"ALL", "ALL_INDIA", "INDIA"}:
        return np.ones(shape, dtype=bool)

    features = _load_features()
    feature = features.get(code_clean)
    if feature is None:
        raise ValueError(f"State code '{scope_code}' not recognized in geospatial dataset.")

    # Meshgrid coordinates for points: [lon, lat]
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    geometry = feature.get("geometry", {})
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])

    mask_flat = np.zeros(points.shape[0], dtype=bool)

    if geom_type == "Polygon":
        for ring in coords:
            path = mpath.Path(ring)
            mask_flat |= path.contains_points(points)
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                path = mpath.Path(ring)
                mask_flat |= path.contains_points(points)

    return mask_flat.reshape(shape)

