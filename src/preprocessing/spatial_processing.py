"""
Spatial Processing Module
--------------------------
Handles spatial coordinate identification, range validation, consistent coordinate sorting,
spatial resolution inspection, duplicate spatial coordinate detection, spatial reference preservation,
and resolution mismatch documentation.

Preserves original spatial references and resolutions by default without premature resampling.
Provides explicit re-gridding utilities (regrid_dataset, regrid_temperature_to_rainfall)
for downstream pipeline invocation.
"""

from typing import Dict, Tuple, Optional, Any, List, Union
import logging
import numpy as np
import xarray as xr

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def identify_spatial_coordinates(ds: Union[xr.Dataset, xr.DataArray]) -> Tuple[str, str]:
    """
    Identifies latitude and longitude coordinate names in an xarray Dataset or DataArray.

    Parameters:
        ds (xr.Dataset | xr.DataArray): Input xarray object.

    Returns:
        tuple: (lat_key, lon_key) coordinate names.

    Raises:
        ValueError: If lat or lon coordinates cannot be identified.
    """
    coords = list(ds.coords.keys()) if hasattr(ds, "coords") else list(ds.dims)

    lat_candidates = ("lat", "latitude", "y", "lats")
    lon_candidates = ("lon", "longitude", "x", "lons")

    lat_key = next((c for c in coords if str(c).lower() in lat_candidates), None)
    lon_key = next((c for c in coords if str(c).lower() in lon_candidates), None)

    if lat_key is None or lon_key is None:
        raise ValueError(f"❌ Could not identify spatial lat/lon coordinates in dataset. Found coords: {coords}")

    return str(lat_key), str(lon_key)


def validate_coordinate_ranges(
    ds: Union[xr.Dataset, xr.DataArray],
    physical_lat_bounds: Tuple[float, float] = (-90.0, 90.0),
    physical_lon_bounds: Tuple[float, float] = (-180.0, 360.0),
    india_lat_bounds: Tuple[float, float] = (5.0, 40.0),
    india_lon_bounds: Tuple[float, float] = (60.0, 105.0)
) -> Dict[str, Any]:
    """
    Validates latitude and longitude coordinate ranges against physical world bounds and India domain bounds.

    Parameters:
        ds (xr.Dataset | xr.DataArray): Input dataset.
        physical_lat_bounds (tuple): Valid physical latitude bounds [-90, +90].
        physical_lon_bounds (tuple): Valid physical longitude bounds [-180, +360].
        india_lat_bounds (tuple): Expected India domain latitude bounds [5°N, 40°N].
        india_lon_bounds (tuple): Expected India domain longitude bounds [60°E, 105°E].

    Returns:
        dict: Coordinate range validation report.
    """
    lat_key, lon_key = identify_spatial_coordinates(ds)

    lats = ds.coords[lat_key].values
    lons = ds.coords[lon_key].values

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    out_of_physical_lat = bool(lat_min < physical_lat_bounds[0] or lat_max > physical_lat_bounds[1])
    out_of_physical_lon = bool(lon_min < physical_lon_bounds[0] or lon_max > physical_lon_bounds[1])

    out_of_india_lat = bool(lat_min < india_lat_bounds[0] or lat_max > india_lat_bounds[1])
    out_of_india_lon = bool(lon_min < india_lon_bounds[0] or lon_max > india_lon_bounds[1])

    is_valid = not (out_of_physical_lat or out_of_physical_lon)

    warnings = []
    if out_of_india_lat:
        warnings.append(f"Latitude range [{lat_min:.2f}, {lat_max:.2f}] extends outside standard India domain [5°N, 40°N].")
    if out_of_india_lon:
        warnings.append(f"Longitude range [{lon_min:.2f}, {lon_max:.2f}] extends outside standard India domain [60°E, 105°E].")

    return {
        "is_valid": is_valid,
        "lat_key": lat_key,
        "lon_key": lon_key,
        "lat_range": (lat_min, lat_max),
        "lon_range": (lon_min, lon_max),
        "out_of_physical_bounds": out_of_physical_lat or out_of_physical_lon,
        "out_of_india_bounds": out_of_india_lat or out_of_india_lon,
        "warnings": warnings
    }


def sort_spatial_coordinates(
    ds: xr.Dataset,
    lat_ascending: bool = True,
    lon_ascending: bool = True
) -> xr.Dataset:
    """
    Sorts dataset observations consistently along spatial coordinates (lat, lon).

    Parameters:
        ds (xr.Dataset): Input dataset.
        lat_ascending (bool): Whether to sort latitude in ascending order (South-to-North). Default: True.
        lon_ascending (bool): Whether to sort longitude in ascending order (West-to-East). Default: True.

    Returns:
        xr.Dataset: Consistently sorted Dataset preserving metadata attributes.
    """
    ds_sorted = ds.copy(deep=True)
    lat_key, lon_key = identify_spatial_coordinates(ds_sorted)

    ds_sorted = ds_sorted.sortby(lat_key, ascending=lat_ascending)
    ds_sorted = ds_sorted.sortby(lon_key, ascending=lon_ascending)

    return ds_sorted


def detect_duplicate_spatial_coordinates(ds: Union[xr.Dataset, xr.DataArray]) -> Dict[str, Any]:
    """
    Detects duplicate values in latitude and longitude coordinate arrays.

    Parameters:
        ds (xr.Dataset | xr.DataArray): Input dataset.

    Returns:
        dict: Summary of duplicate spatial coordinate occurrences.
    """
    lat_key, lon_key = identify_spatial_coordinates(ds)

    lats = ds.coords[lat_key].values
    lons = ds.coords[lon_key].values

    lat_dup_count = len(lats) - len(np.unique(lats))
    lon_dup_count = len(lons) - len(np.unique(lons))

    has_duplicates = (lat_dup_count > 0) or (lon_dup_count > 0)

    return {
        "lat_key": lat_key,
        "lon_key": lon_key,
        "has_duplicates": has_duplicates,
        "lat_duplicate_count": lat_dup_count,
        "lon_duplicate_count": lon_dup_count,
        "lat_grid_points": len(lats),
        "lon_grid_points": len(lons)
    }


def get_spatial_resolution(ds: Union[xr.Dataset, xr.DataArray]) -> Dict[str, Any]:
    """
    Computes spatial resolution (lat/lon grid spacing, bounds, grid regularity, total cells).

    Parameters:
        ds (xr.Dataset | xr.DataArray): Input dataset.

    Returns:
        dict: Detailed spatial resolution properties.
    """
    lat_key, lon_key = identify_spatial_coordinates(ds)

    lats = ds.coords[lat_key].values
    lons = ds.coords[lon_key].values

    lat_diffs = np.diff(lats) if len(lats) > 1 else np.array([0.0])
    lon_diffs = np.diff(lons) if len(lons) > 1 else np.array([0.0])

    dlat = float(abs(lat_diffs.mean()))
    dlon = float(abs(lon_diffs.mean()))

    # Check regularity of grid spacing
    lat_regular = bool(np.allclose(np.abs(lat_diffs), dlat, atol=1e-4)) if len(lats) > 1 else True
    lon_regular = bool(np.allclose(np.abs(lon_diffs), dlon, atol=1e-4)) if len(lons) > 1 else True

    n_lat, n_lon = len(lats), len(lons)

    return {
        "lat_key": lat_key,
        "lon_key": lon_key,
        "lat_step": round(dlat, 4),
        "lon_step": round(dlon, 4),
        "lat_grid_points": n_lat,
        "lon_grid_points": n_lon,
        "total_spatial_cells": n_lat * n_lon,
        "lat_range": (float(lats.min()), float(lats.max())),
        "lon_range": (float(lons.min()), float(lons.max())),
        "is_regular_grid": lat_regular and lon_regular
    }


def document_resolution_mismatch(dataset_map: Dict[str, xr.Dataset]) -> Dict[str, Any]:
    """
    Compares spatial resolutions and grid structures across multiple datasets
    (e.g., Rainfall vs Temperature) and generates a diagnostic mismatch report.

    Does NOT resample datasets. Documents differences for future pipeline execution.

    Parameters:
        dataset_map (dict): Mapping of dataset names (e.g. 'rainfall', 'tmax', 'tmin') to xarray Datasets.

    Returns:
        dict: Comprehensive spatial resolution mismatch diagnostic report.
    """
    mismatch_report: Dict[str, Any] = {
        "has_mismatch": False,
        "datasets": {},
        "mismatch_summary": []
    }

    resolutions = {}

    for ds_name, ds in dataset_map.items():
        res_info = get_spatial_resolution(ds)
        resolutions[ds_name] = res_info
        mismatch_report["datasets"][ds_name] = res_info

    # Compare resolutions across datasets
    keys = list(resolutions.keys())
    if len(keys) > 1:
        base_name = keys[0]
        base_res = resolutions[base_name]

        for other_name in keys[1:]:
            other_res = resolutions[other_name]

            lat_diff = abs(base_res["lat_step"] - other_res["lat_step"]) > 1e-4
            lon_diff = abs(base_res["lon_step"] - other_res["lon_step"]) > 1e-4
            bounds_diff = (base_res["lat_range"] != other_res["lat_range"]) or (base_res["lon_range"] != other_res["lon_range"])

            if lat_diff or lon_diff or bounds_diff:
                mismatch_report["has_mismatch"] = True
                detail = (
                    f"Mismatch between '{base_name}' ({base_res['lat_step']}° x {base_res['lon_step']}°, {base_res['lat_grid_points']}x{base_res['lon_grid_points']}) "
                    f"and '{other_name}' ({other_res['lat_step']}° x {other_res['lon_step']}°, {other_res['lat_grid_points']}x{other_res['lon_grid_points']}). "
                    f"Base domain: Lat {base_res['lat_range']}, Lon {base_res['lon_range']} vs Other domain: Lat {other_res['lat_range']}, Lon {other_res['lon_range']}."
                )
                mismatch_report["mismatch_summary"].append(detail)
                logger.warning(f"Spatial Resolution Mismatch Detected: {detail}")

    return mismatch_report


def generate_spatial_report(dataset_map: Dict[str, xr.Dataset]) -> str:
    """
    Generates a human-readable markdown report documenting spatial coordinate properties and resolution mismatches.

    Parameters:
        dataset_map (dict): Mapping of dataset names to xarray Datasets.

    Returns:
        str: Markdown diagnostic spatial report.
    """
    mismatch_info = document_resolution_mismatch(dataset_map)

    lines = []
    lines.append("# Spatial Preprocessing & Resolution Mismatch Report")
    lines.append("")
    lines.append("## Dataset Spatial Properties")
    lines.append("")

    for ds_name, res in mismatch_info["datasets"].items():
        lines.append(f"### Dataset: `{ds_name}`")
        lines.append(f"- **Latitude Key**: `{res['lat_key']}`")
        lines.append(f"- **Longitude Key**: `{res['lon_key']}`")
        lines.append(f"- **Spatial Resolution**: `{res['lat_step']}° x {res['lon_step']}°`")
        lines.append(f"- **Grid Dimensions**: `{res['lat_grid_points']} lat` x `{res['lon_grid_points']} lon` ({res['total_spatial_cells']:,} cells)")
        lines.append(f"- **Latitude Range**: `[{res['lat_range'][0]:.2f}°N, {res['lat_range'][1]:.2f}°N]`")
        lines.append(f"- **Longitude Range**: `[{res['lon_range'][0]:.2f}°E, {res['lon_range'][1]:.2f}°E]`")
        lines.append(f"- **Regular Grid Spacing**: `{res['is_regular_grid']}`")
        lines.append("")

    lines.append("## Resolution Mismatch Status")
    lines.append("")
    if mismatch_info["has_mismatch"]:
        lines.append("> [!WARNING]")
        lines.append("> **Resolution Mismatch Detected**: The datasets operate at differing spatial resolutions and coordinate bounding boxes.")
        lines.append("")
        for summary in mismatch_info["mismatch_summary"]:
            lines.append(f"- {summary}")
        lines.append("")
        lines.append("### Recommended Action for Subsequent Pipeline Step:")
        lines.append("- Bilinearly re-grid 1.0° Temperature datasets (`tmax`, `tmin`) onto the high-resolution 0.25° Rainfall grid.")
        lines.append("- Crop unified domain to standard India bounding box (`6.5°N - 38.5°N`, `66.5°E - 100.0°E`).")
    else:
        lines.append("All datasets have identical spatial resolutions and grid bounds. No re-gridding required.")

    return "\n".join(lines)


# ==============================================================================
# DOWNSTREAM RE-GRIDDING & CROPPING UTILITIES (EXPLICIT PIPELINE CALLS)
# ==============================================================================

def crop_spatial_domain(
    ds: xr.Dataset,
    lat_bounds: Tuple[float, float] = (6.5, 38.5),
    lon_bounds: Tuple[float, float] = (66.5, 100.0)
) -> xr.Dataset:
    """Crops dataset to a specified latitude and longitude bounding box."""
    res = get_spatial_resolution(ds)
    lat_key, lon_key = res["lat_key"], res["lon_key"]

    lats = ds.coords[lat_key].values
    lons = ds.coords[lon_key].values

    lat_min, lat_max = sorted(lat_bounds)
    lon_min, lon_max = sorted(lon_bounds)

    if lats[0] <= lats[-1]:
        lat_slice = slice(lat_min, lat_max)
    else:
        lat_slice = slice(lat_max, lat_min)

    if lons[0] <= lons[-1]:
        lon_slice = slice(lon_min, lon_max)
    else:
        lon_slice = slice(lon_max, lon_min)

    kwargs = {lat_key: lat_slice, lon_key: lon_slice}
    return ds.sel(**kwargs)


def regrid_dataset(
    source_ds: xr.Dataset,
    target_grid: xr.Dataset,
    method: str = "linear"
) -> xr.Dataset:
    """Re-grids source_ds onto spatial coordinates (lat, lon) of target_grid."""
    src_res = get_spatial_resolution(source_ds)
    tgt_res = get_spatial_resolution(target_grid)

    src_lat_key, src_lon_key = src_res["lat_key"], src_res["lon_key"]
    tgt_lat_key, tgt_lon_key = tgt_res["lat_key"], tgt_res["lon_key"]

    target_lats = target_grid.coords[tgt_lat_key].values
    target_lons = target_grid.coords[tgt_lon_key].values

    ds_renamed = source_ds.copy(deep=True)
    if src_lat_key != tgt_lat_key:
        ds_renamed = ds_renamed.rename({src_lat_key: tgt_lat_key})
    if src_lon_key != tgt_lon_key:
        ds_renamed = ds_renamed.rename({src_lon_key: tgt_lon_key})

    interp_coords = {
        tgt_lat_key: target_lats,
        tgt_lon_key: target_lons
    }

    regridded_ds = ds_renamed.interp(
        coords=interp_coords,
        method=method,
        kwargs={"fill_value": "extrapolate"} if method == "nearest" else {}
    )

    return regridded_ds


def regrid_temperature_to_rainfall(
    temp_ds: xr.Dataset,
    rain_ds: xr.Dataset,
    method: str = "linear"
) -> xr.Dataset:
    """Convenience function to re-grid 1.0-degree temperature onto 0.25-degree rainfall grid."""
    return regrid_dataset(source_ds=temp_ds, target_grid=rain_ds, method=method)


if __name__ == "__main__":
    print("=== Spatial Processing Module Initialized ===")
    print("Available Functions:")
    print("  - identify_spatial_coordinates()")
    print("  - validate_coordinate_ranges()")
    print("  - sort_spatial_coordinates()")
    print("  - detect_duplicate_spatial_coordinates()")
    print("  - get_spatial_resolution()")
    print("  - document_resolution_mismatch()")
    print("  - generate_spatial_report()")
