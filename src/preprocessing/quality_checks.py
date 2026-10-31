"""
Quality Checks Module
---------------------
Provides reusable data-quality checking functions for climate datasets (Rainfall, Tmax, Tmin).
Checks for missing values, NaNs, Infs, coordinate duplicates, out-of-bound lat/lon,
unphysical temperature/rainfall values, incorrect dtypes, and empty dimensions.
All operations are read-only and non-destructive.
"""

from typing import Any, Dict, Optional
import numpy as np
import xarray as xr


def check_dataset_quality(
    ds: xr.Dataset,
    var_name: Optional[str] = None,
    raw_fill_value: Optional[float] = None
) -> Dict[str, Any]:
    """
    Performs comprehensive data-quality checks on an xarray Dataset without modifying the data.

    Parameters:
        ds (xr.Dataset): Climate dataset to inspect.
        var_name (str, optional): Target data variable name (e.g. 'rain', 'tmax', 'tmin').
        raw_fill_value (float, optional): Expected raw missing fill value (e.g., -999.0 or 99.9).

    Returns:
        dict: Detailed validation results dictionary containing quality indicators and flags.
    """
    results: Dict[str, Any] = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "checks": {}
    }

    # 1. Empty Dimensions Check
    dim_sizes = dict(ds.sizes)
    empty_dims = [dim for dim, size in dim_sizes.items() if size == 0]
    results["checks"]["empty_dimensions"] = {
        "has_empty_dims": len(empty_dims) > 0,
        "empty_dims": empty_dims,
        "dimensions": dim_sizes
    }
    if empty_dims:
        results["errors"].append(f"Empty dimensions detected: {empty_dims}")
        results["is_valid"] = False

    # 2. Duplicate Coordinates Check
    duplicate_coords = {}
    for coord_name in ds.coords:
        coords_arr = ds.coords[coord_name].values
        if len(coords_arr) > len(np.unique(coords_arr)):
            duplicate_coords[coord_name] = len(coords_arr) - len(np.unique(coords_arr))

    results["checks"]["duplicate_coordinates"] = {
        "has_duplicates": len(duplicate_coords) > 0,
        "duplicate_counts": duplicate_coords
    }
    if duplicate_coords:
        results["errors"].append(f"Duplicate coordinates found: {duplicate_coords}")
        results["is_valid"] = False

    # 3. Latitude Range Validation (-90 to +90, and India domain check)
    lat_key = next((k for k in ds.coords if k.lower() in ("lat", "latitude")), None)
    if lat_key is not None:
        lats = ds.coords[lat_key].values
        invalid_lat_mask = (lats < -90.0) | (lats > 90.0)
        out_of_bounds_india = (lats < 5.0) | (lats > 40.0)

        results["checks"]["latitude"] = {
            "name": lat_key,
            "min": float(np.nanmin(lats)),
            "max": float(np.nanmax(lats)),
            "out_of_physical_range": bool(np.any(invalid_lat_mask)),
            "out_of_india_domain": bool(np.any(out_of_bounds_india))
        }
        if np.any(invalid_lat_mask):
            results["errors"].append(f"Physical latitude values out of range [-90, +90]: min={lats.min()}, max={lats.max()}")
            results["is_valid"] = False
        elif np.any(out_of_bounds_india):
            results["warnings"].append(f"Latitude extends outside standard India domain [5°N, 40°N]: [{lats.min():.2f}, {lats.max():.2f}]")

    # 4. Longitude Range Validation (-180 to +360)
    lon_key = next((k for k in ds.coords if k.lower() in ("lon", "longitude")), None)
    if lon_key is not None:
        lons = ds.coords[lon_key].values
        invalid_lon_mask = (lons < -180.0) | (lons > 360.0)
        out_of_bounds_india_lon = (lons < 60.0) | (lons > 105.0)

        results["checks"]["longitude"] = {
            "name": lon_key,
            "min": float(np.nanmin(lons)),
            "max": float(np.nanmax(lons)),
            "out_of_physical_range": bool(np.any(invalid_lon_mask)),
            "out_of_india_domain": bool(np.any(out_of_bounds_india_lon))
        }
        if np.any(invalid_lon_mask):
            results["errors"].append(f"Physical longitude values out of range [-180, +360]: min={lons.min()}, max={lons.max()}")
            results["is_valid"] = False
        elif np.any(out_of_bounds_india_lon):
            results["warnings"].append(f"Longitude extends outside standard India domain [60°E, 105°E]: [{lons.min():.2f}, {lons.max():.2f}]")

    # Determine target variable(s) to check
    target_vars = [var_name] if var_name and var_name in ds.data_vars else list(ds.data_vars.keys())

    for v_name in target_vars:
        var_array = ds[v_name]
        arr = var_array.values
        var_checks: Dict[str, Any] = {}

        # 5. Incorrect Data Type Check
        is_numeric = np.issubdtype(arr.dtype, np.number)
        var_checks["data_type"] = {
            "dtype": str(arr.dtype),
            "is_numeric": is_numeric
        }
        if not is_numeric:
            results["errors"].append(f"Variable '{v_name}' has non-numeric data type '{arr.dtype}'")
            results["is_valid"] = False

        # 6. NaN and Infinite Values Check
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())

        var_checks["nan_values"] = {
            "count": nan_count,
            "pct": round((nan_count / arr.size) * 100, 2) if arr.size > 0 else 0.0
        }
        var_checks["infinite_values"] = {
            "count": inf_count,
            "has_inf": inf_count > 0
        }
        if inf_count > 0:
            results["errors"].append(f"Variable '{v_name}' contains {inf_count} infinite values")
            results["is_valid"] = False

        # 7. Raw Fill Value Check
        if raw_fill_value is not None:
            raw_fill_count = int((arr == raw_fill_value).sum())
        else:
            raw_fill_count = int(((arr == -999.0) | (arr == 99.9)).sum())

        var_checks["missing_fill_values"] = {
            "raw_fill_value": raw_fill_value,
            "count": raw_fill_count,
            "pct": round((raw_fill_count / arr.size) * 100, 2) if arr.size > 0 else 0.0
        }

        # Mask fill values and NaNs for physical value checks
        if raw_fill_value is not None:
            mask_fill = (arr == raw_fill_value) | np.isnan(arr)
        else:
            mask_fill = (arr == -999.0) | (arr == 99.9) | (arr > 90.0) | (arr < -900.0) | np.isnan(arr)

        valid_vals = arr[~mask_fill]

        # 8. Suspicious Rainfall Values Validation
        if "rain" in v_name.lower() or "rf" in v_name.lower() or "precip" in v_name.lower():
            neg_rain_count = int((valid_vals < 0.0).sum())
            extreme_rain_count = int((valid_vals > 1000.0).sum())

            var_checks["suspicious_rainfall"] = {
                "negative_count": neg_rain_count,
                "extreme_count_gt_1000mm": extreme_rain_count,
                "min_valid": float(valid_vals.min()) if len(valid_vals) > 0 else None,
                "max_valid": float(valid_vals.max()) if len(valid_vals) > 0 else None
            }
            if neg_rain_count > 0:
                results["errors"].append(f"Rainfall variable '{v_name}' contains {neg_rain_count} negative rainfall values")
                results["is_valid"] = False
            if extreme_rain_count > 0:
                results["warnings"].append(f"Rainfall variable '{v_name}' contains {extreme_rain_count} extreme values (> 1000 mm/day)")

        # 9. Invalid Temperature Values Validation
        if "temp" in v_name.lower() or "tmax" in v_name.lower() or "tmin" in v_name.lower():
            extreme_cold = int((valid_vals < -50.0).sum())
            extreme_heat = int((valid_vals > 60.0).sum())

            var_checks["invalid_temperature"] = {
                "extreme_cold_lt_minus_50": extreme_cold,
                "extreme_heat_gt_plus_60": extreme_heat,
                "min_valid": float(valid_vals.min()) if len(valid_vals) > 0 else None,
                "max_valid": float(valid_vals.max()) if len(valid_vals) > 0 else None
            }
            if extreme_cold > 0 or extreme_heat > 0:
                results["warnings"].append(f"Temperature variable '{v_name}' contains unphysical values (< -50°C or > +60°C)")

        results["checks"][v_name] = var_checks

    # 10. Temperature Tmin vs Tmax Consistency Check
    if "tmax" in ds.data_vars and "tmin" in ds.data_vars:
        tmax = ds["tmax"].values
        tmin = ds["tmin"].values
        valid_mask = (tmax != 99.9) & (tmin != 99.9) & (~np.isnan(tmax)) & (~np.isnan(tmin))
        invalid_temp_diff = int((tmin[valid_mask] > tmax[valid_mask]).sum())

        results["checks"]["tmin_tmax_consistency"] = {
            "invalid_tmin_gt_tmax_count": invalid_temp_diff,
            "is_consistent": invalid_temp_diff == 0
        }
        if invalid_temp_diff > 0:
            results["errors"].append(f"Found {invalid_temp_diff} instances where Minimum Temperature > Maximum Temperature")
            results["is_valid"] = False

    return results


def validate_physical_bounds(
    ds: xr.Dataset,
    var_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validates that latitude, longitude, and climate variable values in the dataset
    fall within physically plausible ranges.

    Parameters:
        ds (xr.Dataset): Input xarray Dataset.
        var_name (str, optional): Target variable to check.

    Returns:
        dict: Validation results focusing on physical boundary adherence.
    """
    report = check_dataset_quality(ds, var_name=var_name)
    lat_ok = not report.get("checks", {}).get("latitude", {}).get("out_of_physical_range", False)
    lon_ok = not report.get("checks", {}).get("longitude", {}).get("out_of_physical_range", False)
    return {
        "is_valid": report["is_valid"] and lat_ok and lon_ok,
        "checks": report["checks"],
        "errors": report["errors"],
        "warnings": report["warnings"]
    }


if __name__ == "__main__":
    print("=== Quality Checks Module initialized ===")
    print("Available Quality Functions: check_dataset_quality(), validate_physical_bounds()")

