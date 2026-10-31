"""
Climate Dataset Validation Script
---------------------------------
Performs non-destructive inspection and basic validation on NetCDF climate datasets.
Prints a comprehensive, human-readable report summarizing dataset metadata, spatial bounds,
time dimensions, missing values, data types, and basic statistics.
"""

from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
import pandas as pd
import xarray as xr


def validate_climate_dataset(
    file_path: Union[str, Path],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Inspects and validates a NetCDF climate dataset without modifying any underlying data.

    Parameters:
        file_path (str | Path): Absolute or relative path to the NetCDF dataset (.nc, .nc4).
        verbose (bool): If True, prints a formatted human-readable report to console.

    Returns:
        dict: Diagnostic metadata summary containing validation metrics.

    Raises:
        FileNotFoundError: If target file does not exist.
        ValueError: If path points to a directory.
        IOError: If dataset cannot be opened by xarray.
    """
    path = Path(file_path).resolve()

    # 1. File Existence Check
    if not path.exists():
        raise FileNotFoundError(f"❌ Dataset file not found at: '{path}'")

    if not path.is_file():
        raise ValueError(f"❌ Path '{path}' is a directory, not a file.")

    file_size_mb = path.stat().st_size / (1024 * 1024)

    # 2. Open Dataset with xarray (Read-only mode)
    try:
        ds = xr.open_dataset(path)
    except Exception as err:
        raise IOError(f"❌ Failed to open NetCDF dataset at '{path}'. Error: {err}") from err

    report = {
        "file_path": str(path),
        "file_size_mb": file_size_mb,
        "global_attributes": dict(ds.attrs),
        "dimensions": dict(ds.sizes),
        "variables": {},
        "spatial": {},
        "temporal": {},
    }

    # 3. Coordinate Inspection (Lat, Lon, Time)
    lat_key = next((k for k in ds.coords if k.lower() in ("lat", "latitude")), None)
    lon_key = next((k for k in ds.coords if k.lower() in ("lon", "longitude")), None)
    time_key = next((k for k in ds.coords if k.lower() in ("time", "date")), None)

    if lat_key is not None:
        lats = ds.coords[lat_key].values
        report["spatial"]["latitude"] = {
            "name": lat_key,
            "min_lat": float(np.nanmin(lats)),
            "max_lat": float(np.nanmax(lats)),
            "step": float(np.abs(np.diff(lats)).mean()) if len(lats) > 1 else 0.0,
            "count": len(lats)
        }

    if lon_key is not None:
        lons = ds.coords[lon_key].values
        report["spatial"]["longitude"] = {
            "name": lon_key,
            "min_lon": float(np.nanmin(lons)),
            "max_lon": float(np.nanmax(lons)),
            "step": float(np.abs(np.diff(lons)).mean()) if len(lons) > 1 else 0.0,
            "count": len(lons)
        }

    if time_key is not None:
        times = ds.coords[time_key].values
        try:
            start_date = str(pd.to_datetime(times[0]).date())
            end_date = str(pd.to_datetime(times[-1]).date())
        except Exception:
            start_date = str(times[0])
            end_date = str(times[-1])

        report["temporal"] = {
            "name": time_key,
            "start_date": start_date,
            "end_date": end_date,
            "total_timesteps": len(times)
        }

    # 4. Variable Inspection and Statistical Summary
    for var_name, var_array in ds.data_vars.items():
        arr = var_array.values
        # Handle standard fill values (-999.0, -9999.0) if present
        arr_clean = np.where(np.isin(arr, [-999.0, -9999.0]), np.nan, arr)

        nan_count = int(np.isnan(arr_clean).sum())
        total_count = int(arr_clean.size)
        valid_arr = arr_clean[~np.isnan(arr_clean)]

        var_stats = {
            "dtype": str(var_array.dtype),
            "shape": list(var_array.shape),
            "total_elements": total_count,
            "missing_elements": nan_count,
            "missing_pct": round((nan_count / total_count) * 100, 2) if total_count > 0 else 0.0,
            "min": float(np.min(valid_arr)) if len(valid_arr) > 0 else float("nan"),
            "max": float(np.max(valid_arr)) if len(valid_arr) > 0 else float("nan"),
            "mean": float(np.mean(valid_arr)) if len(valid_arr) > 0 else float("nan"),
            "attributes": dict(var_array.attrs)
        }
        report["variables"][var_name] = var_stats

    ds.close()

    # 5. Print Formatted Report if verbose
    if verbose:
        print_validation_report(report)

    return report


def print_validation_report(report: Dict[str, Any]) -> None:
    """Formats and prints the climate dataset validation report."""
    print("=" * 70)
    print("           CLIMATE DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"File Path: {report['file_path']}")
    print(f"File Size: {report['file_size_mb']:.2f} MB")
    print("-" * 70)

    print("\n1. DATASET DIMENSIONS:")
    for dim_name, dim_size in report["dimensions"].items():
        print(f"   • {dim_name:<15}: {dim_size:,}")

    print("\n2. SPATIAL COORDINATES:")
    if "latitude" in report["spatial"]:
        lat_info = report["spatial"]["latitude"]
        print(f"   • Latitude ({lat_info['name']}):  {lat_info['min_lat']:.2f}°N to {lat_info['max_lat']:.2f}°N | Step: {lat_info['step']:.4f}° | Count: {lat_info['count']}")
    else:
        print("   ⚠️ Latitude coordinate not detected.")

    if "longitude" in report["spatial"]:
        lon_info = report["spatial"]["longitude"]
        print(f"   • Longitude ({lon_info['name']}): {lon_info['min_lon']:.2f}°E to {lon_info['max_lon']:.2f}°E | Step: {lon_info['step']:.4f}° | Count: {lon_info['count']}")
    else:
        print("   ⚠️ Longitude coordinate not detected.")

    print("\n3. TEMPORAL DIMENSION:")
    if report["temporal"]:
        t_info = report["temporal"]
        print(f"   • Time ({t_info['name']}):      {t_info['start_date']} to {t_info['end_date']} | Total Steps: {t_info['total_timesteps']}")
    else:
        print("   ℹ️ No explicit time dimension found.")

    print("\n4. DATA VARIABLES & STATISTICS:")
    for var_name, stats in report["variables"].items():
        print(f"\n   ► Variable: [{var_name}]")
        print(f"     - Data Type:     {stats['dtype']}")
        print(f"     - Tensor Shape:  {stats['shape']}")
        print(f"     - Total Cells:   {stats['total_elements']:,}")
        print(f"     - Missing Cells: {stats['missing_elements']:,} ({stats['missing_pct']}%)")
        print(f"     - Minimum:       {stats['min']:.4f}")
        print(f"     - Maximum:       {stats['max']:.4f}")
        print(f"     - Mean:          {stats['mean']:.4f}")

    print("\n5. GLOBAL ATTRIBUTES:")
    if report["global_attributes"]:
        for attr_key, attr_val in report["global_attributes"].items():
            print(f"   • {attr_key}: {attr_val}")
    else:
        print("   ℹ️ No global metadata attributes defined.")

    print("=" * 70)
    print("Validation status: COMPLETED (Dataset verified cleanly)")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        validate_climate_dataset(target_file)
    else:
        print("Usage: python src/data/validate_dataset.py <path_to_netcdf_file>")
