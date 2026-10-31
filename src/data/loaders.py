"""
Climate Dataset Loaders Module
------------------------------
Reusable, validated functions for loading climate datasets (NetCDF, CSV, IMD binary .grd files)
into standardized xarray Datasets or pandas DataFrames.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import xarray as xr


def load_netcdf(
    file_path: Union[str, Path],
    engine: Optional[str] = "netcdf4",
    chunks: Optional[dict] = None
) -> xr.Dataset:
    """
    Load a NetCDF climate dataset into an xarray.Dataset with validation.

    Parameters:
        file_path (str | Path): Path to the .nc or .nc4 file.
        engine (str, optional): Engine for opening NetCDF file. Defaults to 'netcdf4'.
        chunks (dict, optional): Chunk size for dask lazy loading (e.g., {'time': 10}).

    Returns:
        xr.Dataset: The loaded multidimensional climate dataset.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"❌ Climate dataset file not found at: '{path}'")

    if not path.is_file():
        raise ValueError(f"❌ Specified path '{path}' is a directory, not a file.")

    valid_extensions = {".nc", ".nc4", ".cdf"}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"⚠️ Unexpected extension '{path.suffix}' for NetCDF loading. Expected: {valid_extensions}")

    try:
        dataset = xr.open_dataset(path, engine=engine, chunks=chunks)
        return dataset
    except Exception as err:
        raise IOError(f"❌ Failed to parse NetCDF dataset at '{path}'. Error: {err}") from err


def load_csv(
    file_path: Union[str, Path],
    parse_dates: Optional[list] = None,
    index_col: Optional[Union[str, int]] = None,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Load a CSV climate dataset into a pandas.DataFrame with validation.

    Parameters:
        file_path (str | Path): Path to the .csv file.
        parse_dates (list, optional): Columns to parse as datetime objects.
        index_col (str | int, optional): Column to set as DataFrame index.
        encoding (str): Text encoding (default: 'utf-8').

    Returns:
        pd.DataFrame: The loaded tabular climate dataframe.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"❌ CSV dataset file not found at: '{path}'")

    if not path.is_file():
        raise ValueError(f"❌ Specified path '{path}' is a directory, not a file.")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"⚠️ Unexpected file extension '{path.suffix}'. Expected '.csv'.")

    if path.stat().st_size == 0:
        raise ValueError(f"❌ File at '{path}' is empty (0 bytes).")

    try:
        df = pd.read_csv(path, parse_dates=parse_dates, index_col=index_col, encoding=encoding)
        return df
    except Exception as err:
        raise IOError(f"❌ Failed to parse CSV climate data at '{path}'. Error: {err}") from err


def load_imd_binary(
    file_path: Union[str, Path],
    var_name: str,
    n_lat: int,
    n_lon: int,
    start_lat: float,
    start_lon: float,
    res: float,
    fill_val: float,
    year: int = 2020
) -> xr.Dataset:
    """
    Converts raw 4-byte unformatted binary IMD climate datasets (.grd / .GRD)
    into a labeled, multidimensional xarray.Dataset.

    Parameters:
        file_path (str | Path): Path to binary .grd file.
        var_name (str): Target variable name ('rain', 'tmax', 'tmin').
        n_lat (int): Number of latitude grid points.
        n_lon (int): Number of longitude grid points.
        start_lat (float): Starting latitude coordinate in degrees North.
        start_lon (float): Starting longitude coordinate in degrees East.
        res (float): Grid spatial resolution in degrees.
        fill_val (float): Raw fill value indicating missing/ocean cells (-999.0 or 99.9).
        year (int): Calendar year of observation (default: 2020).

    Returns:
        xr.Dataset: The loaded xarray Dataset with time, lat, lon dimensions.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"❌ Binary IMD file not found at: '{path}'")

    n_days = 366 if (year % 4 == 0) else 365

    with open(path, "rb") as f:
        raw_bytes = f.read()

    data_flat = np.frombuffer(raw_bytes, dtype=np.float32)
    expected_size = n_days * n_lat * n_lon

    if len(data_flat) != expected_size:
        raise ValueError(
            f"❌ Byte count mismatch for '{path.name}'. "
            f"Expected {expected_size} float32 elements ({n_days} days x {n_lat} lat x {n_lon} lon), "
            f"got {len(data_flat)}."
        )

    tensor = data_flat.reshape((n_days, n_lat, n_lon))

    lats = np.around(np.arange(start_lat, start_lat + n_lat * res, res), decimals=4)
    lons = np.around(np.arange(start_lon, start_lon + n_lon * res, res), decimals=4)
    times = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")

    ds = xr.Dataset(
        data_vars={
            var_name: (["time", "lat", "lon"], tensor, {
                "units": "mm/day" if "rain" in var_name else "degC",
                "fill_value": fill_val,
                "long_name": f"IMD Daily {var_name.upper()}"
            })
        },
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={
            "title": f"IMD Raw {var_name.upper()} Dataset ({year})",
            "source": "India Meteorological Department (IMD)",
            "spatial_resolution": f"{res}° x {res}°",
            "temporal_resolution": "Daily"
        }
    )
    return ds


class IMDDataLoader:
    """
    Helper loader class for automatically discovering and reading IMD rainfall and temperature files.
    """

    def __init__(self, base_data_dir: Union[str, Path]):
        self.base_dir = Path(base_data_dir).resolve()

    def find_raw_file(self, dataset_type: str, year: int = 2020) -> Path:
        """Finds raw file matching dataset type ('rainfall', 'max_temperature', 'min_temperature')."""
        target_dir = self.base_dir / dataset_type
        if not target_dir.exists():
            # Fall back to root base_dir if subdirectories don't exist
            target_dir = self.base_dir

        pattern_map = {
            "rainfall": f"*{year}*.grd",
            "max_temperature": f"*{year}*.GRD",
            "min_temperature": f"*{year}*.GRD"
        }

        pattern = pattern_map.get(dataset_type.lower(), f"*{year}*")
        matches = list(target_dir.rglob(pattern))
        if not matches:
            # Case-insensitive check
            matches = [f for f in target_dir.rglob("*") if str(year) in f.name and f.suffix.lower() in (".grd", ".binary")]

        if not matches:
            raise FileNotFoundError(f"No raw IMD binary file found for '{dataset_type}' year {year} under '{target_dir}'")

        # Select specific file matching dataset type if multiple exist
        if dataset_type.lower() in ("max_temperature", "maxtemp"):
            match = next((f for f in matches if "max" in f.name.lower()), matches[0])
        elif dataset_type.lower() in ("min_temperature", "mintemp"):
            match = next((f for f in matches if "min" in f.name.lower()), matches[0])
        else:
            match = matches[0]

        return match

    def load_rainfall_grd(self, file_path: Union[str, Path], year: int = 2020) -> xr.Dataset:
        """Loads 0.25-degree IMD daily rainfall binary file."""
        return load_imd_binary(
            file_path=file_path,
            var_name="rain",
            n_lat=129,
            n_lon=135,
            start_lat=6.5,
            start_lon=66.5,
            res=0.25,
            fill_val=-999.0,
            year=year
        )

    def load_temperature_grd(self, file_path: Union[str, Path], year: int = 2020, var_name: str = "tmax") -> xr.Dataset:
        """Loads 1.0-degree IMD daily maximum or minimum temperature binary file."""
        return load_imd_binary(
            file_path=file_path,
            var_name=var_name,
            n_lat=31,
            n_lon=31,
            start_lat=7.5,
            start_lon=67.5,
            res=1.0,
            fill_val=99.9,
            year=year
        )


def load_climate_data(
    file_path: Union[str, Path]
) -> Union[xr.Dataset, pd.DataFrame]:
    """Unified auto-loader function for NetCDF, CSV, or binary climate files."""
    path = Path(file_path).resolve()
    suffix = path.suffix.lower()

    if suffix in {".nc", ".nc4", ".cdf"}:
        return load_netcdf(path)
    elif suffix == ".csv":
        return load_csv(path)
    elif suffix == ".grd":
        if "rain" in path.name.lower():
            return load_imd_binary(path, "rain", 129, 135, 6.5, 66.5, 0.25, -999.0)
        elif "max" in path.name.lower():
            return load_imd_binary(path, "tmax", 31, 31, 7.5, 67.5, 1.0, 99.9)
        elif "min" in path.name.lower():
            return load_imd_binary(path, "tmin", 31, 31, 7.5, 67.5, 1.0, 99.9)
        else:
            raise ValueError(f"Unknown IMD binary file structure for '{path.name}'.")
    else:
        raise ValueError(f"❌ Unsupported climate data file extension '{suffix}'.")


if __name__ == "__main__":
    print("=== Climate Data Loaders Module ===")
    print("Module initialized. Available functions:")
    print("  - load_netcdf(), load_csv(), load_imd_binary(), load_climate_data()")
    print("  - IMDDataLoader class")
