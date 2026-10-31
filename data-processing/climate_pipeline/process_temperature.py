
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

LATITUDES = np.arange(7.5, 38.0, 1.0)
LONGITUDES = np.arange(67.5, 98.0, 1.0)

MISSING_VALUE = np.float32(99.9)

LAT_MIN, LAT_MAX = 15.5, 22.1
LON_MIN, LON_MAX = 72.6, 80.9


def read_temperature_file(path, year):
    values = np.fromfile(path, dtype="<f4")

    days = 366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365
    expected_values = days * 31 * 31

    if values.size != expected_values:
        raise ValueError(
            f"{path.name}: expected {expected_values} values, "
            f"found {values.size}"
        )

    # IMD's Fortran example loops over J inside I.
    grid = values.reshape(days, 31, 31)

    # Preserve source missingness instead of treating 99.9 as temperature.
    valid_mask = grid != MISSING_VALUE
    grid = grid.astype(np.float32)
    grid[~valid_mask] = np.nan

    dates = pd.date_range(f"{year}-01-01", periods=days, freq="D")

    coords = {
        "time": dates,
        "lat": LATITUDES,
        "lon": LONGITUDES,
    }

    temperature = xr.DataArray(
        grid,
        dims=("time", "lat", "lon"),
        coords=coords,
        name="temperature",
    )

    validity = xr.DataArray(
        valid_mask,
        dims=("time", "lat", "lon"),
        coords=coords,
        name="valid_mask",
    )

    return temperature, validity


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configurations = [
        ("tmax", "Maxtemp_MaxT_", "tmax_c"),
        ("tmin", "Mintemp_MinT_", "tmin_c"),
    ]

    for folder, prefix, variable in configurations:
        yearly_data = []
        yearly_masks = []

        for year in range(2015, 2026):
            path = RAW_DIR / folder / f"{prefix}{year}.GRD"

            if not path.exists():
                raise FileNotFoundError(f"Missing input file: {path}")

            temperature, valid_mask = read_temperature_file(path, year)
            yearly_data.append(temperature.rename(variable))
            yearly_masks.append(valid_mask)

            print(
                f"{path.name}: valid={int(valid_mask.sum().item())}, "
                f"missing={int((~valid_mask).sum().item())}"
            )

        combined = xr.concat(yearly_data, dim="time")
        combined_mask = xr.concat(yearly_masks, dim="time")

        subset = combined.sel(
            lat=slice(LAT_MIN, LAT_MAX),
            lon=slice(LON_MIN, LON_MAX),
        )
        subset_mask = combined_mask.sel(
            lat=slice(LAT_MIN, LAT_MAX),
            lon=slice(LON_MIN, LON_MAX),
        )

        dataset = subset.to_dataset()
        dataset["valid_mask"] = subset_mask
        dataset[variable].attrs.update({
            "units": "degrees_Celsius",
            "source": "India Meteorological Department",
            "missing_value": "99.9 converted to NaN",
        })
        dataset["valid_mask"].attrs["description"] = (
            "True where the original temperature value is not 99.9"
        )

        dataset.attrs.update({
            "source": "IMD daily gridded temperature data",
            "period": "2015-2025",
            "grid_resolution": "1 degree",
            "spatial_subset": "Maharashtra bounding box",
        })

        output_path = OUTPUT_DIR / f"{variable}_2015_2025.nc"
        dataset.to_netcdf(output_path)

        print(f"Saved: {output_path}")
        print(f"Output dimensions: {dict(dataset.sizes)}")


if __name__ == "__main__":
    main()
