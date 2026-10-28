
from pathlib import Path

import xarray as xr
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "rainfall"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

START_YEAR = 2015
END_YEAR = 2025

LAT_MIN, LAT_MAX = 15.5, 22.1
LON_MIN, LON_MAX = 72.6, 80.9

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

yearly_data = []

for year in range(START_YEAR, END_YEAR + 1):
    file_path = RAW_DIR / f"RF25_ind{year}_rfp25.nc"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing rainfall file: {file_path}")

    with xr.open_dataset(file_path) as source:
        rainfall = source["RAINFALL"].sel(
            LATITUDE=slice(LAT_MIN, LAT_MAX),
            LONGITUDE=slice(LON_MIN, LON_MAX),
        ).load()

        rainfall = rainfall.rename(
            {
                "TIME": "time",
                "LATITUDE": "lat",
                "LONGITUDE": "lon",
            }
        )

        rainfall = rainfall.rename("rainfall_mm")
        rainfall.attrs["units"] = "mm"
        rainfall.attrs["long_name"] = "Daily rainfall"

        dataset = rainfall.to_dataset()
        dataset["valid_mask"] = rainfall.notnull().astype("uint8")
        dataset["valid_mask"].attrs["long_name"] = (
            "Valid rainfall observation: 1=valid, 0=missing"
        )

        yearly_data.append(dataset)

        print(
            f"{year}: {rainfall.sizes['time']} days, "
            f"grid={rainfall.sizes['lat']} x {rainfall.sizes['lon']}"
        )

combined = xr.concat(yearly_data, dim="time").sortby("time")

expected_dates = pd.date_range(
    f"{START_YEAR}-01-01",
    f"{END_YEAR}-12-31",
    freq="D",
)

actual_dates = pd.DatetimeIndex(combined["time"].values)

if not actual_dates.equals(expected_dates):
    raise ValueError("Rainfall output dates are missing, duplicated, or out of order.")

if bool((combined["rainfall_mm"] < 0).any().compute().item()):
    raise ValueError("Negative rainfall values found.")

output_path = OUTPUT_DIR / "rainfall_mm_2015_2025_maharashtra.nc"

combined.attrs["description"] = (
    "Daily IMD rainfall subset for the Maharashtra bounding box; "
    "native 0.25-degree grid; missing values preserved."
)
combined.attrs["source_period"] = "2015-2025"
combined.attrs["spatial_subset"] = (
    f"lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}"
)

combined.to_netcdf(output_path)

print("\nRAINFALL PROCESSING PASSED")
print("Output:", output_path)
print("Date records:", combined.sizes["time"])
print("Grid:", combined.sizes["lat"], "x", combined.sizes["lon"])
print("Missing values:", int(combined["rainfall_mm"].isnull().sum().item()))
print("Negative values: 0")
