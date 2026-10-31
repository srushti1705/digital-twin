
from pathlib import Path
import xarray as xr
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "rainfall"

expected_years = list(range(2015, 2026))
all_dates = []
reference_grid = None
errors = []

print("RAINFALL DATA VALIDATION: 2015-2025")
print("=" * 50)

for year in expected_years:
    file_path = DATA_DIR / f"RF25_ind{year}_rfp25.nc"

    if not file_path.exists():
        errors.append(f"{year}: File missing")
        print(f"{year}: FILE MISSING")
        continue

    try:
        with xr.open_dataset(file_path) as ds:
            if "TIME" not in ds.coords:
                errors.append(f"{year}: TIME coordinate missing")
                continue

            if "RAINFALL" not in ds.data_vars:
                errors.append(f"{year}: RAINFALL variable missing")
                continue

            dates = pd.to_datetime(ds["TIME"].values)
            all_dates.extend(dates)

            grid = (
                tuple(ds["LATITUDE"].values),
                tuple(ds["LONGITUDE"].values),
            )

            if reference_grid is None:
                reference_grid = grid
            elif grid != reference_grid:
                errors.append(f"{year}: Grid differs from reference")

            units = ds["RAINFALL"].attrs.get("units", "not specified")
            if units != "mm":
                errors.append(f"{year}: Unexpected units: {units}")

            expected_days = 366 if pd.Timestamp(year, 12, 31).dayofyear == 366 else 365
            actual_days = len(dates)

            if actual_days != expected_days:
                errors.append(
                    f"{year}: Expected {expected_days} days, found {actual_days}"
                )

            if len(dates) and (
                dates[0].year != year or dates[-1].year != year
            ):
                errors.append(f"{year}: Date range does not match filename")

            print(
                f"{year}: {actual_days} days | "
                f"{dates[0].date()} to {dates[-1].date()} | "
                f"Grid: {ds.sizes}"
            )

    except Exception as exc:
        errors.append(f"{year}: Could not read file: {exc}")

print("\n" + "=" * 50)

if all_dates:
    date_series = pd.DatetimeIndex(all_dates).sort_values()
    duplicates = date_series[date_series.duplicated()].unique()
    full_range = pd.date_range("2015-01-01", "2025-12-31", freq="D")
    missing_dates = full_range.difference(date_series)

    print("Total date records:", len(date_series))
    print("Unique dates:", date_series.nunique())
    print("Duplicate dates:", len(duplicates))
    print("Missing dates:", len(missing_dates))

    if len(duplicates):
        errors.append("Duplicate dates found")

    if len(missing_dates):
        errors.append("Missing dates found")

if errors:
    print("\nVALIDATION RESULT: CHECK REQUIRED")
    for error in errors:
        print("-", error)
else:
    print("\nVALIDATION RESULT: PASSED")
