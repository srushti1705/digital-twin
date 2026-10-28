from pathlib import Path
import xarray as xr

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data" / "raw" / "rainfall"

for year in range(2015, 2026):
    file_path = data_dir / f"RF25_ind{year}_rfp25.nc"

    with xr.open_dataset(file_path) as ds:
        rainfall = ds["RAINFALL"].sel(
            LATITUDE=slice(15.5, 22.1),
            LONGITUDE=slice(72.6, 80.9),
        )

        print(
            f"{year}: "
            f"shape={rainfall.shape}, "
            f"missing={int(rainfall.isnull().sum().values)}, "
            f"negative={int((rainfall < 0).sum().values)}, "
            f"min={float(rainfall.min(skipna=True).values):.2f}, "
            f"max={float(rainfall.max(skipna=True).values):.2f}"
        )