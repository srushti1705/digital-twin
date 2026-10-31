

import xarray as xr
import numpy as np

path = "data/raw/rainfall/RF25_ind2019_rfp25.nc"

with xr.open_dataset(path) as ds:
    rainfall = ds["RAINFALL"].sel(
        LATITUDE=slice(15.5, 22.1),
        LONGITUDE=slice(72.6, 80.9),
    )

    missing_days = rainfall.isnull().sum(dim="TIME")
    always_missing = missing_days == rainfall.sizes["TIME"]

    locations = np.argwhere(always_missing.values)

    print("Always-missing cells:", len(locations))
    print("\nLatitude | Longitude")

    for lat_index, lon_index in locations:
        lat = rainfall["LATITUDE"].values[lat_index]
        lon = rainfall["LONGITUDE"].values[lon_index]
        print(f"{lat:.2f} | {lon:.2f}")
