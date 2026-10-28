from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "rainfall"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall_mm_2015_2025_maharashtra.nc"
)

LAT_MIN, LAT_MAX = 15.5, 22.1
LON_MIN, LON_MAX = 72.6, 80.9

with xr.open_dataset(OUTPUT_FILE) as processed:
    rainfall_out = processed["rainfall_mm"]
    mask_out = processed["valid_mask"]

    assert rainfall_out.sizes["time"] == 4018
    assert rainfall_out.sizes["lat"] == 27
    assert rainfall_out.sizes["lon"] == 33

    assert np.array_equal(
        mask_out.values,
        rainfall_out.notnull().values.astype("uint8"),
    ), "Output mask does not match rainfall missingness."

    for year in range(2015, 2026):
        raw_file = RAW_DIR / f"RF25_ind{year}_rfp25.nc"

        with xr.open_dataset(raw_file) as raw:
            expected = raw["RAINFALL"].sel(
                LATITUDE=slice(LAT_MIN, LAT_MAX),
                LONGITUDE=slice(LON_MIN, LON_MAX),
            )

            actual = rainfall_out.sel(
                time=slice(f"{year}-01-01", f"{year}-12-31")
            )

            assert np.array_equal(
                expected["TIME"].values,
                actual["time"].values,
            ), f"{year}: dates do not match."

            assert np.array_equal(
                expected.isnull().values,
                actual.isnull().values,
            ), f"{year}: missing-value pattern changed."

            np.testing.assert_allclose(
                expected.values,
                actual.values,
                rtol=0,
                atol=0,
                equal_nan=True,
                err_msg=f"{year}: rainfall values changed.",
            )

        print(f"{year}: values, dates, and missingness verified.")

print("\nRAINFALL VERIFICATION PASSED")