

from pathlib import Path
import calendar

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

CONFIGS = [
    ("tmax", "Maxtemp_MaxT_", "tmax_c"),
    ("tmin", "Mintemp_MinT_", "tmin_c"),
]

MISSING_VALUE = 99.9
LAT_START, LAT_END = 8, 15
LON_START, LON_END = 6, 14

all_passed = True

for folder, prefix, variable in CONFIGS:
    output_path = PROCESSED_DIR / f"{variable}_2015_2025.nc"

    with xr.open_dataset(output_path) as dataset:
        print(f"\nChecking {variable}")

        for year in range(2015, 2026):
            days = 366 if calendar.isleap(year) else 365
            raw_path = RAW_DIR / folder / f"{prefix}{year}.GRD"

            if not raw_path.exists():
                print(f"{year}: FAIL - raw file missing")
                all_passed = False
                continue

            raw = np.fromfile(raw_path, dtype="<f4")

            if raw.size != days * 31 * 31:
                print(f"{year}: FAIL - unexpected raw file size")
                all_passed = False
                continue

            raw = raw.reshape(days, 31, 31)
            expected = raw[:, LAT_START:LAT_END, LON_START:LON_END]

            expected_mask = (
                np.isfinite(expected)
                & ~np.isclose(expected, MISSING_VALUE)
            )
            expected = expected.astype(np.float64)
            expected[~expected_mask] = np.nan

            saved = dataset.sel(time=str(year))
            actual = saved[variable].values
            actual_mask = saved["valid_mask"].values.astype(bool)

            dates_ok = (
                saved.sizes["time"] == days
                and str(saved["time"].values[0])[:10] == f"{year}-01-01"
                and str(saved["time"].values[-1])[:10] == f"{year}-12-31"
            )

            mask_ok = np.array_equal(expected_mask, actual_mask)

            values_ok = np.allclose(
                expected,
                actual,
                rtol=0,
                atol=1e-5,
                equal_nan=True,
            )

            passed = dates_ok and mask_ok and values_ok
            all_passed = all_passed and passed

            print(
                f"{year}: {'PASS' if passed else 'FAIL'} | "
                f"dates={dates_ok} | mask={mask_ok} | values={values_ok}"
            )

print("\nOverall result:", "PASS" if all_passed else "FAIL")

if not all_passed:
    raise SystemExit(1)
