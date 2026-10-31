# Climate Data Pipeline

## 1. Purpose

Process and validate India Meteorological Department (IMD) daily gridded rainfall and temperature data for the Climate Digital Twin project, using a rectangular Maharashtra-area bounding box.

## 2. Data Source and Period

- Source: India Meteorological Department (IMD)
- Period: 1 January 2015 to 31 December 2025
- Total daily records: 4,018
- Rainfall input directory: `data/raw/rainfall/`
- Maximum temperature (Tmax) input directory: `data/raw/tmax/`
- Minimum temperature (Tmin) input directory: `data/raw/tmin/`

## 3. Rainfall Processing

**Input format:** NetCDF (`.nc`)

**Native resolution:** 0.25 degrees

**Output:** `data/processed/rainfall_mm_2015_2025_maharashtra.nc`

- Variable: `rainfall_mm`
- Units: mm
- Grid dimensions: 27 × 33
- Selected coordinate limits: latitude 15.5°–22.1°N, longitude 72.6°–80.9°E

The processing pipeline:
1. Reads each yearly IMD rainfall file for 2015–2025.
2. Selects the requested spatial bounding box.
3. Renames the coordinates to `time`, `lat`, and `lon`.
4. Preserves missing observations as missing values.
5. Creates a `valid_mask` variable: 1 for valid observations and 0 for missing observations.
6. Checks daily date continuity and rejects negative rainfall values.
7. Combines the yearly data and exports the result to NetCDF.

## 4. Temperature Processing

**Input format:** Binary `.GRD` files

**Native resolution:** 1 degree

**Missing-value marker:** `99.9`, converted to `NaN`

**Outputs:**
- `data/processed/tmax_c_2015_2025.nc`
- `data/processed/tmin_c_2015_2025.nc`

Each output contains 4,018 daily records on a 7 × 8 grid.

The requested bounding-box limits are latitude 15.5°–22.1°N and longitude 72.6°–80.9°E. Because the source grid is at 1-degree resolution, the selected temperature coordinates are latitude 15.5°–21.5°N and longitude 73.5°–80.5°E.

The processing pipeline:
1. Reads the binary files as little-endian float32 values.
2. Checks the expected number of daily grid values for each year.
3. Reshapes the values into time, latitude, and longitude dimensions.
4. Converts the `99.9` missing-value marker to `NaN`.
5. Selects the available grid coordinates within the bounding-box limits.
6. Preserves a `valid_mask` variable.
7. Combines the yearly records and exports the results to NetCDF.

No spatial interpolation is performed.

## 5. Validation

### Rainfall

`verify_rainfall.py` checks:
- Expected output dimensions and record count.
- Agreement between the validity mask and rainfall missingness.
- Date alignment against raw data for every year from 2015 through 2025.
- Matching missing-value patterns.
- Exact rainfall values against the raw source data.

`process_rainfall.py` also checks daily date continuity and rejects negative rainfall values during processing.

### Temperature

`verify_temperature.py` checks:
- Dates for each year from 2015 through 2025.
- Missing-value masks against the raw Tmax and Tmin files.
- Valid values and missing observations against the raw data.

The latest reported temperature verification passed for all 11 years for both Tmax and Tmin.

## 6. Scripts

### Rainfall
- `process_rainfall.py` — processes yearly rainfall files, checks date continuity and negative values, and exports the combined dataset.
- `verify_rainfall.py` — compares processed rainfall values, dates, and missingness against raw files.
- `check_rainfall_quality.py` — summarizes rainfall missingness, negative values, and minimum/maximum values by year.
- `validate_rainfall.py` — validates rainfall date continuity; consult the script for its exact checks.

### Temperature
- `inspect_temperature_binary.py` — inspects raw temperature binary data.
- `process_temperature.py` — processes Tmax and Tmin files.
- `verify_temperature.py` — verifies processed temperature dates, masks, and values against raw files.

### General inspection and utilities
- `inspect_data.py` — inspects rainfall NetCDF data.
- `check_mask_consistency.py` — inspects rainfall missing-cell patterns; consult the script for its exact checks.
- `verify_manifest.py` — checks that the manifest is valid JSON and that the listed output files exist.

## 7. Important Notes

- Raw input files must remain unchanged.
- Missing values must not be replaced with zero.
- Rainfall and temperature retain their respective native grid resolutions.
- The spatial subset is a rectangular bounding box, not an exact Maharashtra state-boundary mask.
- Temperature coordinates do not reach every requested bounding-box boundary because the native grid is 1 degree.
- The processed historical data covers 2015–2025; it does not, by itself, provide real-time climate observations.
- The manifest records the project period, spatial subsets, output files, and validation status.
