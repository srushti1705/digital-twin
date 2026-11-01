# Legacy Climate Processing Pipeline

The scripts in this directory were written for a 2015–2025 Maharashtra-subset NetCDF processing workflow. That archive and its expected processed NetCDF outputs are not present in the current local installation. The existing manifest and older validation notes describe intended/prior pipeline outputs; they must not be interpreted as proof that those files are currently available or that the 2015–2025 workflow has been rerun against the locally installed data.

## Current locally installed data

The verified local inputs are annual 2025 IMD binary grids:

- Rainfall: `data/raw/rainfall/ind2025_rfp25.grd` (0.25°, 365 daily grids).
- Tmax: `data/raw/max_temp/maxtemp_2025.grd` (1°, 365 daily grids).
- Tmin: `data/raw/min_temp/mintemp_2025.grd` (1°, 365 daily grids).

They are read at their native resolutions by the backend APIs and forecast workflow. They are not processed into the NetCDF outputs described by the legacy scripts. See [`data/DATASET_SUMMARY.md`](../../data/DATASET_SUMMARY.md) and [`data/DATA_SOURCES.md`](../../data/DATA_SOURCES.md) for verified ranges and metadata.

## Legacy scripts and assumptions

`process_rainfall.py`, `process_temperature.py`, and their verification utilities expect yearly inputs covering 2015–2025, particular historical file naming, and/or previously produced NetCDF files. The current downloaded-data locations and filenames differ for some products. Do not run these scripts assuming the archive exists; first obtain the full required years, inspect the script's exact input patterns, and validate its outputs against the original grids.

No interpolation should be used to make the 1° temperature grids appear to have the rainfall grid's 0.25° observational resolution. Missing data must remain missing.

For the currently supported download and one-day forecast workflows, use:

```powershell
python scripts\download_imd_grids.py --years 2025
Set-Location backend
python train_climate_models.py
```

The download command requires access to the official IMD Pune product pages. The training workflow requires all three local 2025 files. Its one-year evaluation is preliminary and does not establish operational forecast skill.
