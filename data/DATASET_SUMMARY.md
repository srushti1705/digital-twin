# Verified Local IMD Dataset Summary

This report describes the raw annual IMD files currently installed locally. It is not a claim that the project has the full IMD archive. The `.grd` files are binary arrays; time, latitude, longitude and units are interpreted from the official product specification and the backend readers, not embedded as NetCDF metadata.

## Installed files and validation

| Variable | Local file (git-ignored) | Date coverage | Native grid | Units | Missing sentinel | Valid values over 2025 |
|---|---|---|---|---|---|---|
| Daily rainfall | `data/raw/rainfall/ind2025_rfp25.grd` | 2025-01-01–2025-12-31 (365 days) | 129 × 135; 0.25°; 6.5–38.5°N, 66.5–100°E | mm/day | -999.0 | min 0.0; max 469.21014; mean 3.47396 |
| Daily maximum temperature | `data/raw/max_temp/maxtemp_2025.grd` | 2025-01-01–2025-12-31 (365 days) | 31 × 31; 1°; 7.5–37.5°N, 67.5–97.5°E | °C | 99.9 | min 5.30674; max 46.15256; mean 30.60410 |
| Daily minimum temperature | `data/raw/min_temp/mintemp_2025.grd` | 2025-01-01–2025-12-31 (365 days) | 31 × 31; 1°; 7.5–37.5°N, 67.5–97.5°E | °C | 99.9 | min -2.45981; max 31.34692; mean 19.19441 |

All three files were opened as little-endian float32 arrays and checked against their expected dimensions and byte sizes. The values above exclude missing sentinels. Missing-cell counts are 4,544,615 of 6,356,475 rainfall cells, 221,213 of 350,765 Tmax cells, and 221,221 of 350,765 Tmin cells.

## Interpretation and limitations

- Rainfall and temperature remain on their distinct native grids. Temperature is not resampled to the rainfall grid.
- These are daily historical observations, not live weather.
- The project has only 2025 files installed locally. Other years can be downloaded with `scripts/download_imd_grids.py`; the legacy processing pipeline's 2015–2025 outputs are not present merely because its documentation or manifest lists them.
- Official IMD product pages: [rainfall](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html), [Tmax](https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html), and [Tmin](https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html).
- Forecast models are trained separately from the observed-data endpoints. Their evaluation is preliminary because only one year is available; see the training report JSON generated under `outputs/climate_forecasts/`.
