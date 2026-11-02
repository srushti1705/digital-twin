# Climate API Integration

Run the API from this directory with `uvicorn main:app --reload`. IMD files are read
from `../data/raw/rainfall/` by default; `IMD_RAINFALL_DIR` can point to another
directory. Raw data files are not tracked in Git.

## IMD Daily Temperature

- `GET /api/climate/temperature/tmax/dates` and
  `GET /api/climate/temperature/tmin/dates` list dates with valid installed grids.
- `GET /api/climate/temperature/{tmax|tmin}?date=YYYY-MM-DD` returns the native
  1° grid in degrees Celsius, coordinates, source attribution and missing-cell counts.
- `GET /api/climate/temperature/{tmax|tmin}/cell?date=YYYY-MM-DD&latitude=19&longitude=75`
  returns the nearest native grid cell. Missing observations return
  `temperature: null` and `missing: true`; locations outside the grid return HTTP 422.

Put official annual little-endian float32 `.grd` files in
`data/raw/max_temp/` and `data/raw/min_temp/`. The annual files are ignored by Git.
`IMD_TEMPERATURE_DIR` may point to a different parent directory containing those two
subdirectories. Grid values use the IMD `99.9` missing sentinel; the loader accounts
for float32 representation when detecting that sentinel.

## IMD Gridded Daily Rainfall

- `GET /api/climate/rainfall/imd/dates` lists installed observation dates.
- `GET /api/climate/rainfall/imd?date=YYYY-MM-DD` returns the native grid,
  coordinates, `mm/day` units, resolution, source attribution and missing-cell counts.
- `GET /api/climate/rainfall/imd/cell?date=YYYY-MM-DD&latitude=19&longitude=75`
  returns the nearest native grid cell. Missing observations return `rainfall: null`
  and `missing: true`; coordinates outside the installed grid return HTTP 422.
- `GET /api/climate/rainfall/sources` reports both independent rainfall sources.

Annual IMD `.grd`, `.nc` and `.nc4` files are supported. The binary loader verifies
annual byte size before reading; NetCDF coordinates and observation dates are read
from the file, not inferred for cell values.

## NASA GPM IMERG

- `GET /api/climate/rainfall/nasa/dates` returns the current time extent advertised by
  NASA GIBS for its daily `IMERG_Precipitation_Rate` layer.
- The frontend requests that public WMS layer directly from Cesium for a date-specific
  rendered overlay. GIBS imagery is visualization-only: the API deliberately does not
  report it as a numeric IMERG grid or provide cell-value inspection.
- NASA GES DISC granule downloads require Earthdata authentication. They returned HTTP
  401 without credentials in this environment, so authenticated numerical NASA data
  access is not implemented. Do not send Earthdata credentials to the frontend.

GIBS availability is refreshed from its GetCapabilities document, cached in the API
for up to 30 minutes, and returned as explicit advertised intervals. A service or
metadata failure returns HTTP 502 rather than a fabricated availability range.

## One-day IMD Prediction Baselines

The prediction API exposes separately trained rainfall, Tmax, and Tmin one-day
grids:

- `GET /api/predictions/catalog` reports installed models and held-out test metrics.
- `GET /api/predictions/{rainfall|tmax|tmin}/dates` lists valid initialization dates
  for the installed model and source observations.
- `GET /api/predictions/{rainfall|tmax|tmin}?initialization_date=YYYY-MM-DD` returns
  a native-resolution grid for the following calendar day.
- `GET /api/predictions/{rainfall|tmax|tmin}/cell?initialization_date=YYYY-MM-DD&latitude=19&longitude=75`
  returns the nearest native grid cell.

Train and evaluate the models from this directory with
`python train_climate_models.py`. The command reads the actual local IMD `.grd`
observations and writes ignored model files under `../models/climate_forecasts/`
and evaluation JSON under `../outputs/climate_forecasts/`. Download missing annual
inputs with `python ../scripts/download_imd_grids.py --years 2025` (the complete
available archive is not required).

Evaluation is chronological: 70% train, 15% validation, 15% held-out test. Seven
preceding daily native-grid observations form each feature history. XGBoost is
selected only when validation RMSE improves by at least 1% and validation MAE
degrades by no more than 1% compared with same-cell persistence; this selection
uses validation data only. The reported test metrics compare the selected model
with persistence on the held-out final period.

Only 2025 observations are installed in the current setup, so these results are
preliminary single-year evaluation, not operational forecast skill. The route
labels output as a historical hindcast or out-of-sample projection, includes the
initialization and forecast dates, and does not imply current weather. Missing
input cells stay missing; neither grid is interpolated or upsampled.
