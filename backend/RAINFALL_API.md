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
