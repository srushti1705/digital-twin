# AI-Powered Digital Twin of India's Climate using National Datasets

A proof-of-concept digital twin framework designed to model, monitor, predict, and simulate climate parameters across the Indian subcontinent using high-resolution national meteorological datasets.

---

## 1. Project Overview

The **AI-Powered Digital Twin of India's Climate** aims to create a dynamic, spatial-temporal digital representation of atmospheric and weather conditions over India. By integrating multi-source national datasets—starting with India Meteorological Department (IMD) high-resolution gridded observations—the system provides a computational foundation for climate state monitoring, short-term forecasting, data assimilation, and interactive what-if scenario simulation.

---

## 2. Problem Statement

India experiences extreme climate diversity, seasonal monsoonal shifts, localized heatwaves, and severe precipitation events. Traditional climate modeling approaches face challenges in:
- Handling diverse spatial-temporal data formats across national agencies.
- Real-time resolution matching and spatial grid alignment over complex topography.
- Providing intuitive, interactive digital twin capabilities for scenario exploration ("what-if" perturbations).

An AI-driven digital twin bridges these gaps by combining multidimensional geospatial grid processing with machine learning models and interactive visualization.

---

## 3. Objectives

- **Standardized Data Pipeline**: Ingest and unify national climate data formats (IMD binary `.grd`, NetCDF `.nc`, GeoTIFF).
- **Spatial-Temporal Alignment**: Re-grid and align rainfall and temperature datasets across uniform India regional spatial coordinates.
- **AI/ML Forecasting**: Develop predictive architectures (e.g., spatial-temporal ML models) for short-term rainfall and temperature forecasting.
- **Data Assimilation**: Integrate satellite observations and reanalysis data to continuously update the digital twin's state representation.
- **What-If Scenario Simulation**: Enable interactive perturbation testing (e.g., localized temperature spikes, monsoonal rainfall shifts).
- **Geospatial Dashboard**: Render intuitive spatial heatmaps and time-series analytics for policy and research insights.

---

## 4. Initial Proof of Concept (POC)

The initial proof-of-concept focuses on core high-impact variables:
- **Rainfall coverage**: the installed IMD grid spans $6.5^\circ\text{N}$ to $38.5^\circ\text{N}$ and $66.5^\circ\text{E}$ to $100^\circ\text{E}$ at native $0.25^\circ$ resolution.
- **Temperature coverage**: the installed IMD Tmax/Tmin grids span $7.5^\circ\text{N}$ to $37.5^\circ\text{N}$ and $67.5^\circ\text{E}$ to $97.5^\circ\text{E}$ at native $1^\circ$ resolution.
- **Temporal scale**: the currently installed local observations cover 2025 daily; the products are not a live feed.

---

## 5. Climate Parameters

### Phase 1 & Initial Focus
- **Rainfall / Precipitation**: Daily total rainfall ($\text{mm/day}$).
- **Temperature**: Daily Minimum, Maximum, and Mean Surface Temperature ($^\circ\text{C}$).

### Future Integrations
- Relative Humidity ($\%$)
- Wind Vectors ($U$ and $V$ components at $10\text{m}$)
- Surface Pressure ($\text{hPa}$)
- Cloud Cover & INSAT Thermal Brightness Temperature ($K$)

---

## 6. Planned Data Sources

1. **India Meteorological Department (IMD)**: Daily gridded rainfall and temperature datasets (`.grd`, NetCDF).
2. **INSAT Satellite Observations (ISRO/IMD)**: Geostationary satellite imagery (VIS, TIR, MIR channels) for cloud dynamics.
3. **ERA5 Reanalysis (ECMWF / NCMRWF)**: High-resolution atmospheric reanalysis for boundary conditions.

---

## 7. Technology Stack

- **Backend and data processing**: Python, FastAPI, NumPy, xarray, and the IMD native-grid readers.
- **Forecast baselines**: scikit-learn metrics and XGBoost, compared with same-cell persistence.
- **Frontend**: React, TypeScript, Vite, CesiumJS, and Resium.
- **Data formats**: official IMD little-endian float32 `.grd`; optional NetCDF rainfall support in the backend loader.

---

## 8. Project Architecture

```text
IMD annual rainfall / Tmax / Tmin GRD files
  -> backend native-grid readers
  -> FastAPI observation routes
  -> frontend climate API service
  -> Cesium imagery overlays and grid-cell inspection

IMD annual GRD files
  -> chronological one-day model training and evaluation
  -> local ignored model artifacts and evaluation JSON
  -> FastAPI prediction routes
  -> frontend forecast raster and click inspection

NASA GIBS IMERG imagery -> Cesium WMS overlay (visualization only; no numeric query)

INSAT/MOSDAC georeferenced cloud feed and structured RSMC cyclone feed
  -> not currently connected
```

---

## 9. Current Verified Implementation

- React, TypeScript, Vite, CesiumJS, and Resium provide an interactive India-focused globe.
- FastAPI serves date-addressable IMD rainfall grids at native 0.25° and Tmax/Tmin grids at native 1° resolution. All three are backed by locally installed 2025 IMD files, which are excluded from Git.
- A reproducible download script retrieves selected annual IMD rainfall and temperature files from official IMD Pune forms and checks their expected binary sizes.
- One-day model forecasts use a reproducible chronological training and evaluation workflow over the locally available native grids. XGBoost candidates are compared with a persistence baseline and selected only when validation RMSE improves by at least 1% without more than 1% validation MAE degradation. Held-out metrics are saved locally and exposed through the prediction API. Single-year results are preliminary, not operational or current-weather forecasts.
- NASA GPM IMERG is currently available only as NASA GIBS precipitation imagery. Numerical IMERG retrieval requires server-side Earthdata access and is not verified.
- No georeferenced cloud-imagery feed or structured official cyclone track feed is connected. These layers remain unavailable rather than showing fabricated data.

## 10. Run Locally

From the repository root, create the Python environment, install the declared
project and API dependencies, download the official 2025 IMD rainfall/Tmax/Tmin
grids, and train the one-day models:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -r requirements.txt -r backend\requirements.txt
python scripts\download_imd_grids.py --years 2025
Set-Location backend
.\.venv\Scripts\python.exe train_climate_models.py
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

In another terminal, from the repository root:

```powershell
Set-Location frontend
npm ci
npm run dev
```

The frontend defaults to the API at `http://localhost:8000`. Downloaded raw data is
git-ignored under `data/raw/`; trained joblib artifacts and evaluation JSON are
git-ignored under `models/climate_forecasts/` and `outputs/climate_forecasts/`.
They must be downloaded and generated locally and are intentionally not committed.

Run the complete backend test suite from the repository root with:

```powershell
backend\.venv\Scripts\python.exe -m pytest tests -q
```

See `data/DATA_SOURCES.md`, `backend/RAINFALL_API.md`, and
`backend/FORECAST_MODEL_REPORT.md` for source details, API routes, and measured
model results. Numerical NASA IMERG requires server-side Earthdata authentication;
georeferenced cloud imagery and structured cyclone tracks are not integrated.
