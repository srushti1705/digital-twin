# India Climate Digital Twin

An interactive web app for exploring India's historical climate data, one-day-ahead model estimates, available live weather information, and simple climate scenarios on a 3D globe.

---

## 1. Project Overview

The project combines daily India Meteorological Department (IMD) rainfall and temperature grids with model estimates and external weather and satellite feeds. A Python API supplies data and calculations to a React dashboard, where users explore the results on an interactive globe. It is a research and demonstration tool, not an operational forecasting system.

---

## 2. Problem Statement

India experiences extreme climate diversity, seasonal monsoonal shifts, localized heatwaves, and severe precipitation events. Traditional climate modeling approaches face challenges in:
- Handling diverse spatial-temporal data formats across national agencies.
- Real-time resolution matching and spatial grid alignment over complex topography.
- Providing intuitive, interactive digital twin capabilities for scenario exploration ("what-if" perturbations).

An AI-driven digital twin bridges these gaps by combining multidimensional geospatial grid processing with machine learning models and interactive 3D visualization.

---

## 3. Features

- **Historical climate:** Select a date and display IMD rainfall, maximum temperature, or minimum temperature on the globe. Click the map to inspect grid-cell values.
- **One-day-ahead estimates:** View model-generated rainfall and temperature grids. The backend evaluates XGBoost against a persistence baseline and uses validation results to select a model. Estimates are not observations or official forecasts.
- **Latest available weather:** See weather records for Indian state and Union Territory capitals, provider status, and NASA GIBS satellite imagery when available. Model-derived weather is labeled separately from station observations.
- **What-if scenarios:** Select a date or prediction baseline, choose India or a state/UT, and adjust rainfall by a percentage and temperature by degrees. The map can display the baseline, scenario, or difference.
- **Interactive map:** Explore the globe, search for places, view state boundaries and city labels, and select climate layers.

Scenario changes are mathematical adjustments to existing data; they do not simulate atmospheric physics or produce an official forecast.

---

## 4. Initial Proof of Concept (POC)

The initial proof-of-concept focuses on core high-impact variables:
- **Rainfall coverage**: the installed IMD grid spans $6.5^\circ\text{N}$ to $38.5^\circ\text{N}$ and $66.5^\circ\text{E}$ to $100^\circ\text{E}$ at native $0.25^\circ$ resolution.
- **Temperature coverage**: the installed IMD Tmax/Tmin grids span $7.5^\circ\text{N}$ to $37.5^\circ\text{N}$ and $67.5^\circ\text{E}$ to $97.5^\circ\text{E}$ at native $1^\circ$ resolution.
- **Temporal scale**: the verified local IMD files cover daily observations for 2025. Other years can be downloaded separately. The live weather feed covers 36 state/UT capitals; satellite products have their own date ranges.

---

## 5. Climate Parameters

### Core Variables
- **Rainfall / Precipitation**: Daily total rainfall ($\text{mm/day}$).
- **Temperature**: Daily minimum and maximum temperature ($^\circ\text{C}$).

### Live & Supplementary Variables
- Relative Humidity ($\%$)
- Wind Vectors & Speed ($\text{km/h}$, degrees)
- Apparent Temperature ($^\circ\text{C}$)
- Satellite Optical Reflectance & Cloud-Top Temperature Brightness ($K$)

---

## 6. Data Sources

- **IMD:** Local historical rainfall (`0.25°`) and maximum/minimum temperature (`1°`) grids. The official live IMD API adapter requires an API key and may require provider authorization.
- **Open-Meteo:** Supplementary weather analysis for state/UT capitals. These are model-derived conditions, not station observations.
- **NASA GIBS:** Date-enabled satellite imagery displayed as a visual map overlay; it does not provide queryable rainfall grid values.
- **NASA GPM IMERG:** Numerical precipitation products may require NASA Earthdata credentials.

Provider availability, credentials, dates, and network access determine which live layers are available at runtime. The app reports unavailable or credential-gated sources rather than filling missing measurements with fabricated values.

---

## 7. Technology Stack

- **Frontend:** React 18, TypeScript, Vite, CesiumJS, and Resium; Lucide icons are used for interface controls.
- **Backend/API:** Python, FastAPI, Pydantic, and Uvicorn.
- **Climate data and models:** NumPy, pandas, xarray, scikit-learn, joblib, and XGBoost. Matplotlib is used for state-boundary masking.
- **Data and maps:** IMD little-endian float32 `.grd` grids, GeoJSON state boundaries, and NASA GIBS imagery over WMS.

---

## 8. Project Architecture

```text
Historical IMD annual rainfall / Tmax / Tmin GRD files
  -> backend native-grid readers
  -> FastAPI observation routes (/api/climate/rainfall, /api/climate/temperature)
  -> frontend climate API service
  -> Cesium imagery overlays and grid-cell inspection

IMD annual GRD files
  -> chronological one-day model training and evaluation
  -> local ignored model artifacts and evaluation JSON
  -> FastAPI prediction routes (/api/predictions)
  -> frontend forecast raster and click inspection

Live atmospheric feeds & external ingestion layer
  -> IMD API adapter (official station telemetry; credentials required)
  -> Open-Meteo supplementary adapter (36 Indian state/UT capitals; NWP analysis)
  -> NASA GIBS WMS adapter (time-enabled MODIS, VIIRS, IMERG precipitation rate imagery)
  -> NASA GPM IMERG numerical adapter (Earthdata URS credentials required)
  -> FastAPI live routes (/api/live/*)
  -> frontend live telemetry ticker & satellite WMS overlay

What-If Scenario Simulation Engine
  -> baseline loader (observation or prediction mode)
  -> geoBoundaries ADM1 polygon ray-casting (all-India & state masking)
  -> mathematical perturbations (multiplicative rainfall, additive Tmax/Tmin)
  -> physical consistency checks (tropospheric inversion detection)
  -> FastAPI scenario routes (/api/scenarios/*)
  -> frontend scenario dock with baseline, scenario, and difference map layers
```

---

## 9. Current Verified Implementation

- **Interactive 3D Globe**: React, TypeScript, Vite, CesiumJS, and Resium render an India-centered 3D globe with the realistic Earth base imagery, dark space background, atmospheric blue glow, state boundaries, and major city labels.
- **Four view modes**:
  1. **Historical**: Explore recorded IMD rainfall (native 0.25°) and Tmax/Tmin (native 1°) by date.
  2. **Live**: View the latest available station observations or explicitly identified model-derived conditions, provider health, and time-enabled NASA GIBS imagery.
  3. **Prediction**: Explore one-day-ahead XGBoost or persistence-baseline estimates across available grid cells.
  4. **What-If**: Perturb historical or supported prediction baselines and compare scenario results with spatial difference overlays.
- **Mode workflows**: Use the Layers panel to choose geographic overlays and climate fields, and the Details panel to review data for the selected mode. Click grid cells to inspect values. Predictions include model/source information and evaluation metrics. What-if results show the baseline, adjusted values, and differences. On narrower screens, Layers and Details open one at a time to keep the globe controls accessible.
- **Scientific Honesty & Separation of Truth**:
  - Live station data and NWP model assimilation (Open-Meteo) are strictly separated. Open-Meteo records are explicitly flagged `is_observation: false` to avoid misrepresenting numerical analysis as IMD automated weather station (AWS) ground truth.
  - The What-If engine carries an explicit non-GCM disclaimer: perturbations represent mathematical diagnostic shifts on empirical baselines and do not simulate coupled general atmospheric circulation dynamics.
  - Unconfigured external sources (e.g. IMD API without `IMD_API_KEY`, NASA Earthdata without login credentials) report clear `credentials_required` statuses rather than fabricating synthetic data in production.

---

## 10. Run Locally

From the repository root, create the Python environment, install dependencies, download the IMD 2025 grids, train the prediction models, and start the backend:

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

The frontend runs at `http://localhost:5173` and communicates with the backend at `http://localhost:8000`.

Run the backend test suite from the repository root:

```powershell
backend\.venv\Scripts\python.exe -m pytest tests -q
```

Run the frontend linting and build checks:

```powershell
cd frontend
npm run lint
npm run build
```

See `backend/LIVE_DATA.md` for live providers, API endpoints, and credentials; `backend/SCENARIO_ENGINE.md` for scenario calculations and geographic masking; and `data/DATASET_SUMMARY.md`, `data/DATA_SOURCES.md`, `backend/RAINFALL_API.md`, and `backend/FORECAST_MODEL_REPORT.md` for dataset coverage, sources, API details, and model evaluation.

See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for the prepared Render configuration, external dataset/model storage strategy, environment variables, and deployment limitations. No hosting resources are created by these instructions.
