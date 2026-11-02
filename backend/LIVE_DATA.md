# Live Climate Data Ingestion Layer

## 1. Overview

The Live Climate Data Ingestion Layer integrates near-real-time atmospheric feeds into the Digital Twin of India's Climate. The architecture adheres to strict scientific honesty principles: genuine observations are separated from numerical weather prediction (NWP) model analyses, missing values are preserved as `null` rather than `0.0`, and unauthenticated or downstream-unavailable sources fail gracefully with clear status signals rather than fabricated data.

---

## 2. Ingestion Architecture & Providers

```
                                  +------------------------------+
                                  |   FastAPI Live API Routes    |
                                  |       (/api/live/*)          |
                                  +--------------+---------------+
                                                 |
                                  +--------------v---------------+
                                  |     LiveDataService          |
                                  |  (Provider Orchestration)    |
                                  +--+--------+---------+-----+--+
                                     |        |         |     |
      +------------------------------+        |         |     +-------------------------+
      |                                       |         |                               |
+-----v-------------+   +---------------------v---+   +-v-------------------+   +-------v----------------+
|    IMD Adapter    |   |    NASA GIBS Adapter    |   |  NASA IMERG Adapter |   |   Open-Meteo Adapter   |
| (api.imd.gov.in)  |   | (gibs.earthdata.nasa)   |   | (urs.earthdata.nasa)|   |  (api.open-meteo.com)  |
| Official Stations |   | Time-Enabled Satellite  |   | Numerical Precip    |   | Supplementary NWP Feed |
| Status: Auth req  |   | MODIS / VIIRS / IMERG   |   | Early/Late/Final    |   | 36 State/UT Capitals   |
+-------------------+   +-------------------------+   +---------------------+   +------------------------+
```

### A. India Meteorological Department (IMD)
* **API Documentation**: [IMD API Reference](https://api.imd.gov.in/public/api_reference.html)
* **Access Requirements**:
  * Registration with IMD's National Data Centre / API Portal.
  * Server-side API key passed via headers or query parameters (`IMD_API_KEY`).
  * IP whitelisting or domain authorization may be enforced by IMD.
* **Endpoints Supported**:
  * Station weather observations (`/v1/weather/current`, `/v1/station/all`)
  * District and state rainfall summaries (`/v1/rainfall/district`, `/v1/rainfall/state`)
  * Weather warnings and nowcasts (`/v1/nowcast/district`)
* **Behavior when Unconfigured**:
  * If `IMD_API_KEY` is not present in `.env`, the adapter reports `status: "credentials_required"`, `is_available: false`.
  * The digital twin **never fabricates** station observations in production.

### B. NASA GIBS (Global Imagery Browse Services)
* **API Documentation**: [NASA GIBS API Docs](https://nasa-gibs.github.io/gibs-api-docs/)
* **Protocol**: OGC Web Map Service (WMS) / WMTS over EPSG:4326.
* **WMS Endpoint**: `https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi`
* **Supported Imagery Products**:
  1. `MODIS_Terra_CorrectedReflectance_TrueColor` (Daily true-color optical imagery, 250m).
  2. `MODIS_Aqua_Cloud_Top_Temp_Day` (Infrared cloud-top temperature brightness).
  3. `IMERG_Precipitation_Rate` (Global Precipitation Measurement precipitation rate visual raster).
  4. `VIIRS_SNPP_CorrectedReflectance_TrueColor` (High-resolution day-night band optical imagery, 375m).
* **Cesium Integration**:
  * Layers are loaded dynamically via Cesium's `WebMapServiceImageryProvider`.
  * Imagery is overlaid directly on the existing 3D realistic globe with adjustable alpha (transparency).
  * Date-addressable via WMS `TIME=YYYY-MM-DD` query parameter.

### C. NASA GPM IMERG (Numerical Data)
* **Product Types**:
  * **Early Run**: ~4 hours latency, low-latency precipitation estimation.
  * **Late Run**: ~14 hours latency, improved calibration with gauge inputs.
  * **Final Run**: ~3.5 months latency, research-quality with full monthly gauge calibration.
* **Authentication**: Requires NASA Earthdata Login account (`EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`, or `EARTHDATA_TOKEN`).
* **Distinction**: GIBS imagery is a rendered visualization raster; numerical IMERG is gridded HDF5/NetCDF precipitation rate ($\text{mm/hr}$). The twin makes this distinction transparent in both API contracts and UI labels.

### D. Open-Meteo Supplementary Weather Feed
* **Role**: Supplementary regional atmospheric telemetry covering 36 Indian state and Union Territory capitals.
* **Scientific Label**: Strictly designated with `is_observation: false` and `source_product: "ECMWF/GFS/NWP NWP model assimilation and analysis"`. It is never represented as physical IMD automated weather station (AWS) ground truth.
* **Parameters Retrieved**:
  * Temperature ($^\circ\text{C}$ at 2m)
  * Relative humidity ($\%$)
  * Apparent / "feels like" temperature ($^\circ\text{C}$)
  * Precipitation ($\text{mm}$)
  * Wind speed ($\text{km/h}$) and wind direction ($^\circ$)
  * Weather code (WMO interpretation)
* **Cache Strategy**: Thread-safe in-memory cache with configurable TTL (default: 900 seconds / 15 minutes). Features stale-while-revalidate fallback: if upstream network fails, the latest valid cached telemetry is returned marked with `data_status: "stale"`.

---

## 3. Normalized Live Data Schema

Every live observation adheres to the Pydantic schema defined in `backend/live_data/schemas.py`:

```json
{
  "variable": "temperature",
  "value": 31.4,
  "units": "°C",
  "latitude": 28.6139,
  "longitude": 77.2090,
  "location_name": "New Delhi",
  "source": "Open-Meteo",
  "source_product": "ECMWF/GFS/NWP atmospheric assimilation",
  "observation_time_utc": "2026-11-02T12:00:00Z",
  "fetched_at_utc": "2026-11-02T12:05:12Z",
  "processing_time_utc": null,
  "data_status": "fresh",
  "quality_flag": "good",
  "is_observation": false,
  "is_forecast": false,
  "spatial_resolution": "0.1° (~11km)",
  "temporal_resolution": "hourly"
}
```

---

## 4. REST API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/live/status` | GET | Status of all live providers (health, auth status, freshness, last fetch). |
| `/api/live/weather` | GET | Normalized weather records for Indian state/UT capitals (supports `min_lat`, `max_lat`, `min_lon`, `max_lon`). |
| `/api/live/rainfall` | GET | Live/near-real-time rainfall observations (supports date filter). |
| `/api/live/satellite/layers` | GET | Catalog of available NASA GIBS satellite products and valid date bounds. |
| `/api/live/satellite/imagery` | GET | Imagery metadata, WMS URL, and temporal constraints for a selected product and date. |
| `/api/live/coverage` | GET | Geographical bounding box, total stations, and update intervals. |

---

## 5. Configuration & Environment Variables

Create or edit `backend/.env`:

```env
# IMD API Key (obtain from https://api.imd.gov.in)
IMD_API_KEY=your_imd_api_key_here
IMD_API_BASE_URL=https://api.imd.gov.in/v1

# NASA Earthdata (obtain from https://urs.earthdata.nasa.gov)
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password
EARTHDATA_TOKEN=your_token

# Open-Meteo cache TTL in seconds (default 900)
OPEN_METEO_CACHE_TTL_SECONDS=900
```

