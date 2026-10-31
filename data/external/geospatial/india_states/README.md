# Indian States & Union Territories GIS Boundary Dataset

## Dataset Specification

This directory stores the official GIS boundary datasets for Indian States and Union Territories for the **Phase 6 3D Digital Twin Application**.

### Directory Structure
```
data/external/geospatial/india_states/
├── india_states_raw.geojson        # Original unmodified GIS boundary GeoJSON
├── india_states_processed.geojson  # Enriched, web-ready GeoJSON with WebGL metadata
└── README.md                        # Documentation & schema reference
```

---

## Technical Features & Property Schema

Each feature in `india_states_processed.geojson` represents an individual State or Union Territory containing:

| Property Key | Type | Description |
| :--- | :--- | :--- |
| `st_nm` | String | Standard State/UT name (e.g., "Maharashtra", "Tamil Nadu") |
| `state_name` | String | Full administrative name |
| `iso_code` | String | Unique ISO 3166-2 state identifier (e.g., "IN-MH", "IN-TN") |
| `TYPE` | String | "State" or "Union Territory" |
| `REGION` | String | Geographical region ("North", "South", "East", "West", "Central", "North-East") |
| `centroid` | Array[2] | `[latitude, longitude]` center point for label positioning |
| `webgl_center` | Array[2] | `[x, y]` coordinates normalized relative to India centroid (22.5°N, 82.5°E) |
| `bbox` | Array[4] | `[min_lat, min_lon, max_lat, max_lon]` bounding box |
| `is_union_territory` | Boolean | `true` if Union Territory, `false` if State |

---

## Coordinate Reference System (CRS)

- **Standard**: **WGS 84 (EPSG:4326)** / OGC CRS84.
- **Geographic Coverage**: Latitude $6.5^\circ\text{N}$ to $38.5^\circ\text{N}$, Longitude $66.5^\circ\text{E}$ to $100.0^\circ\text{E}$.
- **Accuracy Guarantee**: Boundaries are preserved in exact latitude/longitude coordinates without distortion or aggressive vertex reduction.

---

## Legal & Licensing Compliance

- **Source**: DataMeet Community GIS / Survey of India Boundary Standard.
- **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Academic Usage**: Fully authorized for academic research, open-source development, and WebGL digital twin visualizations.
