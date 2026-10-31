# National Climate Data Sources Documentation

This document records the official dataset specifications for the **AI-Powered Digital Twin of India's Climate using National Datasets**. It establishes dataset metadata, spatial-temporal properties, official download URLs, and intended usage in the project.

---

## 1. IMD High-Resolution Gridded Rainfall Dataset

### Confirmed Dataset Properties
- **Parameter**: Daily Precipitation / Rainfall ($\text{mm/day}$)
- **Source Organization**: India Meteorological Department (IMD) / Climate Application & Services Group (CMPG), Pune
- **Official Source URL**: [https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html)
- **Spatial Resolution**: $0.25^\circ \times 0.25^\circ$ latitude/longitude grid
- **Target Local Directory**: `data/raw/rainfall/`

### Expected Properties
- **Expected Temporal Information**: Daily timesteps covering historical monsoon seasons and annual records.
- **Expected Data Format**: Binary `.grd` file format (or NetCDF `.nc` conversion).

### Properties to be Verified After Download
- [ ] Exact grid array dimensions (e.g., $135 \times 129$ lat/lon matrix).
- [ ] Bounding box coordinate origin (start latitude/longitude in $^\circ\text{N}$ and $^\circ\text{E}$).
- [ ] Missing/Fill value encoding over oceans and non-reporting areas (e.g., `-999.0` or `NaN`).
- [ ] Endianness and floating-point precision (`float32` vs `float64`).

### Intended Project Use
- Serves as the primary target variable for monsoon precipitation modeling, spatial pattern mapping, AI/ML short-term rainfall forecasting, and what-if precipitation perturbation simulations.

---

## 2. IMD High-Resolution Maximum Temperature Dataset

### Confirmed Dataset Properties
- **Parameter**: Daily Maximum Surface Temperature ($T_{\text{max}}$, $^\circ\text{C}$)
- **Source Organization**: India Meteorological Department (IMD) / Climate Application & Services Group (CMPG), Pune
- **Official Source URL**: [https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html](https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html)
- **Spatial Resolution**: $1.0^\circ \times 1.0^\circ$ latitude/longitude grid
- **Target Local Directory**: `data/raw/max_temp/`

### Expected Properties
- **Expected Temporal Information**: Daily timesteps across multi-decadal observation records.
- **Expected Data Format**: Binary `.grd` file format (or NetCDF `.nc` conversion).

### Properties to be Verified After Download
- [ ] Exact grid dimensions (e.g., $31 \times 31$ lat/lon grid shape).
- [ ] Bounding box start/end coordinates.
- [ ] Ocean/out-of-bounds fill value representation (e.g., `99.9` or `-999.0`).
- [ ] Scaling factors or unit offsets.

### Intended Project Use
- Provides the baseline for daytime heat dynamics, localized heatwave monitoring, spatial re-gridding to match rainfall grids, AI temperature forecasting, and thermal perturbation testing.

---

## 3. IMD High-Resolution Minimum Temperature Dataset

### Confirmed Dataset Properties
- **Parameter**: Daily Minimum Surface Temperature ($T_{\text{min}}$, $^\circ\text{C}$)
- **Source Organization**: India Meteorological Department (IMD) / Climate Application & Services Group (CMPG), Pune
- **Official Source URL**: [https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html](https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html)
- **Spatial Resolution**: $1.0^\circ \times 1.0^\circ$ latitude/longitude grid
- **Target Local Directory**: `data/raw/min_temp/`

### Expected Properties
- **Expected Temporal Information**: Daily timesteps across multi-decadal observation records.
- **Expected Data Format**: Binary `.grd` file format (or NetCDF `.nc` conversion).

### Properties to be Verified After Download
- [ ] Exact grid dimensions matching $T_{\text{max}}$ grid layout.
- [ ] Bounding box alignment with $T_{\text{max}}$ coordinates.
- [ ] Missing value representation over non-land grid points.
- [ ] Binary record byte structure.

### Intended Project Use
- Used for calculating Diurnal Temperature Range ($DTR = T_{\text{max}} - T_{\text{min}}$), tracking nighttime thermal trends, and building multi-variable climate state vectors for data assimilation.

---

## Summary Matrix

| Dataset Name | Parameter | Resolution | Data Format | Official URL | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IMD Rainfall** | Rainfall ($\text{mm/day}$) | $0.25^\circ \times 0.25^\circ$ | `.grd` / `.nc` | [Rainfall_25_Bin.html](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html) | ⏳ Pending Download |
| **IMD Max Temp** | $T_{\text{max}}$ ($^\circ\text{C}$) | $1.0^\circ \times 1.0^\circ$ | `.grd` / `.nc` | [Max_1_Bin.html](https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html) | ⏳ Pending Download |
| **IMD Min Temp** | $T_{\text{min}}$ ($^\circ\text{C}$) | $1.0^\circ \times 1.0^\circ$ | `.grd` / `.nc` | [Min_1_Bin.html](https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html) | ⏳ Pending Download |
