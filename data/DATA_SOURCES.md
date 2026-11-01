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

### Verified Local Observation
- **Installed file**: `data/raw/rainfall/ind2025_rfp25.grd` (local and git-ignored; not committed).
- **Coverage**: 2025-01-01 through 2025-12-31 (365 daily grids).
- **Grid shape**: 129 latitude rows × 135 longitude columns.
- **Coordinates**: latitude 6.5°N through 38.5°N; longitude 66.5°E through 100.0°E, both on 0.25° increments.
- **Encoding**: little-endian float32; `-999.0` is the missing-value sentinel.
- **Validation**: file size 25,425,900 bytes; decoded 1,811,860 valid cells and 4,544,615 missing cells across the year; valid values range from 0.0 to 469.21014 mm/day.
- **Acquisition**: official IMD Pune year-selection form, selecting 2025. No authentication was required for this archive download.

The backend accepts annual `.grd` files and annual NetCDF `.nc`/`.nc4` files placed in `data/raw/rainfall/`, or in the directory configured by `IMD_RAINFALL_DIR`. See `data/raw/rainfall/README.md`.

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

### Verified Local Observation
- **Installed file**: `data/raw/max_temp/maxtemp_2025.grd` (local and git-ignored; not committed).
- **Coverage**: 2025-01-01 through 2025-12-31 (365 daily grids).
- **Grid shape**: 31 latitude rows × 31 longitude columns.
- **Coordinates**: latitude 7.5°N through 37.5°N; longitude 67.5°E through 97.5°E, both on 1° increments.
- **Encoding**: little-endian float32; `99.9` is the missing-value sentinel (stored as float32 and compared with tolerance).
- **Validation**: file size 1,403,060 bytes; on 2025-07-15, the 355 valid grid cells range from 23.97195°C to 37.49660°C.
- **Acquisition**: official IMD Pune year-selection form, selecting 2025.

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

### Verified Local Observation
- **Installed file**: `data/raw/min_temp/mintemp_2025.grd` (local and git-ignored; not committed).
- **Coverage**: 2025-01-01 through 2025-12-31 (365 daily grids).
- **Grid shape**: 31 latitude rows × 31 longitude columns.
- **Coordinates**: latitude 7.5°N through 37.5°N; longitude 67.5°E through 97.5°E, both on 1° increments.
- **Encoding**: little-endian float32; `99.9` is the missing-value sentinel (stored as float32 and compared with tolerance).
- **Validation**: file size 1,403,060 bytes; on 2025-07-15, the 355 valid grid cells range from 15.52389°C to 27.41776°C.
- **Acquisition**: official IMD Pune year-selection form, selecting 2025.

### Intended Project Use
- Used for calculating Diurnal Temperature Range ($DTR = T_{\text{max}} - T_{\text{min}}$), tracking nighttime thermal trends, and building multi-variable climate state vectors for data assimilation.

---

## Summary Matrix

| Dataset Name | Parameter | Resolution | Data Format | Official URL | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IMD Rainfall** | Rainfall ($\text{mm/day}$) | $0.25^\circ \times 0.25^\circ$ | `.grd` / `.nc` | [Rainfall_25_Bin.html](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html) | ✅ 2025 file downloaded locally and validated |
| **IMD Max Temp** | $T_{\text{max}}$ ($^\circ\text{C}$) | $1.0^\circ \times 1.0^\circ$ | `.grd` | [Max_1_Bin.html](https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html) | ✅ 2025 file downloaded locally and validated |
| **IMD Min Temp** | $T_{\text{min}}$ ($^\circ\text{C}$) | $1.0^\circ \times 1.0^\circ$ | `.grd` | [Min_1_Bin.html](https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html) | ✅ 2025 file downloaded locally and validated |
