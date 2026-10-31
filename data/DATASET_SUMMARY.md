# Phase 2 Dataset Summary Report

**Project Title**: AI-Powered Digital Twin of India's Climate using National Datasets  
**Document Purpose**: Official reference summary comparing empirical metrics, grid geometries, and statistical parameters extracted from the downloaded IMD climate datasets.

---

## 1. Master Comparative Dataset Table

| Parameter / Feature | IMD High-Resolution Rainfall | IMD Maximum Temperature ($T_{\text{max}}$) | IMD Minimum Temperature ($T_{\text{min}}$) |
| :--- | :--- | :--- | :--- |
| **Dataset Title** | IMD Daily Gridded Rainfall (0.25°) | IMD Daily Max Temperature (1.0°) | IMD Daily Min Temperature (1.0°) |
| **Climate Parameter** | Precipitation / Daily Rainfall ($\text{mm/day}$) | Maximum Surface Temperature ($^\circ\text{C}$) | Minimum Surface Temperature ($^\circ\text{C}$) |
| **Source Agency** | IMD Pune / CMPG | IMD Pune / CMPG | IMD Pune / CMPG |
| **Official Page URL** | [Rainfall_25_Bin.html](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html) | [Max_1_Bin.html](https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html) | [Min_1_Bin.html](https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html) |
| **Spatial Resolution** | **$0.25^\circ \times 0.25^\circ$** lat/lon | **$1.00^\circ \times 1.00^\circ$** lat/lon | **$1.00^\circ \times 1.00^\circ$** lat/lon |
| **Spatial Coordinates** | $6.50^\circ\text{N}$–$38.50^\circ\text{N}$, $66.50^\circ\text{E}$–$100.00^\circ\text{E}$ | $7.50^\circ\text{N}$–$37.50^\circ\text{N}$, $67.50^\circ\text{E}$–$97.50^\circ\text{E}$ | $7.50^\circ\text{N}$–$37.50^\circ\text{N}$, $67.50^\circ\text{E}$–$97.50^\circ\text{E}$ |
| **Temporal Coverage** | 2020-01-01 to 2020-12-31 | 2020-01-01 to 2020-12-31 | 2020-01-01 to 2020-12-31 |
| **Timesteps Count** | 366 daily timesteps (Leap Year) | 366 daily timesteps (Leap Year) | 366 daily timesteps (Leap Year) |
| **Grid Dimensions** | **$366 \times 129 \times 135$** | **$366 \times 31 \times 31$** | **$366 \times 31 \times 31$** |
| **Dimension Names** | `time`, `lat`, `lon` | `time`, `lat`, `lon` | `time`, `lat`, `lon` |
| **Primary Variable** | `rain` | `tmax` | `tmin` |
| **File Format** | Unformatted Binary (`.grd`, `float32`) | Unformatted Binary (`.GRD`, `float32`) | Unformatted Binary (`.GRD`, `float32`) |
| **Downloaded Filename**| `Rainfall_ind2020_rfp25.grd` | `Maxtemp_MaxT_2020.GRD` | `Mintemp_MinT_2020.GRD` |
| **File Size on Disk** | $25,495,560\text{ bytes}$ (~24.31 MB) | $1,406,904\text{ bytes}$ (~1.34 MB) | $1,406,904\text{ bytes}$ (~1.34 MB) |
| **Total Elements** | $6,373,890$ cells | $351,726$ cells | $351,726$ cells |
| **Missing Fill Value** | **`-999.0`** | **`99.9`** | **`99.9`** |
| **Missing Count (%)** | $4,557,066$ cells (**$71.50\%$**) | $222,026$ cells (**$63.12\%$**) | $221,868$ cells (**$63.08\%$**) |
| **Minimum Value** | **$0.0000\text{ mm/day}$** | **$2.8524^\circ\text{C}$** | **$-6.8540^\circ\text{C}$** |
| **Maximum Value** | **$539.9128\text{ mm/day}$** | **$46.7761^\circ\text{C}$** | **$31.8596^\circ\text{C}$** |
| **Mean Value** | **$3.4877\text{ mm/day}$** | **$30.4615^\circ\text{C}$** | **$18.9501^\circ\text{C}$** |

---

## 2. Key Observations & Ingest Requirements for Phase 2

1. **Spatial Scale Disparity**:
   - Rainfall features 16x higher spatial resolution density per unit area ($0.25^\circ \times 0.25^\circ$) compared to Temperature ($1.0^\circ \times 1.0^\circ$).
   - *Requirement*: High-resolution spatial interpolation (bilinear re-gridding) is required in the preprocessing pipeline to align temperature fields onto the $0.25^\circ$ rainfall coordinate system.

2. **Fill Value Standardization**:
   - Rainfall encodes non-land ocean cells as `-999.0`, whereas Temperature datasets encode non-land ocean cells as `99.9`.
   - *Requirement*: The data preprocessing pipeline must unify all fill values into standard Python `numpy.nan` objects across variables.

3. **Data Integrity**:
   - All empirical values listed above were extracted directly from the raw IMD binary files without data alteration.
