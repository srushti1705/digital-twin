# Local IMD Maximum Temperature Files

Place official annual IMD maximum-temperature grids in this directory. Raw data files
are intentionally ignored by Git; only this README is tracked.

The backend reads annual little-endian float32 `.grd` files such as
`maxtemp_2025.grd`. Each daily record is a 31 × 31 grid at 1° resolution with grid
centres from 7.5°N to 37.5°N and 67.5°E to 97.5°E. The raw missing-value sentinel is
`99.9`.

The verified 2025 file covers 2025-01-01 through 2025-12-31. To use another storage
location, set `IMD_TEMPERATURE_DIR` to the parent directory containing `max_temp/`
and `min_temp/`.

Official archive:
https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html
