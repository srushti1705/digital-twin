# Local IMD Rainfall Files

Place official IMD annual rainfall files in this directory. The downloaded data files are
intentionally ignored by Git; only this README is tracked.

The backend currently reads:

- Annual IMD binary files such as `ind2025_rfp25.grd` (little-endian float32, 129 × 135 grid).
- Annual NetCDF files with time, latitude, longitude coordinates and a rainfall variable.

The 2025 IMD archive file is installed locally as `ind2025_rfp25.grd`. It covers
2025-01-01 through 2025-12-31. To use another storage location, set `IMD_RAINFALL_DIR`
in the backend environment to the directory containing the files.

Official archive and product details:
https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html
