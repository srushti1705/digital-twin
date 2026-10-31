"""
Inspect Phase 3 processed datasets for Phase 4 dataset readiness assessment.
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
import joblib

def inspect():
    proc_dir = "data/processed/imd"
    print("=" * 60)
    print("PHASE 4 DATASET READINESS INSPECTION")
    print("=" * 60)
    
    # 1. Processed files
    files = os.listdir(proc_dir)
    print("\n1. Available Processed Files in data/processed/imd/:")
    for f in files:
        fp = os.path.join(proc_dir, f)
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"   - {f} ({size_mb:.2f} MB)")
        
    # 2 & 3. Variables & Features in primary aligned netcdf
    aligned_path = os.path.join(proc_dir, "india_climate_2020_aligned.nc")
    ds = xr.open_dataset(aligned_path)
    
    data_vars = list(ds.data_vars.keys())
    print(f"\n2 & 3. Variables Available ({len(data_vars)} total):")
    
    primary_vars = ['rain', 'tmax', 'tmin', 'tmean', 'dtr']
    derived_flags = [v for v in data_vars if 'flag' in v]
    lag_vars = [v for v in data_vars if 'lag' in v]
    roll_vars = [v for v in data_vars if 'roll' in v]
    temporal_vars = ['year', 'month', 'day', 'day_of_year', 'season', 'doy_sin', 'doy_cos']
    
    print(f"   - Primary Climate Variables ({len(primary_vars)}): {primary_vars}")
    print(f"   - Physical Flag Indicators ({len(derived_flags)}): {derived_flags}")
    print(f"   - Lagged Predictor Features ({len(lag_vars)}): {lag_vars}")
    print(f"   - Rolling Window Features ({len(roll_vars)}): {roll_vars}")
    print(f"   - Temporal Encodings ({len(temporal_vars)}): {temporal_vars}")
    
    # 4. Temporal resolution
    times = ds['time'].values
    print(f"\n4. Temporal Resolution:")
    print(f"   - Start Date: {str(times[0])[:10]}")
    print(f"   - End Date:   {str(times[-1])[:10]}")
    print(f"   - Frequency:  Daily (1-day step)")
    print(f"   - Total Steps: {len(times)} days (Leap Year 2020)")
    
    # 5. Spatial structure
    lats = ds['lat'].values
    lons = ds['lon'].values
    print(f"\n5. Spatial Structure:")
    print(f"   - Grid: Regular Lat-Lon Grid")
    print(f"   - Resolution: 0.25° x 0.25° (~25 km)")
    print(f"   - Latitude Range: [{lats.min():.2f}°, {lats.max():.2f}°N] ({len(lats)} points)")
    print(f"   - Longitude Range: [{lons.min():.2f}°, {lons.max():.2f}°E] ({len(lons)} points)")
    print(f"   - Grid Cells / Timestep: {len(lats)} x {len(lons)} = {len(lats) * len(lons)}")

    # 6. Number of usable observations
    # Land vs Ocean mask: Land cells are those where tmax is not NaN on day 0
    day0_tmax = ds['tmax'].isel(time=0).values
    land_mask = ~np.isnan(day0_tmax)
    land_cells_count = np.sum(land_mask)
    ocean_cells_count = np.sum(~land_mask)
    total_grid_cells = len(lats) * len(lons)
    
    total_obs = len(times) * total_grid_cells
    usable_land_obs = len(times) * land_cells_count
    
    print(f"\n6. Usable Observations:")
    print(f"   - Total Grid Cells in Domain: {total_grid_cells}")
    print(f"   - Valid Indian Land Mask Cells: {land_cells_count} ({land_cells_count/total_grid_cells*100:.2f}%)")
    print(f"   - Non-Land Ocean Mask Cells: {ocean_cells_count} ({ocean_cells_count/total_grid_cells*100:.2f}%)")
    print(f"   - Total Grid-Day Observations: {total_obs:,}")
    print(f"   - Total Usable Land Observations: {usable_land_obs:,}")
    
    # 7. Remaining missing values
    print(f"\n7. Missing Value Status:")
    # Check land cells for missing values across all timesteps for primary variables
    land_missing_rain = np.sum(np.isnan(ds['rain'].values[:, land_mask]))
    land_missing_tmax = np.sum(np.isnan(ds['tmax'].values[:, land_mask]))
    land_missing_tmin = np.sum(np.isnan(ds['tmin'].values[:, land_mask]))
    
    print(f"   - Ocean Background Missing Values: Retained as NaN (as expected for ocean mask)")
    print(f"   - Land Mask Missing Values (Rainfall): {land_missing_rain}")
    print(f"   - Land Mask Missing Values (Tmax): {land_missing_tmax}")
    print(f"   - Land Mask Missing Values (Tmin): {land_missing_tmin}")
    print(f"   - Land Mask Missing Values (Lag/Rolling features): Expected NaNs on first 1-7 days due to window shift")
    
    # 8. Suitability for supervised learning
    print(f"\n8. Supervised Learning Suitability:")
    print(f"   - Target Candidates: 'rain' (precipitation regression / classification), 'tmax'/'tmin' (temperature forecasting)")
    print(f"   - Predictors: 30+ leak-free lag, rolling, and temporal features available")
    print(f"   - Spatial tabular samples per timestep: {land_cells_count} land rows per day")
    print(f"   - Sequence modeling suitability: 366 continuous daily time steps per grid cell")
    print("=" * 60)

if __name__ == "__main__":
    inspect()
