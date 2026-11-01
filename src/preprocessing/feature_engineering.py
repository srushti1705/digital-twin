"""
Feature Engineering Module
---------------------------
Provides reusable, leak-free climate feature engineering utilities for IMD climate data:
1. Primary Thermodynamic & Precipitation Indices:
   - Diurnal Temperature Range (DTR = Tmax - Tmin)
   - Daily Mean Temperature (Tmean = (Tmax + Tmin) / 2)
   - Precipitation Intensity Flags (Rainy Day >= 2.5mm, Heavy Rain >= 35.5mm, Dry Day < 2.5mm)
   - Extreme Heatwave Flag (Tmax >= 40.0 °C)
   - Cumulative Annual Precipitation
2. Previous-Day Lag Features (Non-leaking backward shifts along time axis):
   - Previous-day rainfall (rain_lag1, rain_lag2, etc.)
   - Previous-day maximum temperature (tmax_lag1, tmax_lag2, etc.)
   - Previous-day minimum temperature (tmin_lag1, tmin_lag2, etc.)
3. Backward Rolling Statistics (Trailing 3-day, 7-day, 30-day windows):
   - Rolling rainfall averages & cumulative sums (rain_roll_7d_mean, rain_roll_30d_sum)
   - Rolling temperature averages (tmax_roll_7d_mean, tmin_roll_7d_mean, tmean_roll_7d_mean)
4. Temporal Index & Seasonal Features:
   - Year, Month, Day, Day of Year (1-366), IMD Season (Winter, Pre-Monsoon, Monsoon, Post-Monsoon)
   - Cyclical Sine/Cosine Encodings for Day of Year and Month

Strict Leakage Prevention Guarantee:
- Lag features use positive backward shifts (ds.shift(time=+k) for t-k), never referencing future observations t+k.
- Rolling features use non-centered trailing windows (ds.rolling(time=w, center=False)), ensuring t relies only on window [t-w+1, t].

All operations return new xarray Datasets without mutating input objects.
"""

from typing import Dict, Optional, Any, List, Union, Tuple
import logging
import numpy as np
import pandas as pd
import xarray as xr

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# IMD Season mapping
IMD_SEASON_NAMES = {
    1: "Winter",
    2: "Pre-Monsoon",
    3: "Monsoon",
    4: "Post-Monsoon"
}


def create_climate_indices(
    ds: xr.Dataset,
    tmax_name: str = "tmax",
    tmin_name: str = "tmin",
    rain_name: str = "rain"
) -> xr.Dataset:
    """
    Derives fundamental thermodynamic and precipitation indices:
    - Diurnal Temperature Range (DTR = Tmax - Tmin)
    - Daily Mean Temperature (Tmean = (Tmax + Tmin) / 2)
    - Precipitation Flags (Rainy Day >= 2.5mm, Heavy Rain >= 35.5mm, Dry Day < 2.5mm)
    - Heatwave Flag (Tmax >= 40.0 °C)
    - Cumulative Annual Precipitation

    Parameters:
        ds (xr.Dataset): Input climate dataset.
        tmax_name (str): Variable name for Maximum Temperature.
        tmin_name (str): Variable name for Minimum Temperature.
        rain_name (str): Variable name for Rainfall.

    Returns:
        xr.Dataset: Dataset augmented with derived climate indices.
    """
    ds_feat = ds.copy(deep=True)

    has_tmax = tmax_name in ds_feat.data_vars
    has_tmin = tmin_name in ds_feat.data_vars
    has_rain = rain_name in ds_feat.data_vars

    # 1. Temperature Indices (DTR & Tmean)
    if has_tmax and has_tmin:
        tmax_arr = ds_feat[tmax_name]
        tmin_arr = ds_feat[tmin_name]

        # Diurnal Temperature Range: DTR = Tmax - Tmin
        dtr = tmax_arr - tmin_arr
        dtr.attrs = {
            "long_name": "Diurnal Temperature Range",
            "units": "degC",
            "description": "Daily Maximum Temperature minus Daily Minimum Temperature"
        }
        ds_feat["dtr"] = dtr

        # Daily Mean Temperature: Tmean = (Tmax + Tmin) / 2
        tmean = (tmax_arr + tmin_arr) / 2.0
        tmean.attrs = {
            "long_name": "Daily Mean Temperature",
            "units": "degC",
            "description": "Average of Daily Maximum and Minimum Temperature"
        }
        ds_feat["tmean"] = tmean

        # Extreme Heatwave Flag (Tmax >= 40.0 °C)
        heatwave = xr.where(tmax_arr >= 40.0, 1.0, 0.0).where(~np.isnan(tmax_arr), np.nan)
        heatwave.attrs = {
            "long_name": "Extreme Heatwave Indicator Flag",
            "units": "binary (1=Heatwave, 0=Normal)",
            "description": "Flag indicating daily maximum temperature >= 40.0 degC"
        }
        ds_feat["heatwave_flag"] = heatwave

    # 2. Rainfall Indices
    if has_rain:
        rain_arr = ds_feat[rain_name]

        # Rainy Day Flag (Rain >= 2.5 mm/day as per IMD classification)
        rainy_day = xr.where(rain_arr >= 2.5, 1.0, 0.0).where(~np.isnan(rain_arr), np.nan)
        rainy_day.attrs = {
            "long_name": "IMD Rainy Day Flag",
            "units": "binary (1=Rainy Day >= 2.5mm, 0=Non-Rainy)",
            "description": "Flag indicating daily rainfall >= 2.5 mm/day"
        }
        ds_feat["rainy_day_flag"] = rainy_day

        # Heavy Rainfall Flag (Rain >= 35.5 mm/day)
        heavy_rain = xr.where(rain_arr >= 35.5, 1.0, 0.0).where(~np.isnan(rain_arr), np.nan)
        heavy_rain.attrs = {
            "long_name": "IMD Heavy Rainfall Flag",
            "units": "binary (1=Heavy Rain >= 35.5mm, 0=Normal/Light)",
            "description": "Flag indicating daily rainfall >= 35.5 mm/day"
        }
        ds_feat["heavy_rain_flag"] = heavy_rain

        # Dry Day Flag (Rain < 2.5 mm/day)
        dry_day = xr.where(rain_arr < 2.5, 1.0, 0.0).where(~np.isnan(rain_arr), np.nan)
        dry_day.attrs = {
            "long_name": "IMD Dry Day Flag",
            "units": "binary (1=Dry Day < 2.5mm, 0=Rainy)",
            "description": "Flag indicating daily rainfall < 2.5 mm/day"
        }
        ds_feat["dry_day_flag"] = dry_day

        # Cumulative Rainfall across time axis
        if "time" in ds_feat.coords:
            cum_rain = rain_arr.cumsum(dim="time", skipna=True)
            cum_rain.attrs = {
                "long_name": "Cumulative Annual Precipitation",
                "units": "mm",
                "description": "Cumulative sum of daily rainfall starting from Jan 1"
            }
            ds_feat["cumulative_rain"] = cum_rain

    return ds_feat


def create_lag_features(
    ds: xr.Dataset,
    var_names: Optional[List[str]] = None,
    lags: List[int] = [1, 2, 3]
) -> xr.Dataset:
    """
    Creates previous-day lag features for specified climate variables.

    Strict Data Leakage Guarantee:
    - Lag features use positive backward shifts along time dimension (ds[var].shift(time=lag) for t-lag),
      ensuring observation at time t relies strictly on past timesteps (t-1, t-2, ...).
      Future observations (t+1, t+2, ...) are NEVER referenced.

    Parameters:
        ds (xr.Dataset): Input climate dataset with 'time' dimension.
        var_names (list, optional): List of variable names to compute lags for. Defaults to all data variables.
        lags (list): List of integer lag steps (e.g. [1, 2, 3] for t-1, t-2, t-3). Default: [1, 2, 3].

    Returns:
        xr.Dataset: Dataset augmented with lag feature variables (e.g. rain_lag1, tmax_lag1, tmin_lag1).
    """
    if "time" not in ds.dims and "time" not in ds.coords:
        raise ValueError("❌ Dataset must contain a 'time' dimension/coordinate to compute lag features.")

    ds_feat = ds.copy(deep=True)
    target_vars = var_names if var_names is not None else list(ds.data_vars.keys())

    for v_name in target_vars:
        if v_name not in ds.data_vars:
            continue

        for lag in lags:
            if lag <= 0:
                raise ValueError(f"❌ Lag step must be a positive integer > 0 to prevent data leakage. Got lag={lag}.")

            lag_var_name = f"{v_name}_lag{lag}"
            lag_arr = ds[v_name].shift(time=lag)
            lag_arr.attrs = {
                "long_name": f"{v_name.upper()} Previous Day Lag {lag} (t-{lag})",
                "units": ds[v_name].attrs.get("units", ""),
                "description": f"Observation of {v_name} shifted backward by {lag} day(s). Strict zero future leakage."
            }
            ds_feat[lag_var_name] = lag_arr

    return ds_feat


def create_rolling_features(
    ds: xr.Dataset,
    var_names: Optional[List[str]] = None,
    windows: List[int] = [3, 7, 30]
) -> xr.Dataset:
    """
    Creates backward-looking trailing rolling statistics (rolling mean, rolling sum) for climate variables.

    Strict Data Leakage Guarantee:
    - Uses non-centered trailing windows (ds[var].rolling(time=window, center=False, min_periods=1)),
      ensuring calculation at time t depends strictly on window [t-window+1, ..., t] without using future data.

    Parameters:
        ds (xr.Dataset): Input climate dataset with 'time' dimension.
        var_names (list, optional): Variables to calculate rolling statistics for.
        windows (list): Window sizes in days (e.g. [3, 7, 30]). Default: [3, 7, 30].

    Returns:
        xr.Dataset: Dataset augmented with trailing rolling features.
    """
    if "time" not in ds.dims and "time" not in ds.coords:
        raise ValueError("❌ Dataset must contain a 'time' dimension/coordinate to compute rolling features.")

    ds_feat = ds.copy(deep=True)
    target_vars = var_names if var_names is not None else [v for v in ds.data_vars if not v.endswith("_flag")]

    for v_name in target_vars:
        if v_name not in ds.data_vars:
            continue

        for w in windows:
            # Trailing rolling mean over window [t-w+1, t]
            roll_mean_name = f"{v_name}_roll_{w}d_mean"
            roll_mean = ds[v_name].rolling(time=w, center=False, min_periods=1).mean(skipna=True)
            roll_mean.attrs = {
                "long_name": f"{v_name.upper()} Trailing {w}-Day Rolling Mean",
                "units": ds[v_name].attrs.get("units", ""),
                "description": f"Trailing {w}-day moving average up to time t (no future data leakage)."
            }
            ds_feat[roll_mean_name] = roll_mean

            # For precipitation, also compute trailing rolling sum
            if "rain" in v_name.lower() or "rf" in v_name.lower() or "precip" in v_name.lower():
                roll_sum_name = f"{v_name}_roll_{w}d_sum"
                roll_sum = ds[v_name].rolling(time=w, center=False, min_periods=1).sum(skipna=True)
                roll_sum.attrs = {
                    "long_name": f"Rainfall Trailing {w}-Day Cumulative Sum",
                    "units": "mm",
                    "description": f"Trailing {w}-day cumulative rainfall up to time t (no future data leakage)."
                }
                ds_feat[roll_sum_name] = roll_sum

    return ds_feat


def create_temporal_index_features(
    ds: xr.Dataset,
    include_cyclical: bool = True
) -> xr.Dataset:
    """
    Extracts temporal index features (Year, Month, Day, Day of Year, IMD Season, and Cyclical Encodings).

    Parameters:
        ds (xr.Dataset): Input dataset with datetime 'time' coordinate.
        include_cyclical (bool): Whether to include sin/cos cyclical encodings for Day of Year and Month. Default: True.

    Returns:
        xr.Dataset: Dataset augmented with temporal coordinate features.
    """
    if "time" not in ds.coords:
        raise ValueError("❌ Dataset must contain a 'time' coordinate for temporal index features.")

    ds_feat = ds.copy(deep=True)
    dt_times = pd.to_datetime(ds_feat.coords["time"].values)

    years = dt_times.year.values
    months = dt_times.month.values
    days = dt_times.day.values
    doys = dt_times.dayofyear.values

    # IMD Seasons: 1=Winter (Jan-Feb), 2=Pre-Monsoon (Mar-May), 3=Monsoon (Jun-Sep), 4=Post-Monsoon (Oct-Dec)
    seasons = np.select(
        [
            np.isin(months, [1, 2]),
            np.isin(months, [3, 4, 5]),
            np.isin(months, [6, 7, 8, 9]),
            np.isin(months, [10, 11, 12])
        ],
        [1, 2, 3, 4],
        default=1
    )

    ds_feat["year"] = ("time", years, {"long_name": "Calendar Year"})
    ds_feat["month"] = ("time", months, {"long_name": "Month of Year (1-12)"})
    ds_feat["day"] = ("time", days, {"long_name": "Day of Month (1-31)"})
    ds_feat["day_of_year"] = ("time", doys, {"long_name": "Day of Year (1-366)"})
    ds_feat["season"] = ("time", seasons, {"long_name": "IMD Season Code (1=Winter, 2=Pre-Monsoon, 3=Monsoon, 4=Post-Monsoon)"})

    if include_cyclical:
        # Cyclical sin/cos encodings for Month (period=12)
        ds_feat["month_sin"] = ("time", np.sin(2 * np.pi * months / 12.0), {"long_name": "Month Sine Encoding"})
        ds_feat["month_cos"] = ("time", np.cos(2 * np.pi * months / 12.0), {"long_name": "Month Cosine Encoding"})

        # Cyclical sin/cos encodings for Day of Year (period=365.25)
        ds_feat["doy_sin"] = ("time", np.sin(2 * np.pi * doys / 365.25), {"long_name": "Day of Year Sine Encoding"})
        ds_feat["doy_cos"] = ("time", np.cos(2 * np.pi * doys / 365.25), {"long_name": "Day of Year Cosine Encoding"})

    return ds_feat


def create_monthly_climatological_stats(
    ds: xr.Dataset,
    var_names: Optional[List[str]] = None
) -> xr.Dataset:
    """
    Computes monthly climatological statistics (monthly means for temperature, monthly sums/means for rainfall).

    Parameters:
        ds (xr.Dataset): Input dataset with datetime 'time' coordinate.
        var_names (list, optional): Target variables.

    Returns:
        xr.Dataset: Monthly aggregated dataset.
    """
    if "time" not in ds.coords:
        raise ValueError("❌ Dataset must contain a 'time' coordinate for monthly climatological statistics.")

    target_vars = var_names if var_names is not None else [v for v in ds.data_vars if not v.endswith("_flag")]
    monthly_ds = xr.Dataset()

    for v_name in target_vars:
        if v_name not in ds.data_vars:
            continue

        if "rain" in v_name.lower() or "rf" in v_name.lower() or "precip" in v_name.lower():
            monthly_ds[f"{v_name}_monthly_sum"] = ds[v_name].resample(time="1MS").sum(skipna=True)
            monthly_ds[f"{v_name}_monthly_mean"] = ds[v_name].resample(time="1MS").mean(skipna=True)
        else:
            monthly_ds[f"{v_name}_monthly_mean"] = ds[v_name].resample(time="1MS").mean(skipna=True)
            monthly_ds[f"{v_name}_monthly_max"] = ds[v_name].resample(time="1MS").max(skipna=True)
            monthly_ds[f"{v_name}_monthly_min"] = ds[v_name].resample(time="1MS").min(skipna=True)

    return monthly_ds


def engineer_all_climate_features(
    ds: xr.Dataset,
    tmax_name: str = "tmax",
    tmin_name: str = "tmin",
    rain_name: str = "rain",
    include_lags: bool = True,
    lags: List[int] = [1],
    include_rolling: bool = True,
    windows: List[int] = [3, 7],
    include_temporal: bool = True
) -> xr.Dataset:
    """
    Master feature engineering pipeline combining:
    1. Primary Thermodynamic & Precipitation Indices (DTR, Tmean, precipitation intensity, heatwave flags)
    2. Previous-Day Lag Features (rain_lag1, tmax_lag1, tmin_lag1) with zero future data leakage
    3. Trailing Rolling Statistics (rain_roll_7d_mean, tmax_roll_7d_mean) with zero future data leakage
    4. Temporal & Seasonal Index Encodings (Year, Month, Day, Day of Year, IMD Season, cyclical sin/cos)

    Parameters:
        ds (xr.Dataset): Input climate dataset.
        tmax_name (str): Maximum temperature variable name.
        tmin_name (str): Minimum temperature variable name.
        rain_name (str): Rainfall variable name.
        include_lags (bool): Whether to generate previous-day lag features (default: True).
        lags (list): Lag steps to generate (default: [1]).
        include_rolling (bool): Whether to generate trailing rolling statistics (default: True).
        windows (list): Rolling windows in days (default: [3, 7]).
        include_temporal (bool): Whether to generate temporal/seasonal features (default: True).

    Returns:
        xr.Dataset: Comprehensive augmented climate feature dataset.
    """
    logger.info("Executing Master Climate Feature Engineering Pipeline...")
    ds_feat = ds.copy(deep=True)

    # 1. Compute Primary Climate Indices (DTR, Tmean, Precip/Heatwave Flags)
    ds_feat = create_climate_indices(ds_feat, tmax_name=tmax_name, tmin_name=tmin_name, rain_name=rain_name)

    # Base variables for lags and rolling calculations
    base_vars = [v for v in [rain_name, tmax_name, tmin_name, "dtr", "tmean"] if v in ds_feat.data_vars]

    # 2. Previous-Day Lag Features (Non-leaking)
    if include_lags and "time" in ds_feat.coords:
        logger.info(f"Computing previous-day lag features for lags {lags} (zero future leakage)...")
        ds_feat = create_lag_features(ds_feat, var_names=base_vars, lags=lags)

    # 3. Trailing Rolling Features (Non-leaking)
    if include_rolling and "time" in ds_feat.coords:
        logger.info(f"Computing trailing rolling statistics for windows {windows}d (zero future leakage)...")
        ds_feat = create_rolling_features(ds_feat, var_names=base_vars, windows=windows)

    # 4. Temporal Index & Seasonal Features
    if include_temporal and "time" in ds_feat.coords:
        logger.info("Extracting temporal index features (Year, Month, Day, DOY, IMD Season, Sin/Cos encodings)...")
        ds_feat = create_temporal_index_features(ds_feat, include_cyclical=True)

    logger.info(f"Feature Engineering Complete: {len(ds_feat.data_vars)} data variables generated.")
    return ds_feat


def derive_climate_features(
    ds: xr.Dataset,
    tmax_name: str = "tmax",
    tmin_name: str = "tmin",
    rain_name: str = "rain"
) -> xr.Dataset:
    """Wrapper function preserving backward compatibility."""
    return engineer_all_climate_features(
        ds,
        tmax_name=tmax_name,
        tmin_name=tmin_name,
        rain_name=rain_name,
        include_lags=False,
        include_rolling=False,
        include_temporal=False
    )


if __name__ == "__main__":
    print("=== Feature Engineering Module Initialized ===")
    print("Available Feature Engineering Functions:")
    print("  - create_climate_indices()")
    print("  - create_lag_features()")
    print("  - create_rolling_features()")
    print("  - create_temporal_index_features()")
    print("  - create_monthly_climatological_stats()")
    print("  - engineer_all_climate_features()")
