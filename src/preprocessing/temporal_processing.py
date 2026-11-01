"""
Temporal Processing Module
---------------------------
Handles temporal coordinate identification, datetime standardization, chronological sorting,
duplicate timestamp detection, missing time period detection, and extraction of temporal features
(Year, Month, Day, Day of Year, IMD Season).

IMD Seasons in India:
- Winter: Jan - Feb (Months 1, 2) -> Code 1
- Pre-Monsoon (Summer): Mar - May (Months 3, 4, 5) -> Code 2
- SW Monsoon: Jun - Sep (Months 6, 7, 8, 9) -> Code 3
- Post-Monsoon (NE Monsoon): Oct - Dec (Months 10, 11, 12) -> Code 4

All operations return new xarray Datasets without mutating input objects.
Does NOT perform prediction lag feature creation.
"""

from typing import Dict, Optional, Any, Tuple, List, Union
import logging
import numpy as np
import pandas as pd
import xarray as xr

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard IMD Seasons in India
IMD_SEASON_NAMES = {
    1: "Winter",
    2: "Pre-Monsoon",
    3: "Monsoon",
    4: "Post-Monsoon"
}


def identify_time_dimension(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[str]:
    """
    Identifies the time dimension name in an xarray Dataset or DataArray.

    Parameters:
        ds (xr.Dataset | xr.DataArray): Input xarray object.

    Returns:
        str or None: The identified time dimension name (e.g. 'time', 'date', 'day'), or None if not found.
    """
    candidates = ("time", "date", "day", "datetime", "t")

    # Check dataset dimensions
    for dim in ds.dims:
        if str(dim).lower() in candidates:
            return str(dim)

    # Check dataset coordinates
    for coord in ds.coords:
        if str(coord).lower() in candidates:
            return str(coord)

    return None


def standardize_datetime_coordinates(
    ds: xr.Dataset,
    start_date: Optional[str] = "2020-01-01",
    freq: str = "D"
) -> xr.Dataset:
    """
    Converts dataset time coordinates into a consistent pandas.DatetimeIndex representation.

    Parameters:
        ds (xr.Dataset): Input climate dataset.
        start_date (str, optional): Default start date if integer/step time index exists. Default: '2020-01-01'.
        freq (str): Target frequency ('D' for daily).

    Returns:
        xr.Dataset: Dataset with standardized datetime64[ns] 'time' coordinate.
    """
    ds_clean = ds.copy(deep=True)
    time_key = identify_time_dimension(ds_clean)

    if time_key is None:
        raise ValueError("❌ Could not identify time dimension in dataset.")

    time_values = ds_clean.coords[time_key].values

    # Convert to pandas DatetimeIndex if needed
    try:
        dt_index = pd.to_datetime(time_values)
    except Exception:
        # If integer step indices, generate date range from start_date
        num_steps = len(time_values)
        dt_index = pd.date_range(start=start_date, periods=num_steps, freq=freq)

    ds_clean = ds_clean.assign_coords({time_key: dt_index})

    if time_key != "time":
        ds_clean = ds_clean.rename({time_key: "time"})

    return ds_clean


def sort_chronologically(ds: xr.Dataset) -> xr.Dataset:
    """
    Sorts dataset observations strictly chronologically along the time dimension.

    Parameters:
        ds (xr.Dataset): Input xarray Dataset with 'time' dimension.

    Returns:
        xr.Dataset: Chronologically sorted Dataset.
    """
    time_key = identify_time_dimension(ds)
    if time_key is None:
        raise ValueError("❌ No time dimension found for chronological sorting.")

    return ds.sortby(time_key, ascending=True)


def detect_duplicate_timestamps(ds: xr.Dataset) -> Dict[str, Any]:
    """
    Detects duplicate timestamps along the dataset time dimension.

    Parameters:
        ds (xr.Dataset): Input dataset.

    Returns:
        dict: Summary of duplicate timestamps (has_duplicates, duplicate_count, duplicate_dates).
    """
    time_key = identify_time_dimension(ds)
    if time_key is None:
        return {"has_time_dim": False, "has_duplicates": False, "duplicate_count": 0, "duplicate_dates": []}

    times = pd.to_datetime(ds.coords[time_key].values)
    duplicated_mask = times.duplicated(keep=False)
    dup_count = int(times.duplicated().sum())

    dup_dates = []
    if dup_count > 0:
        unique_dups = times[duplicated_mask].unique()
        dup_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in unique_dups[:10]]

    return {
        "has_time_dim": True,
        "time_dimension": time_key,
        "total_timesteps": len(times),
        "has_duplicates": dup_count > 0,
        "duplicate_count": dup_count,
        "duplicate_dates_sample": dup_dates
    }


def detect_missing_time_periods(
    ds: xr.Dataset,
    expected_freq: str = "D"
) -> Dict[str, Any]:
    """
    Detects missing dates or gaps in the expected daily time series.

    Parameters:
        ds (xr.Dataset): Input dataset with datetime 'time' coordinate.
        expected_freq (str): Expected frequency ('D' for daily).

    Returns:
        dict: Detailed missing time period analysis (missing_days_count, date_gaps, complete_sequence_flag).
    """
    time_key = identify_time_dimension(ds)
    if time_key is None:
        return {"has_time_dim": False, "message": "No time dimension found."}

    times = pd.to_datetime(ds.coords[time_key].values)
    start_date = times.min()
    end_date = times.max()

    expected_range = pd.date_range(start=start_date, end=end_date, freq=expected_freq)
    missing_dates = expected_range.difference(times)

    missing_count = len(missing_dates)
    missing_sample = [d.strftime("%Y-%m-%d") for d in missing_dates[:10]]

    return {
        "has_time_dim": True,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "expected_timesteps": len(expected_range),
        "actual_timesteps": len(times),
        "missing_days_count": missing_count,
        "is_complete_sequence": missing_count == 0,
        "sample_missing_dates": missing_sample
    }


def extract_temporal_features(
    ds: xr.Dataset,
    include_cyclical: bool = False
) -> xr.Dataset:
    """
    Extracts useful temporal features from the datetime 'time' coordinate:
    - year (int)
    - month (1-12)
    - day (1-31)
    - day_of_year (1-366)
    - season (1=Winter, 2=Pre-Monsoon, 3=Monsoon, 4=Post-Monsoon)
    - season_name (string label)

    Does NOT create prediction lag features.

    Parameters:
        ds (xr.Dataset): Input xarray Dataset.
        include_cyclical (bool): Whether to include sin/cos cyclical encodings for month and day_of_year.

    Returns:
        xr.Dataset: Dataset augmented with temporal feature coordinate variables.
    """
    ds_feat = ds.copy(deep=True)
    time_key = identify_time_dimension(ds_feat)

    if time_key is None:
        raise ValueError("❌ Time dimension not found for temporal feature extraction.")

    dt_times = pd.to_datetime(ds_feat.coords[time_key].values)

    years = dt_times.year.values
    months = dt_times.month.values
    days = dt_times.day.values
    doys = dt_times.dayofyear.values

    # Map months to IMD Seasons
    # 1=Winter (Jan-Feb), 2=Pre-Monsoon (Mar-May), 3=Monsoon (Jun-Sep), 4=Post-Monsoon (Oct-Dec)
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

    season_names = [IMD_SEASON_NAMES[s] for s in seasons]

    # Assign new temporal coordinates/variables to dataset
    ds_feat["year"] = (time_key, years, {"long_name": "Calendar Year"})
    ds_feat["month"] = (time_key, months, {"long_name": "Month of Year (1-12)"})
    ds_feat["day"] = (time_key, days, {"long_name": "Day of Month (1-31)"})
    ds_feat["day_of_year"] = (time_key, doys, {"long_name": "Day of Year (1-366)"})
    ds_feat["season"] = (time_key, seasons, {"long_name": "IMD Season Code (1=Winter, 2=Pre-Monsoon, 3=Monsoon, 4=Post-Monsoon)"})

    if include_cyclical:
        # Cyclical sin/cos encodings for Month (period 12)
        ds_feat["month_sin"] = (time_key, np.sin(2 * np.pi * months / 12.0), {"long_name": "Month Sine Encoding"})
        ds_feat["month_cos"] = (time_key, np.cos(2 * np.pi * months / 12.0), {"long_name": "Month Cosine Encoding"})

        # Cyclical sin/cos encodings for Day of Year (period 365.25)
        ds_feat["doy_sin"] = (time_key, np.sin(2 * np.pi * doys / 365.25), {"long_name": "Day of Year Sine Encoding"})
        ds_feat["doy_cos"] = (time_key, np.cos(2 * np.pi * doys / 365.25), {"long_name": "Day of Year Cosine Encoding"})

    return ds_feat


def process_temporal_pipeline(
    ds: xr.Dataset,
    start_date: Optional[str] = "2020-01-01",
    freq: str = "D",
    extract_features: bool = True
) -> Tuple[xr.Dataset, Dict[str, Any]]:
    """
    Executes the full temporal processing pipeline:
    1. Identifies time dimension.
    2. Converts dates to consistent pd.DatetimeIndex representation.
    3. Sorts observations chronologically.
    4. Detects duplicate timestamps.
    5. Detects missing time periods/gaps.
    6. Extracts temporal features (Year, Month, Day, Day of Year, IMD Season).

    Parameters:
        ds (xr.Dataset): Input climate dataset.
        start_date (str): Starting date string.
        freq (str): Time frequency.
        extract_features (bool): Whether to derive temporal feature variables.

    Returns:
        tuple: (processed_ds, temporal_report_dict)
    """
    logger.info("Executing Temporal Processing Pipeline...")

    # 1. Standardize Datetime Coordinates
    ds_proc = standardize_datetime_coordinates(ds, start_date=start_date, freq=freq)

    # 2. Sort Chronologically
    ds_proc = sort_chronologically(ds_proc)

    # 3. Detect Duplicate Timestamps
    dup_report = detect_duplicate_timestamps(ds_proc)

    # 4. Detect Missing Time Periods
    gap_report = detect_missing_time_periods(ds_proc, expected_freq=freq)

    # 5. Extract Temporal Features
    if extract_features:
        ds_proc = extract_temporal_features(ds_proc)

    report = {
        "time_dimension": identify_time_dimension(ds_proc),
        "total_timesteps": int(ds_proc.sizes["time"]),
        "start_date": gap_report.get("start_date"),
        "end_date": gap_report.get("end_date"),
        "duplicates_detected": dup_report["has_duplicates"],
        "duplicate_count": dup_report["duplicate_count"],
        "missing_gaps_detected": not gap_report.get("is_complete_sequence", True),
        "missing_days_count": gap_report.get("missing_days_count", 0),
        "features_extracted": ["year", "month", "day", "day_of_year", "season"] if extract_features else []
    }

    logger.info(f"Temporal Pipeline Complete: {report['total_timesteps']} timesteps from {report['start_date']} to {report['end_date']}.")
    return ds_proc, report


def align_temporal_dimension(
    ds: xr.Dataset,
    start_date: str = "2020-01-01",
    freq: str = "D"
) -> xr.Dataset:
    """Wrapper function preserving backward compatibility."""
    ds_proc, _ = process_temporal_pipeline(ds, start_date=start_date, freq=freq, extract_features=False)
    return ds_proc


def compute_monthly_aggregations(ds: xr.Dataset) -> xr.Dataset:
    """Computes monthly aggregations (means for temperature, sums and means for rainfall)."""
    if "time" not in ds.coords:
        raise ValueError("Dataset must have a 'time' coordinate for temporal aggregation.")

    monthly_ds = xr.Dataset()

    for var_name in ds.data_vars:
        if "rain" in var_name.lower() or "rf" in var_name.lower() or "precip" in var_name.lower():
            monthly_ds[f"{var_name}_monthly_sum"] = ds[var_name].resample(time="1MS").sum(skipna=True)
            monthly_ds[f"{var_name}_monthly_mean"] = ds[var_name].resample(time="1MS").mean(skipna=True)
        else:
            monthly_ds[f"{var_name}_monthly_mean"] = ds[var_name].resample(time="1MS").mean(skipna=True)
            monthly_ds[f"{var_name}_monthly_max"] = ds[var_name].resample(time="1MS").max(skipna=True)
            monthly_ds[f"{var_name}_monthly_min"] = ds[var_name].resample(time="1MS").min(skipna=True)

    return monthly_ds


def compute_rolling_statistics(
    ds: xr.Dataset,
    window: int = 7,
    min_periods: int = 1
) -> xr.Dataset:
    """Computes rolling-window statistics (7-day rolling mean, rolling max) along time axis."""
    if "time" not in ds.coords:
        raise ValueError("Dataset must have a 'time' coordinate for rolling statistics.")

    rolling_ds = xr.Dataset()

    for var_name in ds.data_vars:
        rolling_ds[f"{var_name}_roll_{window}d_mean"] = (
            ds[var_name].rolling(time=window, min_periods=min_periods).mean(skipna=True)
        )
        if "rain" in var_name.lower() or "rf" in var_name.lower():
            rolling_ds[f"{var_name}_roll_{window}d_sum"] = (
                ds[var_name].rolling(time=window, min_periods=min_periods).sum(skipna=True)
            )

    return rolling_ds


if __name__ == "__main__":
    print("=== Temporal Processing Module Initialized ===")
    print("Available Functions:")
    print("  - identify_time_dimension()")
    print("  - standardize_datetime_coordinates()")
    print("  - sort_chronologically()")
    print("  - detect_duplicate_timestamps()")
    print("  - detect_missing_time_periods()")
    print("  - extract_temporal_features()")
    print("  - process_temporal_pipeline()")
