"""
Time-Aware Chronological Data Splitting Module for Climate Prediction.

Strictly enforces chronological ordering (train -> validation -> test) to prevent
temporal data leakage and ensure realistic operational forecasting evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import xarray as xr


# Educational documentation on chronological splitting in climate science
CHRONOLOGICAL_SPLITTING_RATIONALE = """
WHY CHRONOLOGICAL SPLITTING IS CRITICAL FOR CLIMATE PREDICTION:
------------------------------------------------------------------
1. Temporal Autocorrelation & Memory: Climate variables (temperature, soil moisture,
   sea surface temperatures) exhibit strong multi-day persistence. Randomly shuffling
   timesteps allows a model to interpolate between adjacent days (e.g., t-1 and t+1),
   leading to severe look-ahead bias and artificially inflated performance.

2. Zero Data Leakage: In operational climate forecasting, future observations are
   never available at prediction time. Chronological splitting guarantees that the
   training set contains strictly historical observations (earliest period), the
   validation set contains a later contiguous period, and the test set contains the
   latest evaluation period.

3. Non-Stationarity & Seasonal Trends: Climate data features complex intra-annual seasonality
   and inter-annual trends. Chronological splits evaluate the model's ability to generalize
   forward in time across changing seasonal transitions (e.g., post-monsoon to winter).
"""


@dataclass
class TimeSplitMetadata:
    """Metadata detailing a specific time split (train, validation, or test)."""
    name: str
    start_date: str
    end_date: str
    num_days: int
    num_land_samples: int
    percentage: float


class ChronologicalSplitter:
    """
    Time-aware dataset splitter for climate time-series datasets.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        warm_up_days: int = 7
    ):
        """
        Parameters
        ----------
        train_ratio : float, default=0.70
            Fraction of time steps allocated to training.
        val_ratio : float, default=0.15
            Fraction of time steps allocated to validation.
        test_ratio : float, default=0.15
            Fraction of time steps allocated to testing.
        warm_up_days : int, default=7
            Number of initial time steps to exclude due to backward-looking rolling features.
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got: {train_ratio:.2f} + {val_ratio:.2f} + {test_ratio:.2f} = {train_ratio + val_ratio + test_ratio:.2f}"
            )
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.warm_up_days = warm_up_days

    def split_xarray_dataset(
        self,
        ds: xr.Dataset,
        time_dim: str = "time",
        verbose: bool = True
    ) -> Dict[str, xr.Dataset]:
        """
        Chronologically splits an xarray Dataset into train, validation, and test subsets.

        Parameters
        ----------
        ds : xr.Dataset
            Input xarray dataset containing a time dimension.
        time_dim : str, default='time'
            Name of the time dimension.
        verbose : bool, default=True
            Whether to print summary statistics.

        Returns
        -------
        Dict[str, xr.Dataset]
            Dictionary with keys 'train', 'val', 'test'.
        """
        # Trim initial warm-up period if specified
        if self.warm_up_days > 0 and len(ds[time_dim]) > self.warm_up_days:
            ds = ds.isel({time_dim: slice(self.warm_up_days, None)})

        total_days = len(ds[time_dim])
        n_train = int(np.floor(total_days * self.train_ratio))
        n_val = int(np.floor(total_days * self.val_ratio))
        
        # Train slice: 0 to n_train
        ds_train = ds.isel({time_dim: slice(0, n_train)})
        # Val slice: n_train to n_train + n_val
        ds_val = ds.isel({time_dim: slice(n_train, n_train + n_val)})
        # Test slice: n_train + n_val to end
        ds_test = ds.isel({time_dim: slice(n_train + n_val, None)})

        splits = {
            "train": ds_train,
            "val": ds_val,
            "test": ds_test
        }

        if verbose:
            self.print_split_summary(splits, time_dim=time_dim)

        return splits

    def print_split_summary(
        self,
        splits: Dict[str, Union[xr.Dataset, pd.DataFrame]],
        time_dim: str = "time"
    ) -> None:
        """
        Prints clean tabular summary of date ranges and sample counts for each split.
        """
        print("=" * 70)
        print("TIME-AWARE CHRONOLOGICAL DATA SPLIT SUMMARY")
        print("=" * 70)
        print(CHRONOLOGICAL_SPLITTING_RATIONALE.strip())
        print("-" * 70)

        total_samples = 0
        summary_rows = []

        for name, data in splits.items():
            if isinstance(data, xr.Dataset):
                times = data[time_dim].values
                start_date = str(np.datetime_as_string(times[0], unit="D"))
                end_date = str(np.datetime_as_string(times[-1], unit="D"))
                num_days = len(times)
                
                # Count valid land samples if 'tmax' is present
                if "tmax" in data.data_vars:
                    # Count non-NaN land cells per day
                    valid_land_cells = int(np.sum(~np.isnan(data["tmax"].isel({time_dim: 0}).values)))
                    num_samples = num_days * valid_land_cells
                else:
                    num_samples = num_days
            elif isinstance(data, pd.DataFrame):
                times = pd.to_datetime(data[time_dim])
                start_date = times.min().strftime("%Y-%m-%d")
                end_date = times.max().strftime("%Y-%m-%d")
                num_days = len(times.unique())
                num_samples = len(data)
            else:
                raise TypeError(f"Unsupported data type for split summary: {type(data)}")

            total_samples += num_samples
            summary_rows.append({
                "Split": name.capitalize(),
                "Start Date": start_date,
                "End Date": end_date,
                "Days": num_days,
                "Land Grid Samples": f"{num_samples:,}"
            })

        summary_df = pd.DataFrame(summary_rows)
        # Calculate percentage
        pcts = [f"{(int(r['Land Grid Samples'].replace(',', '')) / total_samples * 100):.1f}%" for r in summary_rows]
        summary_df["Share"] = pcts

        print(summary_df.to_string(index=False))
        print("-" * 70)
        print(f"Total Usable Timesteps: {sum(r['Days'] for r in summary_rows)} days")
        print(f"Total Usable Spatial-Temporal Samples: {total_samples:,}")
        print("=" * 70)


def split_processed_dataset(
    nc_path: str = "data/processed/imd/india_climate_2020_aligned.nc",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    warm_up_days: int = 7
) -> Dict[str, xr.Dataset]:
    """
    Convenience function to load and chronologically split the processed IMD NetCDF dataset.
    """
    ds = xr.open_dataset(nc_path)
    splitter = ChronologicalSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        warm_up_days=warm_up_days
    )
    return splitter.split_xarray_dataset(ds)


if __name__ == "__main__":
    split_processed_dataset()
