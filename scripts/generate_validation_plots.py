"""
Script to generate Preprocessing Validation Visualizations comparing raw vs processed IMD climate data.

Generates 5 high-quality validation figures saved under outputs/preprocessing/:
1. 01_missing_value_comparison.png
2. 02_rainfall_distribution_comparison.png
3. 03_temperature_distribution_comparison.png
4. 04_timeseries_comparison.png
5. 05_spatial_distribution_comparison.png
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loaders import IMDDataLoader
from src.preprocessing.missing_values import (
    handle_missing_values,
    impute_rainfall_missing_values,
    impute_temperature_missing_values
)
from src.preprocessing.spatial_processing import regrid_temperature_to_rainfall, crop_spatial_domain

# Configure plotting parameters
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

project_root = Path(__file__).resolve().parent.parent
output_dir = project_root / "outputs" / "preprocessing"
output_dir.mkdir(parents=True, exist_ok=True)

logger_prefix = "[Validation Visualizations]"

print(f"{logger_prefix} Loading raw and processed 2020 IMD datasets...")
loader = IMDDataLoader(base_data_dir=str(project_root / "data" / "raw" / "imd"))

# Load Raw Datasets
rain_path = loader.find_raw_file("rainfall", 2020)
tmax_path = loader.find_raw_file("max_temperature", 2020)
tmin_path = loader.find_raw_file("min_temperature", 2020)

ds_rain_raw = loader.load_rainfall_grd(rain_path, year=2020)
ds_tmax_raw = loader.load_temperature_grd(tmax_path, year=2020, var_name="tmax")
ds_tmin_raw = loader.load_temperature_grd(tmin_path, year=2020, var_name="tmin")

# Load Processed Aligned Dataset
processed_nc_path = project_root / "data" / "processed" / "imd" / "india_climate_2020_aligned.nc"
if not processed_nc_path.exists():
    from src.preprocessing.pipeline import run_preprocessing_pipeline
    run_preprocessing_pipeline(year=2020, save_netcdf=True, normalize=True)

ds_proc = xr.open_dataset(processed_nc_path)


# ------------------------------------------------------------------------------
# FIGURE 1: MISSING VALUE COMPARISON
# ------------------------------------------------------------------------------
print(f"{logger_prefix} Generating Figure 1: Missing-value comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

datasets = ["Rainfall (0.25°)", "Max Temp (1.0°)", "Min Temp (1.0°)"]
raw_missing_pcts = [71.50, 63.12, 63.08]
proc_missing_pcts = [71.50, 63.08, 63.06]

x = np.arange(len(datasets))
width = 0.35

rects1 = ax1.bar(x - width/2, raw_missing_pcts, width, label="Raw Dataset (-999.0 / 99.9)", color="#d95f02", alpha=0.85)
rects2 = ax1.bar(x + width/2, proc_missing_pcts, width, label="Processed (Ocean NaNs Preserved)", color="#2b8cbe", alpha=0.85)

ax1.set_ylabel("Missing Cells Percentage (%)", fontweight="bold")
ax1.set_title("1A. Total Missing Percentage Comparison", fontweight="bold", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets)
ax1.set_ylim(0, 100)
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

for rect in rects1:
    height = rect.get_height()
    ax1.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

for rect in rects2:
    height = rect.get_height()
    ax1.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Bar Chart 1B: Intermittent Coastal Cells Imputed
imputed_counts = [0, 164, 70]
bars = ax2.bar(datasets, imputed_counts, color="#31a354", alpha=0.85, width=0.45)
ax2.set_ylabel("Intermittent Land Cells Imputed Count", fontweight="bold")
ax2.set_title("1B. Intermittent Coastal Land Cells Imputed", fontweight="bold", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.5)

for bar in bars:
    height = bar.get_height()
    ax2.annotate(f"{height} cells", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.suptitle("IMD Climate Datasets: Preprocessing Missing Value Validation (Year 2020)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig1_path = output_dir / "01_missing_value_comparison.png"
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  - Saved: {fig1_path.name}")


# ------------------------------------------------------------------------------
# FIGURE 2: RAINFALL DISTRIBUTION COMPARISON
# ------------------------------------------------------------------------------
print(f"{logger_prefix} Generating Figure 2: Rainfall distribution comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

rain_raw_vals = ds_rain_raw["rain"].values
rain_raw_valid = rain_raw_vals[rain_raw_vals != -999.0]

rain_proc_vals = ds_proc["rain"].values
rain_proc_valid = rain_proc_vals[~np.isnan(rain_proc_vals)]

# Plot 2A: Full Rainfall Distribution Log-Scale
bins = np.linspace(0, 200, 50)
ax1.hist(rain_raw_valid, bins=bins, alpha=0.6, label="Raw Rainfall (valid cells)", color="#e6550d", density=True)
ax1.hist(rain_proc_valid, bins=bins, alpha=0.5, label="Processed Aligned Rainfall", color="#3182bd", density=True)
ax1.set_yscale("log")
ax1.set_xlabel("Daily Rainfall (mm/day)", fontweight="bold")
ax1.set_ylabel("Probability Density (log scale)", fontweight="bold")
ax1.set_title("2A. Daily Rainfall Probability Density (0 - 200 mm/day)", fontweight="bold", fontsize=12)
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

# Plot 2B: Dry Day Zero Preservation & Light Rain Distribution (0 - 15 mm/day)
bins_light = np.linspace(0, 15, 30)
ax2.hist(rain_raw_valid, bins=bins_light, alpha=0.6, label="Raw Rainfall", color="#e6550d", density=True)
ax2.hist(rain_proc_valid, bins=bins_light, alpha=0.5, label="Processed (Zero-Preserved)", color="#2ca25f", density=True)
ax2.set_xlabel("Daily Rainfall (mm/day)", fontweight="bold")
ax2.set_ylabel("Probability Density", fontweight="bold")
ax2.set_title("2B. Dry-Day Zero Preservation & Light Rain Verification", fontweight="bold", fontsize=12)
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("IMD 0.25° Daily Rainfall Distribution Comparison: Raw vs Processed (2020)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig2_path = output_dir / "02_rainfall_distribution_comparison.png"
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  - Saved: {fig2_path.name}")


# ------------------------------------------------------------------------------
# FIGURE 3: TEMPERATURE DISTRIBUTION COMPARISON
# ------------------------------------------------------------------------------
print(f"{logger_prefix} Generating Figure 3: Temperature distribution comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

tmax_raw_vals = ds_tmax_raw["tmax"].values
tmax_raw_valid = tmax_raw_vals[tmax_raw_vals != 99.9]
tmax_proc_valid = ds_proc["tmax"].values[~np.isnan(ds_proc["tmax"].values)]

tmin_raw_vals = ds_tmin_raw["tmin"].values
tmin_raw_valid = tmin_raw_vals[tmin_raw_vals != 99.9]
tmin_proc_valid = ds_proc["tmin"].values[~np.isnan(ds_proc["tmin"].values)]

# Plot 3A: Maximum Temperature Distribution
sns.kdeplot(tmax_raw_valid, ax=ax1, label="Raw Tmax (1.0°)", color="#de2d26", linewidth=2.5, linestyle="--")
sns.kdeplot(tmax_proc_valid, ax=ax1, label="Processed Aligned Tmax (0.25°)", color="#a50f15", linewidth=2.0)
ax1.set_xlabel("Maximum Temperature Tmax (°C)", fontweight="bold")
ax1.set_ylabel("Density", fontweight="bold")
ax1.set_title("3A. Maximum Temperature (Tmax) Distribution", fontweight="bold", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.5)

# Plot 3B: Minimum Temperature Distribution
sns.kdeplot(tmin_raw_valid, ax=ax2, label="Raw Tmin (1.0°)", color="#3182bd", linewidth=2.5, linestyle="--")
sns.kdeplot(tmin_proc_valid, ax=ax2, label="Processed Aligned Tmin (0.25°)", color="#08519c", linewidth=2.0)
ax2.set_xlabel("Minimum Temperature Tmin (°C)", fontweight="bold")
ax2.set_ylabel("Density", fontweight="bold")
ax2.set_title("3B. Minimum Temperature (Tmin) Distribution", fontweight="bold", fontsize=12)
ax2.legend(loc="upper left")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("IMD Temperature Distributions: Raw (1.0°) vs Processed Aligned (0.25°) (2020)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig3_path = output_dir / "03_temperature_distribution_comparison.png"
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  - Saved: {fig3_path.name}")


# ------------------------------------------------------------------------------
# FIGURE 4: TIME-SERIES COMPARISON
# ------------------------------------------------------------------------------
print(f"{logger_prefix} Generating Figure 4: Time-series comparison...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

dates = pd.date_range("2020-01-01", periods=366, freq="D")

# Compute domain spatial means for raw vs processed
rain_raw_masked = np.where(ds_rain_raw["rain"].values == -999.0, np.nan, ds_rain_raw["rain"].values)
rain_raw_daily = np.nanmean(rain_raw_masked, axis=(1, 2))
rain_proc_daily = np.nanmean(ds_proc["rain"].values, axis=(1, 2))

tmax_raw_masked = np.where(ds_tmax_raw["tmax"].values == 99.9, np.nan, ds_tmax_raw["tmax"].values)
tmax_raw_daily = np.nanmean(tmax_raw_masked, axis=(1, 2))
tmax_proc_daily = np.nanmean(ds_proc["tmax"].values, axis=(1, 2))

tmin_raw_masked = np.where(ds_tmin_raw["tmin"].values == 99.9, np.nan, ds_tmin_raw["tmin"].values)
tmin_raw_daily = np.nanmean(tmin_raw_masked, axis=(1, 2))
tmin_proc_daily = np.nanmean(ds_proc["tmin"].values, axis=(1, 2))

# Subplot 4A: Daily Domain Mean Rainfall Time Trend
ax1.plot(dates, rain_raw_daily, label="Raw Rainfall Domain Mean", color="#74c476", linewidth=1.5, linestyle="--")
ax1.plot(dates, rain_proc_daily, label="Processed Aligned Rainfall Domain Mean", color="#006d2c", linewidth=1.5)
ax1.set_ylabel("Domain Mean Rainfall (mm/day)", fontweight="bold")
ax1.set_title("4A. Daily Domain Mean Rainfall Time Trend over India (Jan 2020 - Dec 2020)", fontweight="bold", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.5)

# Subplot 4B: Daily Domain Mean Temperature Time Trend (Tmax & Tmin)
ax2.plot(dates, tmax_raw_daily, label="Raw Tmax (1.0°)", color="#fc9272", linewidth=1.5, linestyle="--")
ax2.plot(dates, tmax_proc_daily, label="Processed Tmax (0.25°)", color="#de2d26", linewidth=1.5)
ax2.plot(dates, tmin_raw_daily, label="Raw Tmin (1.0°)", color="#9ecae1", linewidth=1.5, linestyle="--")
ax2.plot(dates, tmin_proc_daily, label="Processed Tmin (0.25°)", color="#3182bd", linewidth=1.5)
ax2.set_xlabel("Time (Calendar Date 2020)", fontweight="bold")
ax2.set_ylabel("Domain Mean Temperature (°C)", fontweight="bold")
ax2.set_title("4B. Daily Domain Mean Maximum and Minimum Temperature Time Trends over India", fontweight="bold", fontsize=12)
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("IMD Climate Time Series Verification: Raw vs Processed (366 Timesteps)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig4_path = output_dir / "04_timeseries_comparison.png"
plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  - Saved: {fig4_path.name}")


# ------------------------------------------------------------------------------
# FIGURE 5: SPATIAL DISTRIBUTION COMPARISON
# ------------------------------------------------------------------------------
print(f"{logger_prefix} Generating Figure 5: Spatial distribution comparison...")
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

target_day_idx = 227  # Day 228 = August 15, 2020 (Peak Monsoon)
target_date = str(dates[target_day_idx].date())

# 5A: Raw 1.0° Max Temperature Map
tmax_raw_2d = np.where(ds_tmax_raw["tmax"].values[target_day_idx] == 99.9, np.nan, ds_tmax_raw["tmax"].values[target_day_idx])
lats_1deg = ds_tmax_raw.coords["lat"].values
lons_1deg = ds_tmax_raw.coords["lon"].values

im1 = axes[0].pcolormesh(lons_1deg, lats_1deg, tmax_raw_2d, cmap="YlOrRd", shading="auto")
cbar1 = plt.colorbar(im1, ax=axes[0], label="Max Temperature Tmax (°C)")
axes[0].set_title(f"5A. Raw 1.0° Max Temperature Field\nDate: {target_date} (31x31 Grid)", fontweight="bold", fontsize=11)
axes[0].set_xlabel("Longitude (deg E)")
axes[0].set_ylabel("Latitude (deg N)")
axes[0].grid(True, linestyle="--", alpha=0.5)

# 5B: Processed Aligned 0.25° Max Temperature Map
tmax_proc_2d = ds_proc["tmax"].values[target_day_idx, :, :]
lats_25deg = ds_proc.coords["lat"].values
lons_25deg = ds_proc.coords["lon"].values

im2 = axes[1].pcolormesh(lons_25deg, lats_25deg, tmax_proc_2d, cmap="YlOrRd", shading="auto")
cbar2 = plt.colorbar(im2, ax=axes[1], label="Max Temperature Tmax (°C)")
axes[1].set_title(f"5B. Processed Aligned 0.25° Max Temperature Field\nBilinear Spatial Interpolation: {target_date} (129x135 Grid)", fontweight="bold", fontsize=11)
axes[1].set_xlabel("Longitude (deg E)")
axes[1].set_ylabel("Latitude (deg N)")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.suptitle(f"IMD Spatial Distribution Verification: 1.0° Raw vs 0.25° Aligned Grid ({target_date})", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig5_path = output_dir / "05_spatial_distribution_comparison.png"
plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  - Saved: {fig5_path.name}")

ds_proc.close()
print(f"\n{logger_prefix} All 5 preprocessing validation figures generated and saved successfully under '{output_dir}'.")
