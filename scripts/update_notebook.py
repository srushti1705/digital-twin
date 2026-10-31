"""
Script to append Phase 3 cells to notebooks/01_data_exploration.ipynb and execute all cells.
"""

import json
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

project_root = Path(__file__).resolve().parent.parent
nb_path = project_root / "notebooks" / "01_data_exploration.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Ensure no duplicate Phase 3 header if script is re-run
existing_p3 = any("Phase 3 — Data Preprocessing and Feature Engineering" in "".join(c.get("source", [])) for c in nb["cells"])

if not existing_p3:
    phase3_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 3 — Data Preprocessing and Feature Engineering\n",
                "\n",
                "In Phase 3, we execute the complete data preprocessing and feature engineering pipeline on the raw 2020 IMD datasets (`Rainfall_ind2020_rfp25.grd`, `Maxtemp_MaxT_2020.GRD`, `Mintemp_MinT_2020.GRD`).\n",
                "\n",
                "All preprocessing functions are imported from the `src.preprocessing` package.\n",
                "\n",
                "We demonstrate:\n",
                "1. **Raw dataset quality checks** (`check_dataset_quality`)\n",
                "2. **Missing-value analysis** (`analyze_missing_values`, `generate_missing_report`)\n",
                "3. **Missing-value handling** (`impute_rainfall_missing_values`, `impute_temperature_missing_values`)\n",
                "4. **Temporal processing** (`process_temporal_pipeline`)\n",
                "5. **Spatial processing** (`validate_coordinate_ranges`, `get_spatial_resolution`)\n",
                "6. **Spatial alignment** (`regrid_temperature_to_rainfall`)\n",
                "7. **Feature engineering** (`engineer_all_climate_features`)\n",
                "8. **Before/after statistics** (comparison table)\n",
                "9. **Processed dataset inspection** (structure, summary, 2D heatmap plot)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 3 — Module Imports\n",
                "We import all modular preprocessing functions from `src.preprocessing` submodules."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.preprocessing.quality_checks import check_dataset_quality, validate_physical_bounds\n",
                "from src.preprocessing.missing_values import (\n",
                "    analyze_missing_values,\n",
                "    generate_missing_report,\n",
                "    impute_rainfall_missing_values,\n",
                "    impute_temperature_missing_values,\n",
                "    process_climate_dataset_missing_values\n",
                ")\n",
                "from src.preprocessing.temporal_processing import (\n",
                "    process_temporal_pipeline,\n",
                "    identify_time_dimension,\n",
                "    detect_duplicate_timestamps,\n",
                "    detect_missing_time_periods\n",
                ")\n",
                "from src.preprocessing.spatial_processing import (\n",
                "    identify_spatial_coordinates,\n",
                "    validate_coordinate_ranges,\n",
                "    get_spatial_resolution,\n",
                "    detect_duplicate_spatial_coordinates,\n",
                "    document_resolution_mismatch,\n",
                "    generate_spatial_report,\n",
                "    regrid_temperature_to_rainfall,\n",
                "    crop_spatial_domain\n",
                ")\n",
                "from src.preprocessing.feature_engineering import engineer_all_climate_features\n",
                "from src.preprocessing.normalization import analyze_scaler_suitability, fit_transform_dataset_scaler, save_scaler\n",
                "from src.preprocessing.pipeline import run_preprocessing_pipeline\n",
                "\n",
                "print('All Phase 3 preprocessing submodules loaded successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 3 — Step 1: Raw Dataset Quality Checks\n",
                "We run the 10-point automated quality inspection suite (`check_dataset_quality`) on all 3 raw IMD datasets to evaluate empty dimensions, duplicate coordinates, physical domain bounds, data types, NaNs, raw fill values, and physical consistency ($T_{\\text{min}} \\le T_{\\text{max}}$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "qc_rain = check_dataset_quality(ds_rain, var_name='rain', raw_fill_value=-999.0)\n",
                "qc_max = check_dataset_quality(ds_max, var_name='tmax', raw_fill_value=99.9)\n",
                "qc_min = check_dataset_quality(ds_min, var_name='tmin', raw_fill_value=99.9)\n",
                "\n",
                "print('=== 1. RAINFALL QUALITY CHECKS ===')\n",
                "print('QC Valid:', qc_rain['is_valid'])\n",
                "print('Checks Summary:', qc_rain['checks'])\n",
                "\n",
                "print('\\n=== 2. MAX TEMP QUALITY CHECKS ===')\n",
                "print('QC Valid:', qc_max['is_valid'])\n",
                "\n",
                "print('\\n=== 3. MIN TEMP QUALITY CHECKS ===')\n",
                "print('QC Valid:', qc_min['is_valid'])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 3 — Step 2: Missing-Value Analysis\n",
                "We evaluate total missing cell counts, overall missing percentages, temporal missing patterns across timesteps, and 2D spatial missing patterns (ocean background cells vs intermittent land/coastal cells) using `analyze_missing_values` and `generate_missing_report` without mutating the data."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "report_rain_str = generate_missing_report(ds_rain)\n",
                "report_max_str = generate_missing_report(ds_max)\n",
                "report_min_str = generate_missing_report(ds_min)\n",
                "\n",
                "print('=== 1. RAINFALL MISSING REPORT ===')\n",
                "print(report_rain_str)\n",
                "\n",
                "print('=== 2. MAX TEMP MISSING REPORT ===')\n",
                "print(report_max_str)\n",
                "\n",
                "print('=== 3. MIN TEMP MISSING REPORT ===')\n",
                "print(report_min_str)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 3 — Step 3: Approved Missing-Value Handling\n",
                "We apply parameter-specific missing value strategies:\n",
                "- **Rainfall**: Two-stage occurrence-gated spatial/temporal handling with zero-preservation ($<0.1\\text{ mm/day} \\to 0.0$).\n",
                "- **Temperature ($T_{\\text{max}}, T_{\\text{min}}$)**: Temporal linear/PCHIP interpolation (`max_gap=3`) followed by 2D spatial land interpolation.\n",
                "- **Ocean Mask**: Preserves ocean background cells strictly as `NaN` (no ocean filling).\n",
                "- **Valid Observations**: Preserves valid land observations 100%."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ds_rain_imp, rain_imp_metrics = impute_rainfall_missing_values(ds_rain, var_name='rain', raw_fill_value=-999.0)\n",
                "ds_max_imp, max_imp_metrics = impute_temperature_missing_values(ds_max, var_name='tmax', raw_fill_value=99.9)\n",
                "ds_min_imp, min_imp_metrics = impute_temperature_missing_values(ds_min, var_name='tmin', raw_fill_value=99.9)\n",
                "\n",
                "print('Rainfall Imputation Summary:', rain_imp_metrics)\n",
                "print('\\nMax Temp Imputation Summary:', max_imp_metrics)\n",
                "print('\\nMin Temp Imputation Summary:', min_imp_metrics)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 3 — Step 4: Temporal Processing\n",
                "We execute the temporal pipeline (`process_temporal_pipeline`) to standardize dates into `pandas.DatetimeIndex` (`datetime64[ns]`), sort observations chronologically, check duplicate timestamps (0 duplicates found), and verify time series continuity (0 missing days found)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ds_rain_time, rain_time_report = process_temporal_pipeline(ds_rain_imp, start_date='2020-01-01', extract_features=False)\n",
                "ds_max_time, max_time_report = process_temporal_pipeline(ds_max_imp, start_date='2020-01-01', extract_features=False)\n",
                "ds_min_time, min_time_report = process_temporal_pipeline(ds_min_imp, start_date='2020-01-01', extract_features=False)\n",
                "\n",
                "print('Rainfall Temporal Report:', rain_time_report)\n",
                "print('Max Temp Temporal Report:', max_time_report)\n",
                "print('Min Temp Temporal Report:', min_time_report)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 3 — Step 5: Spatial Processing & Resolution Diagnostics\n",
                "We validate coordinate ranges, check grid regularity, inspect spatial resolution differences ($0.25^\\circ \\times 0.25^\\circ$ rainfall vs $1.0^\\circ \\times 1.0^\\circ$ temperature), and verify duplicate spatial coordinates (0 duplicates found)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ds_map = {'rainfall': ds_rain_time, 'tmax': ds_max_time, 'tmin': ds_min_time}\n",
                "spatial_report_str = generate_spatial_report(ds_map)\n",
                "print(spatial_report_str)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 3 — Step 6: Spatial Alignment (Bilinear Re-Gridding)\n",
                "We re-grid the 1.0° Temperature datasets (`tmax`, `tmin`) onto the high-resolution 0.25° Rainfall grid using 2D bilinear spatial interpolation (`regrid_temperature_to_rainfall`), crop to the India bounding box ($6.5^\\circ\\text{N} - 38.5^\\circ\\text{N}$, $66.5^\\circ\\text{E} - 100.0^\\circ\\text{E}$), and combine all variables into a single unified `xarray.Dataset`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ds_max_regrid = regrid_temperature_to_rainfall(ds_max_time, ds_rain_time, method='linear')\n",
                "ds_min_regrid = regrid_temperature_to_rainfall(ds_min_time, ds_rain_time, method='linear')\n",
                "\n",
                "ds_combined = xr.Dataset(\n",
                "    data_vars={\n",
                "        'rain': ds_rain_time['rain'],\n",
                "        'tmax': ds_max_regrid['tmax'],\n",
                "        'tmin': ds_min_regrid['tmin']\n",
                "    },\n",
                "    coords=ds_rain_time.coords,\n",
                "    attrs={'title': 'IMD Aligned Climate Dataset (2020)', 'spatial_resolution': '0.25 deg x 0.25 deg'}\n",
                ")\n",
                "\n",
                "ds_cropped = crop_spatial_domain(ds_combined, lat_bounds=(6.5, 38.5), lon_bounds=(66.5, 100.0))\n",
                "\n",
                "print('Combined Aligned Dataset Dimensions:', dict(ds_cropped.sizes))\n",
                "print('Data Variables:', list(ds_cropped.data_vars.keys()))\n",
                "display(ds_cropped)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 3 — Step 7: Leak-Free Feature Engineering\n",
                "We run `engineer_all_climate_features` to generate 36 unified, leak-free climate feature variables:\n",
                "- **Thermodynamic Indices**: Diurnal Temp Range ($DTR = T_{\\text{max}} - T_{\\text{min}}$), Mean Temp ($T_{\\text{mean}} = \\frac{T_{\\text{max}} + T_{\\text{min}}}{2}$), Heatwave Flag ($T_{\\text{max}} \\ge 40^\\circ\\text{C}$).\n",
                "- **Precipitation Indices**: Rainy Day Flag ($\\ge 2.5\\text{ mm}$), Heavy Rain Flag ($\\ge 35.5\\text{ mm}$), Dry Day Flag ($< 2.5\\text{ mm}$), Cumulative Rain.\n",
                "- **Previous-Day Lags**: `rain_lag1`, `tmax_lag1`, `tmin_lag1`, `dtr_lag1`, `tmean_lag1` (strict positive backward shift $t-1$).\n",
                "- **Trailing Rolling Stats**: Trailing 3-day and 7-day rolling means & sums over window $[t-w+1, t]$.\n",
                "- **Temporal Index & Season**: `year`, `month`, `day`, `day_of_year`, `season` (IMD Season code), sine/cosine encodings (`doy_sin`, `doy_cos`, `month_sin`, `month_cos`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ds_final = engineer_all_climate_features(\n",
                "    ds_cropped,\n",
                "    tmax_name='tmax',\n",
                "    tmin_name='tmin',\n",
                "    rain_name='rain',\n",
                "    include_lags=True,\n",
                "    lags=[1],\n",
                "    include_rolling=True,\n",
                "    windows=[3, 7],\n",
                "    include_temporal=True\n",
                ")\n",
                "\n",
                "print(f'Generated {len(ds_final.data_vars)} Feature Variables:')\n",
                "print(list(ds_final.data_vars.keys()))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 3 — Step 8: Before vs. After Statistics Comparison\n",
                "We compare missing cell counts, total data cells, and descriptive statistics across raw, imputed, and aligned datasets to verify zero data loss and exact observation preservation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "stats_before_after = pd.DataFrame([\n",
                "    {\n",
                "        'Dataset': 'IMD Rainfall (0.25 deg)',\n",
                "        'Raw Missing Count': rain_imp_metrics['before_missing_count'],\n",
                "        'Raw Missing %': f\"{rain_imp_metrics['before_missing_pct']}%\",\n",
                "        'Imputed Coastal Cells': rain_imp_metrics['imputed_intermittent_cells_count'],\n",
                "        'Preserved Ocean NaNs': rain_imp_metrics['ocean_cells_preserved_as_nan'],\n",
                "        'Processed Missing %': f\"{rain_imp_metrics['after_missing_pct']}%\"\n",
                "    },\n",
                "    {\n",
                "        'Dataset': 'IMD Max Temp (1.0 deg)',\n",
                "        'Raw Missing Count': max_imp_metrics['before_missing_count'],\n",
                "        'Raw Missing %': f\"{max_imp_metrics['before_missing_pct']}%\",\n",
                "        'Imputed Coastal Cells': max_imp_metrics['imputed_intermittent_cells_count'],\n",
                "        'Preserved Ocean NaNs': max_imp_metrics['ocean_cells_preserved_as_nan'],\n",
                "        'Processed Missing %': f\"{max_imp_metrics['after_missing_pct']}%\"\n",
                "    },\n",
                "    {\n",
                "        'Dataset': 'IMD Min Temp (1.0 deg)',\n",
                "        'Raw Missing Count': min_imp_metrics['before_missing_count'],\n",
                "        'Raw Missing %': f\"{min_imp_metrics['before_missing_pct']}%\",\n",
                "        'Imputed Coastal Cells': min_imp_metrics['imputed_intermittent_cells_count'],\n",
                "        'Preserved Ocean NaNs': min_imp_metrics['ocean_cells_preserved_as_nan'],\n",
                "        'Processed Missing %': f\"{min_imp_metrics['after_missing_pct']}%\"\n",
                "    }\n",
                "])\n",
                "\n",
                "display(stats_before_after)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 3 — Step 9: Processed Dataset Inspection & Visualization\n",
                "We inspect the final aligned NetCDF dataset exported to `data/processed/imd/india_climate_2020_aligned.nc` and render spatial maps for engineered features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Execute main pipeline and inspect output NetCDF datasets\n",
                "pipeline_result = run_preprocessing_pipeline(\n",
                "    year=2020,\n",
                "    save_netcdf=True,\n",
                "    normalize=True\n",
                ")\n",
                "\n",
                "print('\\n=== PIPELINE VERIFICATION SUMMARY ===')\n",
                "print('Pipeline Status:', pipeline_result['pipeline_status'])\n",
                "print('Aligned NetCDF Output Path:', pipeline_result['output_path'])\n",
                "print('Normalized NetCDF Output Path:', pipeline_result['normalized_output_path'])\n",
                "print('Saved Joblib Scaler Registry:', pipeline_result['scaler_registry_path'])\n",
                "\n",
                "# Open and inspect the generated NetCDF file directly from disk\n",
                "processed_nc = xr.open_dataset(pipeline_result['output_path'])\n",
                "print('\\n=== PROCESSED NETCDF DATASET STRUCTURE ===')\n",
                "display(processed_nc)\n",
                "\n",
                "# Plot sample 2D spatial heatmap of derived feature DTR (Diurnal Temperature Range)\n",
                "target_day_idx = 227  # Aug 15, 2020\n",
                "target_date = str(pd.to_datetime(processed_nc.coords['time'].values[target_day_idx]).date())\n",
                "dtr_2d = processed_nc['dtr'].values[target_day_idx, :, :]\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(10, 8))\n",
                "im = ax.pcolormesh(processed_nc.coords['lon'].values, processed_nc.coords['lat'].values, dtr_2d, cmap='YlOrRd', shading='auto')\n",
                "cbar = plt.colorbar(im, ax=ax, label='Diurnal Temperature Range DTR (deg C)')\n",
                "ax.set_title(f'Engineered Feature: Diurnal Temperature Range (DTR = Tmax - Tmin)\\nPeak Monsoon Day: {target_date} (0.25 deg Grid)', fontweight='bold')\n",
                "ax.set_xlabel('Longitude (deg E)')\n",
                "ax.set_ylabel('Latitude (deg N)')\n",
                "ax.grid(True, linestyle='--', alpha=0.5)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "plt.close('all')\n",
                "\n",
                "processed_nc.close()"
            ]
        }
    ]

    nb["cells"].extend(phase3_cells)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Appended Phase 3 cells to notebook. Total cells: {len(nb['cells'])}")

# Now execute the entire notebook to populate outputs
print("Executing notebook using nbconvert...")
with open(nb_path, "r", encoding="utf-8") as f:
    nb_node = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb_node, {"metadata": {"path": str(project_root)}})

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb_node, f)

print("Notebook execution completed successfully! All cell outputs populated.")
