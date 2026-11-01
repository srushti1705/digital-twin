# Preliminary One-day IMD Forecast Evaluation

## Data and method

This evaluation used only the local 2025 annual IMD grids:

- Rainfall: 0.25° grid, mm/day.
- Tmax and Tmin: 1° grids, °C.

The target is the next daily observation. Each native grid cell uses the previous
seven consecutive daily observations, rolling three- and seven-day means, target
day-of-year sine/cosine, and native latitude/longitude. Cells with missing values
in any input day or in the target are excluded; no filling, interpolation, or
upsampling is performed.

Targets are split chronologically: 70% train, 15% validation, 15% held-out test.
Model choice uses validation only. XGBoost is selected only when its validation
RMSE is at least 1% lower than persistence and its validation MAE is no more than
1% worse. The selected model is then refit on the train-plus-validation period.
Persistence predicts the latest observed value in the same grid cell.

## Current held-out results

| Variable | Selected model | Held-out samples | Selected MAE | Persistence MAE | Selected RMSE | Persistence RMSE | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Rainfall | XGBoost | 273,020 | 0.6684 mm/day | 0.3276 mm/day | 2.6518 mm/day | 3.1240 mm/day | Lower RMSE, substantially higher MAE than persistence. Not a consistent improvement. |
| Tmax | Persistence | 19,517 | 0.6180°C | 0.6180°C | 0.8663°C | 0.8663°C | XGBoost failed the validation-selection threshold; output equals baseline. |
| Tmin | Persistence | 19,497 | 0.6615°C | 0.6615°C | 0.8793°C | 0.8793°C | XGBoost validation metrics were worse; output equals baseline. |

The validation comparison likewise showed the Tmax candidate's tiny RMSE advantage
was insufficient and its MAE was worse. Rainfall met the predeclared validation
threshold, but its final held-out MAE regression shows that the one-year test does
not establish reliable forecast skill.

## Coverage and limitations

- Training and validation targets span 2025-01-08 through 2025-11-06; held-out
  targets span 2025-11-07 through 2025-12-31.
- With the latest installed observation date (2025-12-31), the latest available
  one-day output is dated 2026-01-01. It is an out-of-sample projection, **not a
  current weather forecast**.
- This is a preliminary single-year evaluation. It does not show multi-year,
  seasonal, event-specific, or operational forecast performance.
- The rainfall and temperature grids are independent products and retain their
  respective native resolutions.

To reproduce, install the three local 2025 IMD files and run
`python train_climate_models.py` from this directory. The generated joblib models
and detailed evaluation JSON are local ignored artifacts in
`../models/climate_forecasts/` and `../outputs/climate_forecasts/`.
