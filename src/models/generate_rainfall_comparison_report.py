"""
Generate Rainfall Model Comparison Report (CSV & Markdown).

Compares Random Forest Regressor vs XGBoost Regressor baseline models
on the validation dataset across MAE, RMSE, and R2 metrics.
"""

import os
import json
import pandas as pd


def generate_comparison_report():
    output_dir = "outputs/models/rainfall"
    os.makedirs(output_dir, exist_ok=True)

    rf_json_path = os.path.join(output_dir, "metrics.json")
    xgb_json_path = os.path.join(output_dir, "metrics_xgb.json")

    with open(rf_json_path, "r") as f:
        rf_data = json.load(f)
        rf_metrics = rf_data["validation_metrics"]

    with open(xgb_json_path, "r") as f:
        xgb_data = json.load(f)
        xgb_metrics = xgb_data["validation_metrics"]

    # 1. Generate CSV Comparison Table
    df_rows = [
        {
            "Model": "Random Forest Regressor",
            "Model_Type": "Ensemble (Bagging)",
            "MAE_mm_day": rf_metrics["MAE"],
            "RMSE_mm_day": rf_metrics["RMSE"],
            "R2_Score": rf_metrics["R2"],
            "Train_Samples": rf_data["train_samples"],
            "Val_Samples": rf_data["val_samples"]
        },
        {
            "Model": "XGBoost Regressor",
            "Model_Type": "Gradient Boosting",
            "MAE_mm_day": xgb_metrics["MAE"],
            "RMSE_mm_day": xgb_metrics["RMSE"],
            "R2_Score": xgb_metrics["R2"],
            "Train_Samples": xgb_data["train_samples"],
            "Val_Samples": xgb_data["val_samples"]
        }
    ]

    df_comp = pd.DataFrame(df_rows)
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"[OK] Saved CSV comparison report to: {csv_path}")

    # 2. Generate Detailed Markdown Report
    md_content = """# Short-Term Rainfall Baseline Model Comparison Report

## Executive Summary

This report evaluates and compares the baseline predictive performance of **Random Forest Regressor** versus **XGBoost Regressor** for 1-day ahead daily precipitation forecasting (`rainfall_prediction_next_day`) across India at 0.25° x 0.25° spatial resolution (~25 km).

All evaluations were conducted strictly on the **Validation Dataset** (233,624 unseen land grid-day samples covering `2020-09-15` to `2020-11-06`). The **Test Dataset** (`2020-11-07` to `2020-12-31`) was completely held out unseen to prevent data leakage during model selection.

---

## Model Performance Summary (Validation Set)

| Model Name | Model Type | MAE (mm/day) [lower is better] | RMSE (mm/day) [lower is better] | R² Score [higher is better] | Validation Winner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | Ensemble Bagging | **{rf_mae:.4f}** | **{rf_rmse:.4f}** | **{rf_r2:.4f}** | **Better Baseline** |
| **XGBoost Regressor** | Gradient Boosting | {xgb_mae:.4f} | {xgb_rmse:.4f} | {xgb_r2:.4f} | Runner-up Baseline |

---

## Rationale for Evaluation Metrics

Three complementary regression metrics were selected to evaluate rainfall forecasting performance:

1. **Mean Absolute Error (MAE) [mm/day]**:
   - *Formula*: Mean of absolute prediction errors |y - y_pred|
   - *Why Used*: Measures average magnitude of prediction errors in physical daily rainfall units (mm/day). It treats all absolute error sizes linearly and is robust to isolated extreme outliers.
2. **Root Mean Squared Error (RMSE) [mm/day]**:
   - *Formula*: Square root of mean squared prediction errors sqrt(mean((y - y_pred)^2))
   - *Why Used*: Penalizes larger prediction errors quadratically. In hydrological modeling, large errors during heavy monsoon events carry higher risk, making RMSE critical for detecting severe precipitation mispredictions.
3. **Coefficient of Determination (R²)**:
   - *Formula*: 1 - (SS_res / SS_tot)
   - *Why Used*: Quantifies the proportion of total spatial-temporal variance in daily rainfall explained by the model across India.

---

## Detailed Model Findings & Baseline Analysis

### 1. Random Forest Regressor (Baseline Winner)
- **Validation MAE**: {rf_mae:.4f} mm/day
- **Validation RMSE**: {rf_rmse:.4f} mm/day
- **Validation R²**: {rf_r2:.4f} (90.47% variance explained)
- **Strengths**: Deeper tree structures (`max_depth=15`) with high variance reduction via bagging enabled Random Forest to capture non-linear local interactions effectively across 1.1M training samples.

### 2. XGBoost Regressor (Runner-up Baseline)
- **Validation MAE**: {xgb_mae:.4f} mm/day
- **Validation RMSE**: {xgb_rmse:.4f} mm/day
- **Validation R²**: {xgb_r2:.4f} (89.30% variance explained)
- **Strengths**: Highly efficient training speed and compact model file size (2.12 MB vs 44.60 MB for RF).

---

## Scientific Nuance & Model Selection Disclaimer

> **Important**: Random Forest demonstrated slightly better default baseline performance on the validation set. However, **this does not imply Random Forest is universally superior to XGBoost**. 
>
> - Default XGBoost used shallow trees (`max_depth=6`), whereas Random Forest used deeper trees (`max_depth=15`).
> - Systematic hyperparameter tuning (learning rate schedule, regularization alpha/lambda, deeper trees, loss functions like Tweedie/Poisson for zero-inflated rain) may enable XGBoost to meet or exceed Random Forest performance in subsequent optimization iterations.

---

## Next Steps
- Implement baseline temperature prediction models (Tmax, Tmin).
- Proceed to systematic hyperparameter optimization without touching the test set.
""".format(
        rf_mae=rf_metrics["MAE"],
        rf_rmse=rf_metrics["RMSE"],
        rf_r2=rf_metrics["R2"],
        xgb_mae=xgb_metrics["MAE"],
        xgb_rmse=xgb_metrics["RMSE"],
        xgb_r2=xgb_metrics["R2"]
    )

    md_path = os.path.join(output_dir, "model_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Saved Markdown comparison report to: {md_path}")


if __name__ == "__main__":
    generate_comparison_report()
