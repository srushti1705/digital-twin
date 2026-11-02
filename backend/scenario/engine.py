"""Core What-If climate perturbation engine."""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

try:
    from ..climate_forecast import predict_grid, _read_observation
    from ..rainfall_api import IMD_LATITUDE_ORIGIN, IMD_LONGITUDE_ORIGIN, IMD_GRID_RESOLUTION
    from ..temperature_api import LATITUDE_ORIGIN, LONGITUDE_ORIGIN, GRID_RESOLUTION
except ImportError:
    from climate_forecast import predict_grid, _read_observation
    from rainfall_api import IMD_LATITUDE_ORIGIN, IMD_LONGITUDE_ORIGIN, IMD_GRID_RESOLUTION
    from temperature_api import LATITUDE_ORIGIN, LONGITUDE_ORIGIN, GRID_RESOLUTION

from .schemas import (
    CompactGridData,
    GridSummaryMetrics,
    ScenarioResult,
    ScenarioRunRequest,
    ScenarioVariableSummary,
)
from .spatial import get_spatial_mask, get_supported_states
from .storage import scenario_store


def _as_json_rows(arr: np.ndarray) -> list[list[float | None]]:
    """Convert a 2D numpy array with NaNs to nested lists with JSON nulls."""
    return [
        [None if not np.isfinite(val) else round(float(val), 3) for val in row]
        for row in arr
    ]


def _compute_metrics(grid: np.ndarray) -> GridSummaryMetrics:
    """Calculate summary statistics over valid, non-missing grid cells."""
    valid_mask = np.isfinite(grid)
    valid_count = int(valid_mask.sum())
    missing_count = int((~valid_mask).sum())
    if valid_count == 0:
        return GridSummaryMetrics(
            mean=None, min=None, max=None, std=None,
            valid_cell_count=0, missing_cell_count=missing_count
        )
    vals = grid[valid_mask]
    return GridSummaryMetrics(
        mean=round(float(np.mean(vals)), 2),
        min=round(float(np.min(vals)), 2),
        max=round(float(np.max(vals)), 2),
        std=round(float(np.std(vals)), 2),
        valid_cell_count=valid_count,
        missing_cell_count=missing_count,
    )


class ScenarioEngine:
    """Computes mathematical climate perturbations and comparisons."""

    @staticmethod
    def load_baseline(
        variable: str,
        baseline_source: str,
        baseline_date: str,
    ) -> np.ndarray:
        """Load baseline grid for rainfall, tmax, or tmin from historical records or predictions."""
        parsed_date = date.fromisoformat(baseline_date)
        if baseline_source == "historical":
            return _read_observation(variable, parsed_date).copy()
        elif baseline_source == "prediction":
            pred = predict_grid(variable, parsed_date)
            arr = np.array(pred.values, dtype=np.float64)
            arr[~np.isfinite(arr)] = np.nan
            return arr
        else:
            raise ValueError(f"Unknown baseline_source: {baseline_source}")

    @classmethod
    def run_scenario(cls, req: ScenarioRunRequest) -> ScenarioResult:
        """Execute a What-If climate scenario."""
        scenario_id = f"scen_{uuid.uuid4().hex[:10]}"
        now_utc = datetime.now(timezone.utc).isoformat()
        warnings: list[str] = []

        # Find friendly name of geographic scope
        supported = get_supported_states()
        scope_entry = next((s for s in supported if s["code"].upper() == req.geographic_scope.upper()), None)
        scope_name = scope_entry["name"] if scope_entry else req.geographic_scope

        assumptions = [
            f"Rainfall perturbation: {req.rainfall_change_percent:+.1f}% applied multiplicatively and clamped to >= 0.0 mm/day.",
            f"Maximum Temperature perturbation: {req.tmax_delta:+.2f} °C applied additively.",
            f"Minimum Temperature perturbation: {req.tmin_delta:+.2f} °C applied additively.",
            f"Spatial scope: {scope_name} (unselected regions remain at baseline).",
            "Missing cells (NaN/nodata) are strictly preserved as nulls and excluded from aggregations.",
        ]

        disclaimer = (
            "This scenario is a mathematical perturbation on baseline meteorological grids. "
            "It is NOT a physically coupled General Circulation Model (GCM) and does not model "
            "thermodynamic atmospheric dynamics, hydrodynamic flood inundation, or agronomic crop feedbacks."
        )

        rainfall_summary: ScenarioVariableSummary | None = None
        tmax_summary: ScenarioVariableSummary | None = None
        tmin_summary: ScenarioVariableSummary | None = None

        rainfall_base_compact: CompactGridData | None = None
        rainfall_scen_compact: CompactGridData | None = None
        rainfall_diff_compact: CompactGridData | None = None

        tmax_base_compact: CompactGridData | None = None
        tmax_scen_compact: CompactGridData | None = None
        tmax_diff_compact: CompactGridData | None = None

        tmin_base_compact: CompactGridData | None = None
        tmin_scen_compact: CompactGridData | None = None
        tmin_diff_compact: CompactGridData | None = None

        # 1. Rainfall Perturbation
        if req.variable_selection in {"rainfall", "both"}:
            try:
                base_rain = cls.load_baseline("rainfall", req.baseline_source, req.baseline_date)
                rain_mask = get_spatial_mask(req.geographic_scope, "rainfall")
                affected_cells = int((rain_mask & np.isfinite(base_rain)).sum())

                # Formula: scenario_rainfall = baseline * (1 + pct/100), clamped to min 0
                scen_rain = base_rain.copy()
                scen_rain[rain_mask] = np.maximum(0.0, base_rain[rain_mask] * (1.0 + req.rainfall_change_percent / 100.0))
                diff_rain = scen_rain - base_rain

                base_rain_metrics = _compute_metrics(base_rain)
                scen_rain_metrics = _compute_metrics(scen_rain)
                delta_mean = (
                    round(scen_rain_metrics.mean - base_rain_metrics.mean, 2)
                    if scen_rain_metrics.mean is not None and base_rain_metrics.mean is not None
                    else None
                )
                delta_pct = (
                    round((delta_mean / base_rain_metrics.mean) * 100.0, 1)
                    if delta_mean is not None and base_rain_metrics.mean and base_rain_metrics.mean > 0
                    else 0.0
                )

                rainfall_summary = ScenarioVariableSummary(
                    variable="rainfall",
                    units="mm/day",
                    baseline=base_rain_metrics,
                    scenario=scen_rain_metrics,
                    delta_mean=delta_mean,
                    delta_percent=delta_pct,
                    affected_cells_count=affected_cells,
                )

                rain_lats = (IMD_LATITUDE_ORIGIN + np.arange(base_rain.shape[0]) * IMD_GRID_RESOLUTION).tolist()
                rain_lons = (IMD_LONGITUDE_ORIGIN + np.arange(base_rain.shape[1]) * IMD_GRID_RESOLUTION).tolist()

                rainfall_base_compact = CompactGridData(
                    variable="rainfall",
                    units="mm/day",
                    spatial_resolution_degrees=IMD_GRID_RESOLUTION,
                    grid_shape=list(base_rain.shape),
                    latitudes=rain_lats,
                    longitudes=rain_lons,
                    values=_as_json_rows(base_rain),
                    valid_cell_count=base_rain_metrics.valid_cell_count,
                    missing_cell_count=base_rain_metrics.missing_cell_count,
                )
                rainfall_scen_compact = CompactGridData(
                    variable="rainfall",
                    units="mm/day",
                    spatial_resolution_degrees=IMD_GRID_RESOLUTION,
                    grid_shape=list(scen_rain.shape),
                    latitudes=rain_lats,
                    longitudes=rain_lons,
                    values=_as_json_rows(scen_rain),
                    valid_cell_count=scen_rain_metrics.valid_cell_count,
                    missing_cell_count=scen_rain_metrics.missing_cell_count,
                )
                rainfall_diff_compact = CompactGridData(
                    variable="rainfall_difference",
                    units="mm/day",
                    spatial_resolution_degrees=IMD_GRID_RESOLUTION,
                    grid_shape=list(diff_rain.shape),
                    latitudes=rain_lats,
                    longitudes=rain_lons,
                    values=_as_json_rows(diff_rain),
                    valid_cell_count=scen_rain_metrics.valid_cell_count,
                    missing_cell_count=scen_rain_metrics.missing_cell_count,
                )
            except Exception as err:
                warnings.append(f"Rainfall baseline could not be loaded: {err}")

        # 2. Temperature Perturbations (Tmax & Tmin)
        if req.variable_selection in {"temperature", "both"}:
            temp_mask = get_spatial_mask(req.geographic_scope, "temperature")
            scen_tmax_arr: np.ndarray | None = None
            scen_tmin_arr: np.ndarray | None = None

            # Tmax
            try:
                base_tmax = cls.load_baseline("tmax", req.baseline_source, req.baseline_date)
                scen_tmax = base_tmax.copy()
                scen_tmax[temp_mask] = base_tmax[temp_mask] + req.tmax_delta
                diff_tmax = scen_tmax - base_tmax
                scen_tmax_arr = scen_tmax

                base_tmax_metrics = _compute_metrics(base_tmax)
                scen_tmax_metrics = _compute_metrics(scen_tmax)
                delta_tmax_mean = (
                    round(scen_tmax_metrics.mean - base_tmax_metrics.mean, 2)
                    if scen_tmax_metrics.mean is not None and base_tmax_metrics.mean is not None
                    else None
                )
                tmax_summary = ScenarioVariableSummary(
                    variable="tmax",
                    units="°C",
                    baseline=base_tmax_metrics,
                    scenario=scen_tmax_metrics,
                    delta_mean=delta_tmax_mean,
                    delta_percent=None,
                    affected_cells_count=int((temp_mask & np.isfinite(base_tmax)).sum()),
                )

                temp_lats = (LATITUDE_ORIGIN + np.arange(base_tmax.shape[0]) * GRID_RESOLUTION).tolist()
                temp_lons = (LONGITUDE_ORIGIN + np.arange(base_tmax.shape[1]) * GRID_RESOLUTION).tolist()

                tmax_base_compact = CompactGridData(
                    variable="tmax",
                    units="°C",
                    spatial_resolution_degrees=GRID_RESOLUTION,
                    grid_shape=list(base_tmax.shape),
                    latitudes=temp_lats,
                    longitudes=temp_lons,
                    values=_as_json_rows(base_tmax),
                    valid_cell_count=base_tmax_metrics.valid_cell_count,
                    missing_cell_count=base_tmax_metrics.missing_cell_count,
                )
                tmax_scen_compact = CompactGridData(
                    variable="tmax",
                    units="°C",
                    spatial_resolution_degrees=GRID_RESOLUTION,
                    grid_shape=list(scen_tmax.shape),
                    latitudes=temp_lats,
                    longitudes=temp_lons,
                    values=_as_json_rows(scen_tmax),
                    valid_cell_count=scen_tmax_metrics.valid_cell_count,
                    missing_cell_count=scen_tmax_metrics.missing_cell_count,
                )
                tmax_diff_compact = CompactGridData(
                    variable="tmax_difference",
                    units="°C",
                    spatial_resolution_degrees=GRID_RESOLUTION,
                    grid_shape=list(diff_tmax.shape),
                    latitudes=temp_lats,
                    longitudes=temp_lons,
                    values=_as_json_rows(diff_tmax),
                    valid_cell_count=scen_tmax_metrics.valid_cell_count,
                    missing_cell_count=scen_tmax_metrics.missing_cell_count,
                )
            except Exception as err:
                warnings.append(f"Tmax baseline could not be loaded: {err}")

            # Tmin
            try:
                base_tmin = cls.load_baseline("tmin", req.baseline_source, req.baseline_date)
                scen_tmin = base_tmin.copy()
                scen_tmin[temp_mask] = base_tmin[temp_mask] + req.tmin_delta
                diff_tmin = scen_tmin - base_tmin
                scen_tmin_arr = scen_tmin

                base_tmin_metrics = _compute_metrics(base_tmin)
                scen_tmin_metrics = _compute_metrics(scen_tmin)
                delta_tmin_mean = (
                    round(scen_tmin_metrics.mean - base_tmin_metrics.mean, 2)
                    if scen_tmin_metrics.mean is not None and base_tmin_metrics.mean is not None
                    else None
                )
                tmin_summary = ScenarioVariableSummary(
                    variable="tmin",
                    units="°C",
                    baseline=base_tmin_metrics,
                    scenario=scen_tmin_metrics,
                    delta_mean=delta_tmin_mean,
                    delta_percent=None,
                    affected_cells_count=int((temp_mask & np.isfinite(base_tmin)).sum()),
                )

                temp_lats = (LATITUDE_ORIGIN + np.arange(base_tmin.shape[0]) * GRID_RESOLUTION).tolist()
                temp_lons = (LONGITUDE_ORIGIN + np.arange(base_tmin.shape[1]) * GRID_RESOLUTION).tolist()

                tmin_base_compact = CompactGridData(
                    variable="tmin",
                    units="°C",
                    spatial_resolution_degrees=GRID_RESOLUTION,
                    grid_shape=list(base_tmin.shape),
                    latitudes=temp_lats,
                    longitudes=temp_lons,
                    values=_as_json_rows(base_tmin),
                    valid_cell_count=base_tmin_metrics.valid_cell_count,
                    missing_cell_count=base_tmin_metrics.missing_cell_count,
                )
                tmin_scen_compact = CompactGridData(
                    variable="tmin",
                    units="°C",
                    spatial_resolution_degrees=GRID_RESOLUTION,
                    grid_shape=list(scen_tmin.shape),
                    latitudes=temp_lats,
                    longitudes=temp_lons,
                    values=_as_json_rows(scen_tmin),
                    valid_cell_count=scen_tmin_metrics.valid_cell_count,
                    missing_cell_count=scen_tmin_metrics.missing_cell_count,
                )
                tmin_diff_compact = CompactGridData(
                    variable="tmin_difference",
                    units="°C",
                    spatial_resolution_degrees=GRID_RESOLUTION,
                    grid_shape=list(diff_tmin.shape),
                    latitudes=temp_lats,
                    longitudes=temp_lons,
                    values=_as_json_rows(diff_tmin),
                    valid_cell_count=scen_tmin_metrics.valid_cell_count,
                    missing_cell_count=scen_tmin_metrics.missing_cell_count,
                )
            except Exception as err:
                warnings.append(f"Tmin baseline could not be loaded: {err}")

            # Temperature Inversion Check: Tmin > Tmax
            if scen_tmax_arr is not None and scen_tmin_arr is not None:
                inverted = (scen_tmin_arr > scen_tmax_arr) & np.isfinite(scen_tmin_arr) & np.isfinite(scen_tmax_arr)
                if inverted.any():
                    inv_count = int(inverted.sum())
                    warnings.append(
                        f"Physical Inversion Notice: In {inv_count} cell(s), the requested perturbation resulted in "
                        f"Tmin exceeding Tmax (max inversion: {float(np.max(scen_tmin_arr[inverted] - scen_tmax_arr[inverted])):.1f}°C). "
                        "Check realistic temperature bounds."
                    )

        result = ScenarioResult(
            scenario_id=scenario_id,
            created_at_utc=now_utc,
            parameters=req,
            geographic_scope_name=scope_name,
            scientific_disclaimer=disclaimer,
            mathematical_assumptions=assumptions,
            warnings=warnings,
            rainfall_summary=rainfall_summary,
            tmax_summary=tmax_summary,
            tmin_summary=tmin_summary,
            rainfall_baseline_grid=rainfall_base_compact,
            rainfall_scenario_grid=rainfall_scen_compact,
            rainfall_difference_grid=rainfall_diff_compact,
            tmax_baseline_grid=tmax_base_compact,
            tmax_scenario_grid=tmax_scen_compact,
            tmax_difference_grid=tmax_diff_compact,
            tmin_baseline_grid=tmin_base_compact,
            tmin_scenario_grid=tmin_scen_compact,
            tmin_difference_grid=tmin_diff_compact,
        )

        scenario_store.put(result)
        return result

