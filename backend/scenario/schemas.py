"""Schemas for the interactive What-If climate scenario perturbation engine."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ScenarioRunRequest(BaseModel):
    """Parameters required to compute an interactive What-If climate perturbation."""

    scenario_name: str = Field(
        default="Custom Scenario",
        description="Human-readable scenario label or simulation title",
    )
    geographic_scope: str = Field(
        default="all_india",
        description="'all_india' or a specific state/UT ISO code, e.g. 'IN-MH'",
    )
    baseline_source: Literal["historical", "prediction"] = Field(
        default="historical",
        description="Baseline data: 'historical' (IMD recorded grid) or 'prediction' (AI forecast)",
    )
    baseline_date: str = Field(
        description="Date of the baseline observation or forecast initialization (YYYY-MM-DD)",
    )
    variable_selection: Literal["rainfall", "temperature", "both"] = Field(
        default="both",
        description="Target climate variables to perturb: 'rainfall', 'temperature', or 'both'",
    )
    rainfall_change_percent: float = Field(
        default=0.0,
        ge=-100.0,
        le=300.0,
        description="Percentage change applied to baseline rainfall (-100% to +300%)",
    )
    tmax_delta: float = Field(
        default=0.0,
        ge=-15.0,
        le=15.0,
        description="Additive change in Maximum Temperature in °C (-15°C to +15°C)",
    )
    tmin_delta: float = Field(
        default=0.0,
        ge=-15.0,
        le=15.0,
        description="Additive change in Minimum Temperature in °C (-15°C to +15°C)",
    )
    duration_days: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Scenario duration in days (currently single-day perturbation supported)",
    )
    description: str | None = Field(
        default=None,
        description="Optional researcher notes or context",
    )


class GridSummaryMetrics(BaseModel):
    """Statistical summary over valid grid cells for baseline, scenario, or difference."""

    mean: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None
    valid_cell_count: int
    missing_cell_count: int


class ScenarioVariableSummary(BaseModel):
    """Comparison metrics for a single perturbed variable."""

    variable: str
    units: str
    baseline: GridSummaryMetrics
    scenario: GridSummaryMetrics
    delta_mean: float | None = None
    delta_percent: float | None = None
    affected_cells_count: int


class CompactGridData(BaseModel):
    """Georeferenced grid values formatted for Cesium SingleTileImagery rendering."""

    variable: str
    units: str
    spatial_resolution_degrees: float
    grid_shape: list[int]
    latitudes: list[float]
    longitudes: list[float]
    values: list[list[float | None]]
    valid_cell_count: int
    missing_cell_count: int


class ScenarioResult(BaseModel):
    """Full execution output of a What-If climate perturbation scenario."""

    scenario_id: str
    created_at_utc: str
    parameters: ScenarioRunRequest
    geographic_scope_name: str
    scientific_disclaimer: str
    mathematical_assumptions: list[str]
    warnings: list[str]
    rainfall_summary: ScenarioVariableSummary | None = None
    tmax_summary: ScenarioVariableSummary | None = None
    tmin_summary: ScenarioVariableSummary | None = None
    rainfall_baseline_grid: CompactGridData | None = None
    rainfall_scenario_grid: CompactGridData | None = None
    rainfall_difference_grid: CompactGridData | None = None
    tmax_baseline_grid: CompactGridData | None = None
    tmax_scenario_grid: CompactGridData | None = None
    tmax_difference_grid: CompactGridData | None = None
    tmin_baseline_grid: CompactGridData | None = None
    tmin_scenario_grid: CompactGridData | None = None
    tmin_difference_grid: CompactGridData | None = None


class ScenarioCompareRequest(BaseModel):
    """Request to compare two scenario IDs or baseline vs scenario."""

    scenario_id_a: str
    scenario_id_b: str


class ScenarioCompareResult(BaseModel):
    """Side-by-side metric comparison between two scenarios."""

    scenario_a_id: str
    scenario_a_name: str
    scenario_b_id: str
    scenario_b_name: str
    rainfall_mean_a: float | None = None
    rainfall_mean_b: float | None = None
    rainfall_delta_b_minus_a: float | None = None
    tmax_mean_a: float | None = None
    tmax_mean_b: float | None = None
    tmax_delta_b_minus_a: float | None = None
    tmin_mean_a: float | None = None
    tmin_mean_b: float | None = None
    tmin_delta_b_minus_a: float | None = None
    summary_text: str

