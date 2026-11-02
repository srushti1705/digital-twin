"""FastAPI router for interactive What-If climate scenarios."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException

if __package__:
    from .scenario import (
        ScenarioCompareRequest,
        ScenarioCompareResult,
        ScenarioEngine,
        ScenarioResult,
        ScenarioRunRequest,
        get_supported_states,
        scenario_store,
    )
else:
    from scenario import (
        ScenarioCompareRequest,
        ScenarioCompareResult,
        ScenarioEngine,
        ScenarioResult,
        ScenarioRunRequest,
        get_supported_states,
        scenario_store,
    )

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])


@router.get("/scopes")
def get_scenario_scopes() -> list[dict[str, str]]:
    """Return list of supported geographic scopes (All India and Indian States/UTs)."""
    return get_supported_states()


@router.post("/run", response_model=ScenarioResult)
def run_scenario(request: ScenarioRunRequest) -> ScenarioResult:
    """Run an interactive What-If perturbation scenario on rainfall or temperature grids."""
    try:
        return ScenarioEngine.run_scenario(request)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=f"Baseline dataset unavailable: {err}") from err
    except ValueError as err:
        raise HTTPException(status_code=422, detail=f"Scenario parameter error: {err}") from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Scenario calculation error: {err}") from err


@router.get("/{scenario_id}", response_model=ScenarioResult)
def get_scenario(scenario_id: str) -> ScenarioResult:
    """Retrieve full results and georeferenced grids for an executed scenario."""
    result = scenario_store.get(scenario_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found in active session.")
    return result


@router.get("/{scenario_id}/summary")
def get_scenario_summary(scenario_id: str) -> dict[str, Any]:
    """Retrieve statistical summary and delta metrics for an executed scenario without large grid matrices."""
    result = scenario_store.get(scenario_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found in active session.")
    return {
        "scenario_id": result.scenario_id,
        "created_at_utc": result.created_at_utc,
        "parameters": result.parameters,
        "geographic_scope_name": result.geographic_scope_name,
        "scientific_disclaimer": result.scientific_disclaimer,
        "mathematical_assumptions": result.mathematical_assumptions,
        "warnings": result.warnings,
        "rainfall_summary": result.rainfall_summary,
        "tmax_summary": result.tmax_summary,
        "tmin_summary": result.tmin_summary,
    }


@router.post("/compare", response_model=ScenarioCompareResult)
def compare_scenarios(req: ScenarioCompareRequest) -> ScenarioCompareResult:
    """Compare two scenario results side-by-side."""
    scen_a = scenario_store.get(req.scenario_id_a)
    if scen_a is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id_a}' not found.")
    scen_b = scenario_store.get(req.scenario_id_b)
    if scen_b is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id_b}' not found.")

    rain_a = scen_a.rainfall_summary.scenario.mean if scen_a.rainfall_summary else None
    rain_b = scen_b.rainfall_summary.scenario.mean if scen_b.rainfall_summary else None
    rain_diff = round(rain_b - rain_a, 2) if (rain_a is not None and rain_b is not None) else None

    tmax_a = scen_a.tmax_summary.scenario.mean if scen_a.tmax_summary else None
    tmax_b = scen_b.tmax_summary.scenario.mean if scen_b.tmax_summary else None
    tmax_diff = round(tmax_b - tmax_a, 2) if (tmax_a is not None and tmax_b is not None) else None

    tmin_a = scen_a.tmin_summary.scenario.mean if scen_a.tmin_summary else None
    tmin_b = scen_b.tmin_summary.scenario.mean if scen_b.tmin_summary else None
    tmin_diff = round(tmin_b - tmin_a, 2) if (tmin_a is not None and tmin_b is not None) else None

    rain_str = f"{rain_diff:+.2f} mm/day" if rain_diff is not None else "N/A"
    tmax_str = f"{tmax_diff:+.2f} °C" if tmax_diff is not None else "N/A"
    tmin_str = f"{tmin_diff:+.2f} °C" if tmin_diff is not None else "N/A"

    summary_text = (
        f"Comparison of '{scen_a.parameters.scenario_name}' vs '{scen_b.parameters.scenario_name}': "
        f"Rainfall Delta: {rain_str}; Tmax Delta: {tmax_str}; Tmin Delta: {tmin_str}."
    )

    return ScenarioCompareResult(
        scenario_a_id=scen_a.scenario_id,
        scenario_a_name=scen_a.parameters.scenario_name,
        scenario_b_id=scen_b.scenario_id,
        scenario_b_name=scen_b.parameters.scenario_name,
        rainfall_mean_a=rain_a,
        rainfall_mean_b=rain_b,
        rainfall_delta_b_minus_a=rain_diff,
        tmax_mean_a=tmax_a,
        tmax_mean_b=tmax_b,
        tmax_delta_b_minus_a=tmax_diff,
        tmin_mean_a=tmin_a,
        tmin_mean_b=tmin_b,
        tmin_delta_b_minus_a=tmin_diff,
        summary_text=summary_text,
    )
