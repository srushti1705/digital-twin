"""Unit tests for What-If scenario engine, spatial masking, and API routes."""

from __future__ import annotations

from unittest.mock import patch
import numpy as np
import pytest
from pydantic import ValidationError

from backend.scenario.engine import ScenarioEngine, _compute_metrics
from backend.scenario.schemas import (
    ScenarioCompareRequest,
    ScenarioRunRequest,
)
from backend.scenario.spatial import get_spatial_mask, get_supported_states
from backend.scenario_api import (
    compare_scenarios,
    get_scenario,
    get_scenario_scopes,
    get_scenario_summary,
    run_scenario,
)


def test_rainfall_perturbation_math_and_clamping():
    """Verify that rainfall scaling is multiplicative, preserves missing cells, and clamps at 0.0."""
    baseline = np.array([
        [10.0, 0.0, np.nan],
        [50.0, 25.0, np.nan],
    ])
    mask = np.ones(baseline.shape, dtype=bool)

    # 1. Increase by +20%
    pct = 20.0
    scen = baseline.copy()
    scen[mask] = np.maximum(0.0, baseline[mask] * (1.0 + pct / 100.0))
    assert scen[0, 0] == pytest.approx(12.0)
    assert scen[0, 1] == pytest.approx(0.0)
    assert np.isnan(scen[0, 2])
    assert scen[1, 0] == pytest.approx(60.0)

    # 2. Clamped at -100% -> must be 0, never negative
    pct = -100.0
    scen_zero = np.maximum(0.0, baseline * (1.0 + pct / 100.0))
    assert scen_zero[0, 0] == 0.0
    assert scen_zero[1, 0] == 0.0
    assert np.isnan(scen_zero[0, 2])


def test_temperature_perturbation_math():
    """Verify additive temperature shifts and NaN preservation."""
    base_tmax = np.array([[32.0, 35.5], [np.nan, 28.0]])
    delta = 2.5
    scen_tmax = base_tmax + delta
    assert scen_tmax[0, 0] == pytest.approx(34.5)
    assert scen_tmax[0, 1] == pytest.approx(38.0)
    assert np.isnan(scen_tmax[1, 0])
    assert scen_tmax[1, 1] == pytest.approx(30.5)


def test_compute_metrics_ignores_nans():
    """Verify statistical metrics calculation properly filters out missing cells."""
    grid = np.array([[10.0, 20.0], [np.nan, 30.0]])
    metrics = _compute_metrics(grid)
    assert metrics.valid_cell_count == 3
    assert metrics.missing_cell_count == 1
    assert metrics.mean == pytest.approx(20.0)
    assert metrics.min == pytest.approx(10.0)
    assert metrics.max == pytest.approx(30.0)


def test_validation_ranges_reject_invalid_inputs():
    """Ensure extreme or invalid input ranges are rejected by schema validation."""
    with pytest.raises(ValidationError):
        ScenarioRunRequest(
            baseline_date="2025-07-15",
            rainfall_change_percent=-150.0,  # Below -100%
        )

    with pytest.raises(ValidationError):
        ScenarioRunRequest(
            baseline_date="2025-07-15",
            tmax_delta=25.0,  # Above +15°C
        )


def test_spatial_mask_all_india_and_state():
    """Verify spatial masking for All India vs a specific state."""
    mask_all = get_spatial_mask("all_india", "rainfall")
    assert mask_all.shape == (129, 135)
    assert mask_all.all()

    # Specific state, e.g., Maharashtra (IN-MH)
    mask_mh = get_spatial_mask("IN-MH", "rainfall")
    assert mask_mh.shape == (129, 135)
    assert mask_mh.sum() > 0  # Contains cells
    assert not mask_mh.all()  # Does not cover entire subcontinent

    # Check that temperature mask has temperature shape
    mask_temp_mh = get_spatial_mask("IN-MH", "temperature")
    assert mask_temp_mh.shape == (31, 31)
    assert mask_temp_mh.sum() > 0


def test_scenario_run_and_inversion_warning():
    """Verify scenario execution produces physical inversion warning when Tmin > Tmax."""
    dummy_tmax = np.full((31, 31), 30.0)
    dummy_tmin = np.full((31, 31), 25.0)

    req = ScenarioRunRequest(
        scenario_name="Heat Inversion Test",
        geographic_scope="all_india",
        baseline_source="historical",
        baseline_date="2025-06-01",
        variable_selection="temperature",
        tmax_delta=-6.0,  # 30 - 6 = 24
        tmin_delta=+2.0,  # 25 + 2 = 27 -> Tmin (27) > Tmax (24)
    )

    with patch.object(ScenarioEngine, "load_baseline") as mock_load:
        def side_effect(var, src, dt):
            return dummy_tmax if var == "tmax" else dummy_tmin
        mock_load.side_effect = side_effect

        result = ScenarioEngine.run_scenario(req)
        assert result.tmax_summary is not None
        assert result.tmin_summary is not None
        assert result.tmax_summary.scenario.mean == pytest.approx(24.0)
        assert result.tmin_summary.scenario.mean == pytest.approx(27.0)
        # Inversion warning must be present
        assert any("Physical Inversion" in w for w in result.warnings)


def test_scenario_api_endpoints():
    """Directly test scenario router endpoint handlers."""
    # 1. Scopes
    scopes = get_scenario_scopes()
    assert len(scopes) > 1
    assert any(s["code"] == "all_india" for s in scopes)

    # 2. Run
    dummy_rain = np.full((129, 135), 20.0)
    req = ScenarioRunRequest(
        scenario_name="Monsoon Surge +25%",
        geographic_scope="all_india",
        baseline_source="historical",
        baseline_date="2025-07-20",
        variable_selection="rainfall",
        rainfall_change_percent=25.0,
    )

    with patch.object(ScenarioEngine, "load_baseline", return_value=dummy_rain):
        result = run_scenario(req)
        assert result.scenario_id.startswith("scen_")
        assert result.rainfall_summary is not None
        assert result.rainfall_summary.scenario.mean == pytest.approx(25.0)

        # 3. Retrieve
        retrieved = get_scenario(result.scenario_id)
        assert retrieved.scenario_id == result.scenario_id

        # 4. Summary
        summary = get_scenario_summary(result.scenario_id)
        assert summary["scenario_id"] == result.scenario_id
        assert summary["rainfall_summary"] is not None

        # 5. Compare with second scenario
        req2 = ScenarioRunRequest(
            scenario_name="Monsoon Deficit -20%",
            geographic_scope="all_india",
            baseline_source="historical",
            baseline_date="2025-07-20",
            variable_selection="rainfall",
            rainfall_change_percent=-20.0,
        )
        result2 = run_scenario(req2)
        compare_res = compare_scenarios(
            ScenarioCompareRequest(
                scenario_id_a=result.scenario_id,
                scenario_id_b=result2.scenario_id,
            )
        )
        assert compare_res.rainfall_delta_b_minus_a is not None
        assert "Comparison" in compare_res.summary_text

