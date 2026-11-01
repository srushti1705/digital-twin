from datetime import date

import numpy as np
import pytest

from backend import climate_forecast, prediction_api
from scripts.download_imd_grids import _FormParser, _expected_bytes


def test_native_forecast_features_use_only_complete_history():
    history = np.ones((7, 31, 31), dtype=np.float32)
    history[-1] = 3
    history[-2] = 2
    history[0, 0, 0] = np.nan

    features, valid, shape = climate_forecast._feature_grid(
        history, date(2025, 1, 8), "tmax"
    )

    assert features.shape == (31, 31, 7)
    assert shape == (31, 31)
    assert valid.sum() == 31 * 31 - 1
    assert not valid[0, 0]
    np.testing.assert_allclose(features[1, 1, :3], [3, 2, 10 / 7])
    assert features[1, 1, 5] == pytest.approx(8.5)
    assert features[1, 1, 6] == pytest.approx(68.5)


def test_forecast_feature_builder_rejects_non_native_history_shape():
    with pytest.raises(ValueError, match="seven consecutive"):
        climate_forecast._feature_grid(
            np.ones((6, 31, 31), dtype=np.float32), date(2025, 1, 8), "tmax"
        )


def test_xgboost_selection_requires_material_validation_gain_without_mae_regression():
    persistence = {"RMSE": 10.0, "MAE": 5.0}

    assert climate_forecast._select_xgboost(
        {"RMSE": 9.8, "MAE": 5.0}, persistence
    )
    assert not climate_forecast._select_xgboost(
        {"RMSE": 9.95, "MAE": 5.0}, persistence
    )
    assert not climate_forecast._select_xgboost(
        {"RMSE": 9.0, "MAE": 5.1}, persistence
    )


def test_prediction_api_returns_actual_grid_values_and_nearest_cell(monkeypatch):
    grid = climate_forecast.ForecastGrid(
        variable="tmax",
        initialization_date="2025-12-31",
        forecast_date="2026-01-01",
        run_type="out_of_sample_projection",
        model_name="Persistence baseline",
        units="°C",
        spatial_resolution_degrees=1.0,
        latitudes=[19.5, 20.5],
        longitudes=[73.5, 74.5],
        values=[[25.0, None], [27.0, 28.0]],
        valid_cell_count=3,
        missing_cell_count=1,
        source="India Meteorological Department",
        training_period={"start": "2025-01-08", "end": "2025-11-06"},
        metadata={
            "test_metrics": {"sample_count": 1, "MAE": 1.0, "RMSE": 1.0, "R2": 0.0},
            "persistence_test_metrics": {
                "sample_count": 1,
                "MAE": 1.0,
                "RMSE": 1.0,
                "R2": 0.0,
            },
            "selection_criterion": {
                "minimum_validation_rmse_improvement_fraction": 0.01,
                "maximum_validation_mae_degradation_fraction": 0.01,
                "selected_model": "Persistence baseline",
            },
            "caveat": "Single year.",
        },
    )
    monkeypatch.setattr(prediction_api, "_cached_forecast", lambda *_args: grid)

    response = prediction_api.prediction_grid("tmax", date(2025, 12, 31))
    cell = prediction_api.prediction_cell(
        "tmax", date(2025, 12, 31), latitude=20.4, longitude=74.4
    )

    assert response["forecast_date"] == "2026-01-01"
    assert response["values"] == [[25.0, None], [27.0, 28.0]]
    assert cell["latitude"] == 20.5
    assert cell["longitude"] == 74.5
    assert cell["value"] == 28.0
    assert cell["run_type"] == "out_of_sample_projection"


def test_imd_download_parser_and_expected_annual_sizes():
    parser = _FormParser()
    parser.feed(
        '<form action="rainfall.php"><select name="rain">'
        '<option value="2024">2024</option><option value="2025">2025</option>'
        "</select></form>"
    )

    assert parser.form_action == "rainfall.php"
    assert parser.options == {2024, 2025}
    assert _expected_bytes(2024, 31 * 31) == 366 * 31 * 31 * 4
    assert _expected_bytes(2025, 129 * 135) == 365 * 129 * 135 * 4
