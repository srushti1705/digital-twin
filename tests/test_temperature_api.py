from datetime import date

import numpy as np

from backend import temperature_api


def test_temperature_grid_dates_and_cells_exclude_float32_missing_values(
    tmp_path, monkeypatch
):
    day_count = 366
    annual_grid = np.full((day_count, 31, 31), 99.9, dtype="<f4")
    annual_grid[0, 12, 6] = 28.5
    product_directory = tmp_path / "max_temp"
    product_directory.mkdir()
    annual_grid.tofile(product_directory / "maxtemp_2024.grd")
    monkeypatch.setenv("IMD_TEMPERATURE_DIR", str(tmp_path))

    dates = temperature_api._dates_for_product("tmax")
    metadata, grid = temperature_api._load_temperature_grid("tmax", date(2024, 1, 1))
    available_cell = temperature_api.temperature_cell(
        "tmax", date(2024, 1, 1), latitude=19.5, longitude=73.5
    )
    missing_cell = temperature_api.temperature_cell(
        "tmax", date(2024, 1, 1), latitude=18.5, longitude=73.5
    )

    assert dates == ["2024-01-01"]
    assert metadata["valid_cell_count"] == 1
    assert metadata["missing_cell_count"] == 31 * 31 - 1
    assert grid[12, 6] == 28.5
    assert np.isnan(grid[11, 6])
    assert available_cell["temperature"] == 28.5
    assert available_cell["missing"] is False
    assert missing_cell["temperature"] is None
    assert missing_cell["missing"] is True
