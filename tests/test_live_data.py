"""Unit and integration tests for live climate data adapters and API endpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from backend.live_data.adapters.imd import ImdLiveAdapter
from backend.live_data.adapters.nasa_gibs import NasaGibsAdapter
from backend.live_data.adapters.nasa_imerg import NasaImergAdapter
from backend.live_data.adapters.open_meteo import OpenMeteoAdapter
from backend.live_data.service import LiveDataService
from backend.live_api import (
    get_live_coverage,
    get_live_rainfall,
    get_live_status,
    get_live_weather,
    get_satellite_imagery,
    get_satellite_layers,
)


def test_imd_adapter_missing_credentials():
    """Verify that when IMD_API_KEY is not set, the adapter reports credentials_required."""
    with patch.dict("os.environ", {}, clear=True):
        adapter = ImdLiveAdapter()
        status = adapter.get_status()
        assert status.configured is False
        assert status.data_status == "credentials_required"
        assert status.requires_credentials is True
        assert "IMD_API_KEY" in status.credential_env_vars
        assert adapter.fetch_current_weather() == []


def test_imd_adapter_with_credentials_and_mock_response():
    """Verify IMD adapter parses station observations when valid JSON is returned."""
    mock_payload = {
        "stations": [
            {
                "station_id": "IMD_43003",
                "station_name": "Mumbai Colaba",
                "state": "Maharashtra",
                "state_code": "IN-MH",
                "latitude": 18.90,
                "longitude": 72.81,
                "temperature": 31.2,
                "rainfall": 12.5,
                "relative_humidity": 78.0,
                "observation_time": "2026-11-02T12:00:00Z",
            }
        ]
    }
    with patch.dict("os.environ", {"IMD_API_KEY": "test_token_123"}):
        adapter = ImdLiveAdapter()
        status = adapter.get_status()
        assert status.configured is True

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            records = adapter.fetch_current_weather()
            assert len(records) == 1
            rec = records[0]
            assert rec.location_name == "Mumbai Colaba"
            assert rec.temperature_c == 31.2
            assert rec.rainfall_mm == 12.5
            assert rec.is_observation is True
            assert rec.source == "imd"


def test_nasa_gibs_adapter_layers_and_metadata():
    """Verify NASA GIBS satellite products catalog and metadata formatting."""
    adapter = NasaGibsAdapter()
    status = adapter.get_status()
    assert status.configured is True
    assert status.available is True

    layers = adapter.get_layers()
    assert len(layers) >= 3
    layer_ids = [l.layer_id for l in layers]
    assert "true_color_terra" in layer_ids
    assert "cloud_top_temp" in layer_ids
    assert "imerg_precipitation_rate" in layer_ids

    meta = adapter.get_layer_metadata("true_color_terra", "2026-10-31")
    assert meta["layer_name"] == "MODIS_Terra_CorrectedReflectance_TrueColor"
    assert meta["time"] == "2026-10-31"
    assert meta["wms_parameters"]["transparent"] == "true"


def test_nasa_imerg_adapter_credentials_status():
    """Verify NASA GPM IMERG handles missing vs present Earthdata credentials."""
    with patch.dict("os.environ", {}, clear=True):
        adapter = NasaImergAdapter()
        status = adapter.get_status()
        assert status.configured is False
        assert status.data_status == "credentials_required"
        summary = adapter.fetch_latest_imerg_summary()
        assert summary["available"] is False
        assert summary["reason"] == "credentials_required"

    with patch.dict("os.environ", {"EARTHDATA_USERNAME": "test_user", "EARTHDATA_PASSWORD": "secret_pwd"}):
        adapter_auth = NasaImergAdapter()
        status_auth = adapter_auth.get_status()
        assert status_auth.configured is True
        summary_auth = adapter_auth.fetch_latest_imerg_summary()
        assert summary_auth["available"] is True


def test_open_meteo_adapter_parsing_and_null_preservation():
    """Verify Open-Meteo adapter preserves nulls and does not substitute zero."""
    mock_response_data = [
        {
            "latitude": 28.6139,
            "longitude": 77.2090,
            "elevation": 216.0,
            "current": {
                "time": "2026-11-02T12:00",
                "temperature_2m": 29.4,
                "relative_humidity_2m": 52.0,
                "apparent_temperature": 30.1,
                "precipitation": None,  # Preserved as null, NOT 0.0
                "rain": None,
                "weather_code": 1,
                "wind_speed_10m": 8.5,
                "surface_pressure": 1012.3,
            },
        }
    ]

    adapter = OpenMeteoAdapter()
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        records = adapter.fetch_weather()
        assert len(records) >= 1
        delhi = records[0]
        assert delhi.location_name == "New Delhi"
        assert delhi.temperature_c == 29.4
        assert delhi.rainfall_mm is None  # Must remain None, not converted to 0
        assert delhi.is_observation is False  # Must be strictly model assimilation
        assert delhi.source == "open_meteo"


def test_open_meteo_adapter_stale_cache_on_upstream_failure():
    """Verify that when upstream fails, existing cached data is retained and returned."""
    adapter = OpenMeteoAdapter()
    # Pre-populate cache and artificially expire timestamp so fetch is attempted
    adapter.set_cache([MagicMock(location_name="Cached Station")], observation_time_utc="2026-11-02T10:00:00Z")
    adapter._cache_timestamp = 0.0  # Force is_cache_valid() to False

    with patch("urllib.request.urlopen", side_effect=URLError("Connection timed out")):
        records = adapter.fetch_weather()
        assert len(records) == 1
        assert records[0].location_name == "Cached Station"
        assert adapter._last_error is not None


def test_live_data_service_filtering():
    """Verify state code and search query filtering in LiveDataService."""
    service = LiveDataService()
    dummy_records = [
        MagicMock(location_name="Mumbai", state="Maharashtra", state_code="IN-MH", rainfall_mm=2.5, latitude=19.0, longitude=72.8, source="test", source_product="p", observation_time_utc="2026-11-02T12:00:00Z", fetched_at_utc="2026-11-02T12:00:00Z", data_status="fresh", is_observation=True),
        MagicMock(location_name="Bengaluru", state="Karnataka", state_code="IN-KA", rainfall_mm=0.0, latitude=12.9, longitude=77.5, source="test", source_product="p", observation_time_utc="2026-11-02T12:00:00Z", fetched_at_utc="2026-11-02T12:00:00Z", data_status="fresh", is_observation=True),
    ]

    with patch.object(service.open_meteo_adapter, "fetch_weather", return_value=dummy_records), \
         patch.object(service.imd_adapter, "fetch_current_weather", return_value=[]):
        filtered_mh = service.get_weather(state_code="IN-MH")
        assert len(filtered_mh) == 1
        assert filtered_mh[0].location_name == "Mumbai"

        search_res = service.get_weather(search_query="beng")
        assert len(search_res) == 1
        assert search_res[0].location_name == "Bengaluru"


def test_live_api_endpoint_handlers():
    """Directly test all FastAPI live route handlers."""
    # 1. get_live_status
    providers = get_live_status()
    assert isinstance(providers, list)
    assert any(p.provider_id == "imd" for p in providers)
    assert any(p.provider_id == "open_meteo" for p in providers)

    # 2. get_satellite_layers
    layers = get_satellite_layers()
    assert len(layers) >= 3

    # 3. get_satellite_imagery
    img = get_satellite_imagery(layer_id="true_color_terra")
    assert "wms_parameters" in img

    # 4. get_live_coverage
    cov = get_live_coverage()
    assert cov.coverage_extent is not None
    assert cov.supported_locations_count >= 0
