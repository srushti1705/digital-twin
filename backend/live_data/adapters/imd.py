"""Official India Meteorological Department (IMD) API Adapter."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
import urllib.request
import urllib.error
import json

from .base import BaseLiveAdapter
from ..schemas import LiveObservation, ProviderStatus, StationWeatherRecord


class ImdLiveAdapter(BaseLiveAdapter):
    """Adapter for the official IMD API management portal (api.imd.gov.in).

    Requirements & Authentication:
    - Documentation: https://api.imd.gov.in/public/api_reference.html
    - Registration requires an institutional or developer account on api.imd.gov.in.
    - Endpoints require an API token bound to the server's public IP address.
    - Configured via environment variables: IMD_API_KEY and IMD_API_BASE_URL.
    """

    def __init__(self) -> None:
        super().__init__(provider_id="imd", name="India Meteorological Department (Official API)", cache_ttl_seconds=900)
        self.base_url = os.environ.get("IMD_API_BASE_URL", "https://api.imd.gov.in/api").rstrip("/")
        self.api_key = os.environ.get("IMD_API_KEY")

    def get_status(self) -> ProviderStatus:
        configured = bool(self.api_key and self.api_key.strip())
        now = datetime.now(timezone.utc)
        freshness: float | None = None
        if self._last_observation_utc:
            try:
                obs_dt = datetime.fromisoformat(self._last_observation_utc.replace("Z", "+00:00"))
                freshness = max(0.0, (now - obs_dt).total_seconds())
            except ValueError:
                freshness = None

        data_status = "credentials_required"
        if configured:
            data_status = "fresh" if self.is_cache_valid() else ("stale" if self._cached_data else "unavailable")

        return ProviderStatus(
            provider_id=self.provider_id,
            name=self.name,
            description=(
                "Official Indian national meteorological observations including AWS/ARG stations, "
                "nowcasts, and district rainfall bulletins from IMD Pune / New Delhi."
            ),
            configured=configured,
            available=configured and self._last_error is None,
            data_status=data_status,
            requires_credentials=True,
            credential_env_vars=["IMD_API_KEY", "IMD_API_BASE_URL"],
            documentation_url="https://api.imd.gov.in/public/api_reference.html",
            last_fetched_utc=self._last_fetch_utc,
            last_observation_utc=self._last_observation_utc,
            freshness_seconds=freshness,
            error_message=self._last_error if configured else "IMD API Key not configured. Register at api.imd.gov.in.",
            attribution="India Meteorological Department (IMD), Ministry of Earth Sciences",
        )

    def fetch_current_weather(self) -> list[StationWeatherRecord]:
        """Fetch current weather observations from IMD station network if credentials are configured."""
        if not self.api_key:
            return []

        if self.is_cache_valid() and self._cached_data:
            return self._cached_data

        url = f"{self.base_url}/current_weather"
        headers = {
            "User-Agent": "IndiaClimateTwin/1.0",
            "X-API-KEY": self.api_key,
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    raise urllib.error.HTTPError(url, response.status, "IMD API Error", response.headers, None)
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            self.record_error(f"IMD API request failed: {err}")
            # Stale cache retention
            if self._cached_data:
                return self._cached_data
            return []

        # Parse IMD response structure
        now_utc = datetime.now(timezone.utc).isoformat()
        records: list[StationWeatherRecord] = []
        station_list = data if isinstance(data, list) else data.get("stations", data.get("data", []))

        for item in station_list:
            try:
                obs_time = item.get("observation_time", now_utc)
                rec = StationWeatherRecord(
                    location_id=str(item.get("station_id", item.get("id", ""))),
                    location_name=str(item.get("station_name", item.get("city", "Unknown Station"))),
                    state=str(item.get("state", "")),
                    state_code=str(item.get("state_code", "")),
                    latitude=float(item["latitude"]),
                    longitude=float(item["longitude"]),
                    elevation_m=float(item["elevation"]) if "elevation" in item else None,
                    source="imd",
                    source_product="IMD AWS/ARG Surface Observation",
                    is_observation=True,
                    data_status="fresh",
                    observation_time_utc=obs_time,
                    fetched_at_utc=now_utc,
                    temperature_c=float(item["temperature"]) if item.get("temperature") is not None else None,
                    apparent_temperature_c=float(item["apparent_temperature"]) if item.get("apparent_temperature") is not None else None,
                    rainfall_mm=float(item["rainfall"]) if item.get("rainfall") is not None else None,
                    relative_humidity_percent=float(item["relative_humidity"]) if item.get("relative_humidity") is not None else None,
                    wind_speed_kmh=float(item["wind_speed"]) if item.get("wind_speed") is not None else None,
                    surface_pressure_hpa=float(item["pressure"]) if item.get("pressure") is not None else None,
                    weather_condition=item.get("weather_condition"),
                    weather_code=item.get("weather_code"),
                )
                records.append(rec)
            except (KeyError, ValueError, TypeError):
                continue

        if records:
            latest_obs = max(r.observation_time_utc for r in records)
            self.set_cache(records, observation_time_utc=latest_obs)
        return records

