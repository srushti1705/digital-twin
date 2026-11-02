"""Open-Meteo supplementary live weather analysis adapter."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import urllib.request
import urllib.error
from typing import Any

from .base import BaseLiveAdapter
from ..schemas import LiveObservation, ProviderStatus, StationWeatherRecord


# 36 reference telemetry anchors covering all 28 Indian States and 8 Union Territories
INDIAN_STATIONS_CATALOG = [
    {"id": "IN-DL", "name": "New Delhi", "state": "Delhi", "state_code": "IN-DL", "lat": 28.6139, "lon": 77.2090},
    {"id": "IN-MH", "name": "Mumbai", "state": "Maharashtra", "state_code": "IN-MH", "lat": 19.0760, "lon": 72.8777},
    {"id": "IN-KA", "name": "Bengaluru", "state": "Karnataka", "state_code": "IN-KA", "lat": 12.9716, "lon": 77.5946},
    {"id": "IN-TN", "name": "Chennai", "state": "Tamil Nadu", "state_code": "IN-TN", "lat": 13.0827, "lon": 80.2707},
    {"id": "IN-WB", "name": "Kolkata", "state": "West Bengal", "state_code": "IN-WB", "lat": 22.5726, "lon": 88.3639},
    {"id": "IN-TG", "name": "Hyderabad", "state": "Telangana", "state_code": "IN-TG", "lat": 17.3850, "lon": 78.4867},
    {"id": "IN-GJ", "name": "Ahmedabad", "state": "Gujarat", "state_code": "IN-GJ", "lat": 23.0225, "lon": 72.5714},
    {"id": "IN-RJ", "name": "Jaipur", "state": "Rajasthan", "state_code": "IN-RJ", "lat": 26.9124, "lon": 75.7873},
    {"id": "IN-UP", "name": "Lucknow", "state": "Uttar Pradesh", "state_code": "IN-UP", "lat": 26.8467, "lon": 80.9462},
    {"id": "IN-BR", "name": "Patna", "state": "Bihar", "state_code": "IN-BR", "lat": 25.5941, "lon": 85.1376},
    {"id": "IN-MP", "name": "Bhopal", "state": "Madhya Pradesh", "state_code": "IN-MP", "lat": 23.2599, "lon": 77.4126},
    {"id": "IN-CH", "name": "Chandigarh", "state": "Chandigarh", "state_code": "IN-CH", "lat": 30.7333, "lon": 76.7794},
    {"id": "IN-KL", "name": "Thiruvananthapuram", "state": "Kerala", "state_code": "IN-KL", "lat": 8.5241, "lon": 76.9366},
    {"id": "IN-OR", "name": "Bhubaneswar", "state": "Odisha", "state_code": "IN-OR", "lat": 20.2961, "lon": 85.8245},
    {"id": "IN-JH", "name": "Ranchi", "state": "Jharkhand", "state_code": "IN-JH", "lat": 23.3441, "lon": 85.3096},
    {"id": "IN-CT", "name": "Raipur", "state": "Chhattisgarh", "state_code": "IN-CT", "lat": 21.2514, "lon": 81.6296},
    {"id": "IN-AS", "name": "Guwahati", "state": "Assam", "state_code": "IN-AS", "lat": 26.1445, "lon": 91.7362},
    {"id": "IN-JK", "name": "Srinagar", "state": "Jammu & Kashmir", "state_code": "IN-JK", "lat": 34.0837, "lon": 74.7973},
    {"id": "IN-HP", "name": "Shimla", "state": "Himachal Pradesh", "state_code": "IN-HP", "lat": 31.1048, "lon": 77.1734},
    {"id": "IN-UT", "name": "Dehradun", "state": "Uttarakhand", "state_code": "IN-UT", "lat": 30.3165, "lon": 78.0322},
    {"id": "IN-GA", "name": "Panaji", "state": "Goa", "state_code": "IN-GA", "lat": 15.4909, "lon": 73.8278},
    {"id": "IN-TR", "name": "Agartala", "state": "Tripura", "state_code": "IN-TR", "lat": 23.8315, "lon": 91.2868},
    {"id": "IN-ML", "name": "Shillong", "state": "Meghalaya", "state_code": "IN-ML", "lat": 25.5788, "lon": 91.8933},
    {"id": "IN-MN", "name": "Imphal", "state": "Manipur", "state_code": "IN-MN", "lat": 24.8170, "lon": 93.9368},
    {"id": "IN-MZ", "name": "Aizawl", "state": "Mizoram", "state_code": "IN-MZ", "lat": 23.7271, "lon": 92.7176},
    {"id": "IN-NL", "name": "Kohima", "state": "Nagaland", "state_code": "IN-NL", "lat": 25.6751, "lon": 94.1086},
    {"id": "IN-SK", "name": "Gangtok", "state": "Sikkim", "state_code": "IN-SK", "lat": 27.3389, "lon": 88.6065},
    {"id": "IN-AR", "name": "Itanagar", "state": "Arunachal Pradesh", "state_code": "IN-AR", "lat": 27.0844, "lon": 93.6053},
    {"id": "IN-AN", "name": "Port Blair", "state": "Andaman & Nicobar", "state_code": "IN-AN", "lat": 11.6234, "lon": 92.7265},
    {"id": "IN-LD", "name": "Kavaratti", "state": "Lakshadweep", "state_code": "IN-LD", "lat": 10.5667, "lon": 72.6417},
    {"id": "IN-PY", "name": "Puducherry", "state": "Puducherry", "state_code": "IN-PY", "lat": 11.9416, "lon": 79.8083},
    {"id": "IN-LA", "name": "Leh", "state": "Ladakh", "state_code": "IN-LA", "lat": 34.1526, "lon": 77.5771},
    {"id": "IN-DH", "name": "Daman", "state": "Dadra & Nagar Haveli and Daman & Diu", "state_code": "IN-DH", "lat": 20.4283, "lon": 72.8397},
    {"id": "IN-AP", "name": "Amaravati", "state": "Andhra Pradesh", "state_code": "IN-AP", "lat": 16.5062, "lon": 80.6480},
    {"id": "IN-PB", "name": "Amritsar", "state": "Punjab", "state_code": "IN-PB", "lat": 31.6340, "lon": 74.8723},
    {"id": "IN-HR", "name": "Gurugram", "state": "Haryana", "state_code": "IN-HR", "lat": 28.4595, "lon": 77.0266},
]

WMO_WEATHER_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class OpenMeteoAdapter(BaseLiveAdapter):
    """Supplementary adapter using Open-Meteo NWP model blending and reanalysis.

    Scientific Distinction:
    - Values are numerical model assimilation/analyses (ECMWF IFS, DWD ICON, GFS blends).
    - Labeled strictly as is_observation=False to prevent conflating model analyses with in-situ IMD ground stations.
    """

    def __init__(self) -> None:
        super().__init__(provider_id="open_meteo", name="Open-Meteo Climate Analysis (Supplementary)", cache_ttl_seconds=600)

    def get_status(self) -> ProviderStatus:
        now = datetime.now(timezone.utc)
        freshness: float | None = None
        if self._last_observation_utc:
            try:
                obs_dt = datetime.fromisoformat(self._last_observation_utc.replace("Z", "+00:00"))
                freshness = max(0.0, (now - obs_dt).total_seconds())
            except ValueError:
                freshness = None

        return ProviderStatus(
            provider_id=self.provider_id,
            name=self.name,
            description=(
                "Supplementary real-time weather assimilation service combining DWD ICON, ECMWF, and NOAA GFS "
                "analyses over the Indian subcontinent. Freely accessible without API keys."
            ),
            configured=True,
            available=self._last_error is None,
            data_status="fresh" if self.is_cache_valid() else ("stale" if self._cached_data else "unavailable"),
            requires_credentials=False,
            credential_env_vars=[],
            documentation_url="https://open-meteo.com/en/docs",
            last_fetched_utc=self._last_fetch_utc,
            last_observation_utc=self._last_observation_utc,
            freshness_seconds=freshness,
            error_message=self._last_error,
            attribution="Open-Meteo / ECMWF / DWD ICON (Supplementary Analysis, Not IMD Ground Stations)",
        )

    def fetch_weather(self) -> list[StationWeatherRecord]:
        """Fetch current weather across all 36 Indian state and territory reference locations."""
        if self.is_cache_valid() and self._cached_data:
            return self._cached_data

        lats = ",".join(str(s["lat"]) for s in INDIAN_STATIONS_CATALOG)
        lons = ",".join(str(s["lon"]) for s in INDIAN_STATIONS_CATALOG)
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lats}&longitude={lons}&"
            f"current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m,surface_pressure&"
            f"timezone=UTC"
        )

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "IndiaClimateDigitalTwin/1.0 (Climate Research; contact@digitaltwin.in)"},
        )

        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                if response.status != 200:
                    raise urllib.error.HTTPError(url, response.status, "Open-Meteo Error", response.headers, None)
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            self.record_error(f"Open-Meteo fetch failed: {err}")
            if self._cached_data:
                return self._cached_data
            return []

        now_utc = datetime.now(timezone.utc).isoformat()
        records: list[StationWeatherRecord] = []

        # Open-Meteo returns a list of result objects when multiple lat/lons are provided
        results = payload if isinstance(payload, list) else [payload]

        for idx, result in enumerate(results):
            if idx >= len(INDIAN_STATIONS_CATALOG):
                break
            station = INDIAN_STATIONS_CATALOG[idx]
            current = result.get("current", {})
            obs_time = current.get("time")
            if obs_time and not obs_time.endswith("Z"):
                obs_time = f"{obs_time}:00Z"
            elif not obs_time:
                obs_time = now_utc

            code = current.get("weather_code")
            cond = WMO_WEATHER_DESCRIPTIONS.get(code, "Weather condition unavailable") if code is not None else None

            rec = StationWeatherRecord(
                location_id=station["id"],
                location_name=station["name"],
                state=station["state"],
                state_code=station["state_code"],
                latitude=station["lat"],
                longitude=station["lon"],
                elevation_m=result.get("elevation"),
                source="open_meteo",
                source_product="Open-Meteo NWP Assimilation (DWD-ICON/ECMWF Blend)",
                is_observation=False,
                data_status="fresh",
                observation_time_utc=obs_time,
                fetched_at_utc=now_utc,
                temperature_c=current.get("temperature_2m"),
                apparent_temperature_c=current.get("apparent_temperature"),
                rainfall_mm=current.get("precipitation"),
                relative_humidity_percent=current.get("relative_humidity_2m"),
                wind_speed_kmh=current.get("wind_speed_10m"),
                surface_pressure_hpa=current.get("surface_pressure"),
                weather_condition=cond,
                weather_code=code,
            )
            records.append(rec)

        if records:
            latest_obs = max(r.observation_time_utc for r in records)
            self.set_cache(records, observation_time_utc=latest_obs)
        return records

