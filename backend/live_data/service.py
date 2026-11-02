"""Central Live Data Service orchestrating IMD, NASA, and Open-Meteo adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .adapters.imd import ImdLiveAdapter
from .adapters.nasa_gibs import NasaGibsAdapter
from .adapters.nasa_imerg import NasaImergAdapter
from .adapters.open_meteo import OpenMeteoAdapter
from .schemas import (
    LiveCoverageInfo,
    LiveObservation,
    ProviderStatus,
    SatelliteLayerInfo,
    StationWeatherRecord,
)


class LiveDataService:
    """Coordinates climate data ingestion, caching, and multi-source fallbacks."""

    def __init__(self) -> None:
        self.imd_adapter = ImdLiveAdapter()
        self.nasa_gibs_adapter = NasaGibsAdapter()
        self.nasa_imerg_adapter = NasaImergAdapter()
        self.open_meteo_adapter = OpenMeteoAdapter()

    def get_provider_statuses(self) -> list[ProviderStatus]:
        """Return the operational and credential status for each climate provider."""
        return [
            self.imd_adapter.get_status(),
            self.nasa_gibs_adapter.get_status(),
            self.nasa_imerg_adapter.get_status(),
            self.open_meteo_adapter.get_status(),
        ]

    def get_weather(
        self,
        state_code: str | None = None,
        search_query: str | None = None,
    ) -> list[StationWeatherRecord]:
        """Fetch current weather across Indian stations.

        Precedence:
        - If official IMD API credentials are provided and return valid station data, IMD is prioritized.
        - Supplementary Open-Meteo NWP assimilation is provided as standard continuous coverage.
        """
        records: list[StationWeatherRecord] = []

        # 1. Check official IMD
        imd_records = self.imd_adapter.fetch_current_weather()
        if imd_records:
            records.extend(imd_records)

        # 2. Add supplementary Open-Meteo records if IMD did not supply records or to provide full nationwide coverage
        if not records:
            records.extend(self.open_meteo_adapter.fetch_weather())

        # Filtering
        if state_code:
            code_upper = state_code.strip().upper()
            records = [r for r in records if r.state_code.upper() == code_upper]

        if search_query:
            q = search_query.strip().lower()
            records = [
                r for r in records
                if q in r.location_name.lower() or q in r.state.lower() or q in r.state_code.lower()
            ]

        return records

    def get_rainfall_observations(self) -> list[LiveObservation]:
        """Return normalized live/near-real-time rainfall values."""
        weather_records = self.get_weather()
        observations: list[LiveObservation] = []

        for w in weather_records:
            observations.append(
                LiveObservation(
                    variable="rainfall",
                    value=w.rainfall_mm,
                    units="mm",
                    latitude=w.latitude,
                    longitude=w.longitude,
                    location_name=w.location_name,
                    state=w.state,
                    state_code=w.state_code,
                    source=w.source,
                    source_product=w.source_product,
                    observation_time_utc=w.observation_time_utc,
                    fetched_at_utc=w.fetched_at_utc,
                    data_status=w.data_status,
                    is_observation=w.is_observation,
                    is_forecast=False,
                    spatial_resolution="point",
                    temporal_resolution="hourly",
                )
            )
        return observations

    def get_satellite_layers(self) -> list[SatelliteLayerInfo]:
        """Return list of supported NASA GIBS satellite products."""
        return self.nasa_gibs_adapter.get_layers()

    def get_satellite_imagery_metadata(self, layer_id: str, observation_date: str | None = None) -> dict[str, Any]:
        """Return WMS connection parameters for a satellite layer."""
        return self.nasa_gibs_adapter.get_layer_metadata(layer_id, observation_date)

    def get_coverage(self) -> LiveCoverageInfo:
        """Return summary of all-India coverage and active providers."""
        statuses = self.get_provider_statuses()
        active_count = sum(1 for s in statuses if s.available)
        weather_list = self.get_weather()

        now_utc = datetime.now(timezone.utc).isoformat()
        latest_update = max(
            (s.last_fetched_utc for s in statuses if s.last_fetched_utc),
            default=now_utc,
        )

        return LiveCoverageInfo(
            supported_locations_count=len(weather_list),
            active_providers_count=active_count,
            latest_update_utc=latest_update,
            providers=statuses,
        )


# Singleton instance
live_service = LiveDataService()

