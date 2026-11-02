"""FastAPI router for live and near-real-time climate services."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException

if __package__:
    from .live_data import (
        LiveCoverageInfo,
        LiveObservation,
        ProviderStatus,
        SatelliteLayerInfo,
        StationWeatherRecord,
        live_service,
    )
else:
    from live_data import (
        LiveCoverageInfo,
        LiveObservation,
        ProviderStatus,
        SatelliteLayerInfo,
        StationWeatherRecord,
        live_service,
    )

router = APIRouter(prefix="/api/live", tags=["live"])


@router.get("/status", response_model=list[ProviderStatus])
def get_live_status() -> list[ProviderStatus]:
    """Return live status, credentials configuration, and freshness for all providers."""
    return live_service.get_provider_statuses()


@router.get("/weather", response_model=list[StationWeatherRecord])
def get_live_weather(
    state_code: str | None = None,
    search: str | None = None,
) -> list[StationWeatherRecord]:
    """Return normalized latest weather observations across supported Indian stations."""
    return live_service.get_weather(state_code=state_code, search_query=search)


@router.get("/rainfall", response_model=list[LiveObservation])
def get_live_rainfall() -> list[LiveObservation]:
    """Return available live or near-real-time rainfall observations across India."""
    return live_service.get_rainfall_observations()


@router.get("/satellite/layers", response_model=list[SatelliteLayerInfo])
def get_satellite_layers() -> list[SatelliteLayerInfo]:
    """Return available NASA GIBS satellite products and supported date extents."""
    return live_service.get_satellite_layers()


@router.get("/satellite/imagery")
def get_satellite_imagery(
    layer_id: str,
    observation_date: str | None = None,
) -> dict[str, Any]:
    """Return WMS configuration and metadata for rendering the satellite overlay in CesiumJS."""
    try:
        return live_service.get_satellite_imagery_metadata(layer_id, observation_date)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@router.get("/coverage", response_model=LiveCoverageInfo)
def get_live_coverage() -> LiveCoverageInfo:
    """Return geographical coverage, active providers, and freshness information."""
    return live_service.get_coverage()
