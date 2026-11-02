"""Pydantic schemas and types for live climate data integration."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class LiveObservation(BaseModel):
    """Normalized observation schema matching the unified climate data specification."""

    variable: str = Field(description="Climate variable name, e.g., 'rainfall', 'temperature', 'humidity'")
    value: float | None = Field(description="Numerical value, or null if missing/unavailable")
    units: str = Field(description="Measurement units, e.g., 'mm', '°C', '%', 'km/h', 'hPa'")
    latitude: float = Field(ge=-90.0, le=90.0, description="WGS84 latitude")
    longitude: float = Field(ge=-180.0, le=180.0, description="WGS84 longitude")
    location_name: str = Field(description="Name of the city, station, or territory")
    state: str | None = Field(default=None, description="Indian State or Union Territory name")
    state_code: str | None = Field(default=None, description="ISO/standard state code, e.g. 'IN-MH'")
    source: str = Field(description="Originating provider ID: 'imd', 'open_meteo', 'nasa_gpm', etc.")
    source_product: str = Field(description="Exact product name, e.g. 'IMD AWS Station Observation'")
    observation_time_utc: str = Field(description="ISO-8601 UTC timestamp of the measurement")
    fetched_at_utc: str = Field(description="ISO-8601 UTC timestamp when our backend retrieved the data")
    processing_time_utc: str | None = Field(default=None, description="Provider publication/processing timestamp if supplied")
    data_status: Literal["fresh", "stale", "cached", "credentials_required", "unavailable"] = Field(
        default="fresh", description="Freshness status of this observation"
    )
    quality_flag: str | None = Field(default=None, description="Provider quality control indicator")
    is_observation: bool = Field(description="True if measured by in-situ station/radar; False if model-derived analysis")
    is_forecast: bool = Field(default=False, description="True if a predicted value")
    spatial_resolution: str | None = Field(default=None, description="Spatial resolution, e.g. 'point', '0.25°'")
    temporal_resolution: str | None = Field(default=None, description="Temporal resolution, e.g. 'hourly', 'daily'")


class StationWeatherRecord(BaseModel):
    """Aggregated current weather record for a single Indian geographical location."""

    location_id: str
    location_name: str
    state: str
    state_code: str
    latitude: float
    longitude: float
    elevation_m: float | None = None
    source: str
    source_product: str
    is_observation: bool
    data_status: Literal["fresh", "stale", "cached", "credentials_required", "unavailable"]
    observation_time_utc: str
    fetched_at_utc: str
    temperature_c: float | None = None
    apparent_temperature_c: float | None = None
    rainfall_mm: float | None = None
    relative_humidity_percent: float | None = None
    wind_speed_kmh: float | None = None
    surface_pressure_hpa: float | None = None
    weather_condition: str | None = None
    weather_code: int | None = None


class ProviderStatus(BaseModel):
    """Health, configuration, and data freshness status of an external climate provider."""

    provider_id: str
    name: str
    description: str
    configured: bool
    available: bool
    data_status: Literal["fresh", "stale", "cached", "credentials_required", "unavailable"]
    requires_credentials: bool
    credential_env_vars: list[str]
    documentation_url: str
    last_fetched_utc: str | None = None
    last_observation_utc: str | None = None
    freshness_seconds: float | None = None
    error_message: str | None = None
    attribution: str


class SatelliteLayerInfo(BaseModel):
    """Metadata for an Earth-observation satellite imagery overlay product."""

    layer_id: str
    name: str
    description: str
    category: Literal["true_color", "clouds", "precipitation", "thermal"]
    service_type: Literal["wms", "wmts"]
    service_url: str
    layer_name: str
    format: str = "image/png"
    time_support: bool = True
    default_date: str
    date_start: str | None = None
    date_end: str | None = None
    attribution: str
    legend_url: str | None = None
    opacity_default: float = 0.75


class LiveCoverageInfo(BaseModel):
    """Geographical and temporal coverage information for live data."""

    coverage_extent: dict[str, float] = Field(
        default_factory=lambda: {
            "min_latitude": 6.5,
            "max_latitude": 38.5,
            "min_longitude": 66.5,
            "max_longitude": 100.0,
        }
    )
    supported_locations_count: int
    active_providers_count: int
    latest_update_utc: str | None
    providers: list[ProviderStatus]

