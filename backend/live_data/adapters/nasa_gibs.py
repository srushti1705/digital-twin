"""NASA Global Imagery Browse Services (GIBS) Adapter."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from typing import Any

from .base import BaseLiveAdapter
from ..schemas import ProviderStatus, SatelliteLayerInfo


NASA_GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
NASA_GIBS_CAPABILITIES_URL = (
    f"{NASA_GIBS_WMS_URL}?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.1.1"
)

SUPPORTED_GIBS_LAYERS: list[SatelliteLayerInfo] = [
    SatelliteLayerInfo(
        layer_id="true_color_terra",
        name="NASA MODIS Terra (True Color)",
        description="Daily natural true-color corrected reflectance satellite imagery from EOS Terra (250m resolution).",
        category="true_color",
        service_type="wms",
        service_url=NASA_GIBS_WMS_URL,
        layer_name="MODIS_Terra_CorrectedReflectance_TrueColor",
        format="image/jpeg",
        time_support=True,
        default_date=(datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat(),
        date_start="2000-02-24",
        date_end=datetime.now(timezone.utc).date().isoformat(),
        attribution="NASA Global Imagery Browse Services (GIBS) / EOSDIS Terra MODIS",
        opacity_default=0.75,
    ),
    SatelliteLayerInfo(
        layer_id="cloud_top_temp",
        name="NASA MODIS Aqua (Cloud Top Temperature)",
        description="Daily daytime thermal cloud top temperature indicating deep convective cloud formations and thunderstorm tops.",
        category="clouds",
        service_type="wms",
        service_url=NASA_GIBS_WMS_URL,
        layer_name="MODIS_Aqua_Cloud_Top_Temp_Day",
        format="image/png",
        time_support=True,
        default_date=(datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat(),
        date_start="2002-07-04",
        date_end=datetime.now(timezone.utc).date().isoformat(),
        attribution="NASA GIBS / Atmosphere Science Team / Aqua MODIS",
        opacity_default=0.70,
    ),
    SatelliteLayerInfo(
        layer_id="imerg_precipitation_rate",
        name="NASA GPM IMERG (Precipitation Rate Imagery)",
        description="Multi-satellite precipitation rate imagery calibrated with gauge data across the globe (0.1° resolution).",
        category="precipitation",
        service_type="wms",
        service_url=NASA_GIBS_WMS_URL,
        layer_name="IMERG_Precipitation_Rate",
        format="image/png",
        time_support=True,
        default_date=(datetime.now(timezone.utc).date() - timedelta(days=2)).isoformat(),
        date_start="2000-06-01",
        date_end=(datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat(),
        attribution="NASA GIBS / GPM IMERG Precipitation Processing System",
        legend_url="https://gibs.earthdata.nasa.gov/legends/IMERG_Precipitation_Rate_V06.png",
        opacity_default=0.82,
    ),
    SatelliteLayerInfo(
        layer_id="viirs_snpp_true_color",
        name="NASA VIIRS SNPP (Corrected Reflectance)",
        description="High-resolution imagery from the Suomi NPP satellite showing high detail clouds, storms, and atmospheric features.",
        category="true_color",
        service_type="wms",
        service_url=NASA_GIBS_WMS_URL,
        layer_name="VIIRS_SNPP_CorrectedReflectance_TrueColor",
        format="image/jpeg",
        time_support=True,
        default_date=(datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat(),
        date_start="2012-01-20",
        date_end=datetime.now(timezone.utc).date().isoformat(),
        attribution="NASA GIBS / Suomi NPP VIIRS",
        opacity_default=0.75,
    ),
]


class NasaGibsAdapter(BaseLiveAdapter):
    """Adapter for NASA Global Imagery Browse Services (GIBS)."""

    def __init__(self) -> None:
        super().__init__(provider_id="nasa_gibs", name="NASA GIBS Satellite Imagery", cache_ttl_seconds=3600)

    def get_status(self) -> ProviderStatus:
        now = datetime.now(timezone.utc)
        latest_date = (now.date() - timedelta(days=1)).isoformat()
        return ProviderStatus(
            provider_id=self.provider_id,
            name=self.name,
            description="NASA Global Imagery Browse Services delivering live/near-real-time full-resolution satellite imagery layers via WMS/WMTS.",
            configured=True,
            available=True,
            data_status="fresh",
            requires_credentials=False,
            credential_env_vars=[],
            documentation_url="https://nasa-gibs.github.io/gibs-api-docs/",
            last_fetched_utc=self._last_fetch_utc or now.isoformat(),
            last_observation_utc=latest_date,
            freshness_seconds=86400.0,
            error_message=self._last_error,
            attribution="NASA EOSDIS Global Imagery Browse Services (GIBS)",
        )

    def get_layers(self) -> list[SatelliteLayerInfo]:
        """Return available satellite imagery overlay products."""
        return SUPPORTED_GIBS_LAYERS

    def get_layer_metadata(self, layer_id: str, observation_date: str | None = None) -> dict[str, Any]:
        """Return WMS parameters and valid time extents for CesiumJS layer rendering."""
        layer = next((l for l in SUPPORTED_GIBS_LAYERS if l.layer_id == layer_id), None)
        if layer is None:
            raise ValueError(f"Unknown satellite layer ID: {layer_id}")

        selected_date = observation_date if isinstance(observation_date, str) and observation_date.strip() else layer.default_date
        # Validate date format YYYY-MM-DD
        try:
            date.fromisoformat(selected_date)
        except ValueError:
            selected_date = layer.default_date

        return {
            "layer_id": layer.layer_id,
            "layer_name": layer.layer_name,
            "service_url": layer.service_url,
            "service_type": layer.service_type,
            "format": layer.format,
            "time": selected_date,
            "opacity": layer.opacity_default,
            "attribution": layer.attribution,
            "legend_url": layer.legend_url,
            "wms_parameters": {
                "service": "WMS",
                "version": "1.1.1",
                "request": "GetMap",
                "layers": layer.layer_name,
                "format": layer.format,
                "transparent": "true",
                "time": selected_date,
            },
        }
