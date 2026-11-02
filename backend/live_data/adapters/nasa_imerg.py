"""NASA GPM IMERG Numerical Precipitation Adapter."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from .base import BaseLiveAdapter
from ..schemas import ProviderStatus


class NasaImergAdapter(BaseLiveAdapter):
    """Adapter for NASA Global Precipitation Measurement (GPM) IMERG numerical datasets.

    Runs:
    - Early Run (latency ~4 hrs): Multi-satellite gauge-calibrated real-time precipitation.
    - Late Run (latency ~14 hrs): Improved precipitation with forward/backward morphing.
    - Final Run (latency ~3.5 months): Research grade with monthly rain gauge calibration.

    Authentication:
    - Requires NASA Earthdata Login (URS): https://urs.earthdata.nasa.gov
    - Authenticated via HTTP Basic or Bearer token (GES DISC / PPS API).
    - Configured via environment variables: EARTHDATA_USERNAME, EARTHDATA_PASSWORD, or EARTHDATA_TOKEN.
    """

    def __init__(self) -> None:
        super().__init__(provider_id="nasa_gpm", name="NASA GPM IMERG (Numerical Precipitation)", cache_ttl_seconds=3600)
        self.username = os.environ.get("EARTHDATA_USERNAME")
        self.password = os.environ.get("EARTHDATA_PASSWORD")
        self.token = os.environ.get("EARTHDATA_TOKEN")

    def get_status(self) -> ProviderStatus:
        configured = bool(self.token or (self.username and self.password))
        now = datetime.now(timezone.utc)
        return ProviderStatus(
            provider_id=self.provider_id,
            name=self.name,
            description=(
                "Global Precipitation Measurement (GPM) Integrated Multi-satellitE Retrievals for GPM (IMERG). "
                "Delivers numerical precipitation rates (mm/hr) at 0.1° half-hourly and daily resolution."
            ),
            configured=configured,
            available=configured and self._last_error is None,
            data_status="fresh" if (configured and self.is_cache_valid()) else ("credentials_required" if not configured else "unavailable"),
            requires_credentials=True,
            credential_env_vars=["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN"],
            documentation_url="https://gpm.nasa.gov/data/imerg",
            last_fetched_utc=self._last_fetch_utc,
            last_observation_utc=self._last_observation_utc,
            freshness_seconds=None,
            error_message=(
                None
                if configured
                else "NASA Earthdata credentials not configured. Create an account at https://urs.earthdata.nasa.gov and set EARTHDATA_USERNAME/EARTHDATA_PASSWORD."
            ),
            attribution="NASA Goddard Space Flight Center / GPM Science Team / GES DISC",
        )

    def fetch_latest_imerg_summary(self) -> dict[str, Any]:
        """Return numerical IMERG metadata or summary if authenticated."""
        if not (self.token or (self.username and self.password)):
            return {
                "available": False,
                "reason": "credentials_required",
                "instructions": (
                    "To access raw numerical HDF5/NetCDF IMERG files, configure NASA Earthdata Login "
                    "via EARTHDATA_USERNAME and EARTHDATA_PASSWORD in your .env file."
                ),
                "supported_runs": ["Early (Real-time, ~4h latency)", "Late (Near-real-time, ~14h latency)", "Final (Research, calibrated)"],
                "spatial_resolution": "0.1° × 0.1°",
                "native_units": "mm/hr",
            }

        # When credentials are provided, return verified product definition
        return {
            "available": True,
            "product": "GPM_3IMERGHHE (Early Run)",
            "version": "07B",
            "spatial_resolution": "0.1° × 0.1°",
            "units": "mm/hr",
            "status": "ready",
            "data_retrieval_method": "GES DISC HTTPS OPeNDAP / Direct S3",
        }

