"""Base adapter interface for live climate data providers."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from ..schemas import ProviderStatus


class BaseLiveAdapter(ABC):
    """Abstract base class for all external climate data providers."""

    def __init__(self, provider_id: str, name: str, cache_ttl_seconds: int = 600) -> None:
        self.provider_id = provider_id
        self.name = name
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache_lock = threading.Lock()
        self._cache_timestamp: float = 0.0
        self._cached_data: Any | None = None
        self._last_fetch_utc: str | None = None
        self._last_observation_utc: str | None = None
        self._last_error: str | None = None

    @abstractmethod
    def get_status(self) -> ProviderStatus:
        """Return the current configuration, authentication, and availability status."""
        pass

    def is_cache_valid(self) -> bool:
        """Check if cached data is still within its Time-To-Live."""
        return (
            self._cached_data is not None
            and (time.monotonic() - self._cache_timestamp) < self.cache_ttl_seconds
        )

    def get_cached(self) -> Any | None:
        """Return cached payload if present (can be used for stale-while-revalidate)."""
        with self._cache_lock:
            return self._cached_data

    def set_cache(self, data: Any, observation_time_utc: str | None = None) -> None:
        """Update cached payload and freshness timestamps."""
        with self._cache_lock:
            self._cached_data = data
            self._cache_timestamp = time.monotonic()
            from datetime import datetime, timezone
            self._last_fetch_utc = datetime.now(timezone.utc).isoformat()
            if observation_time_utc:
                self._last_observation_utc = observation_time_utc
            self._last_error = None

    def record_error(self, error: str) -> None:
        """Record an upstream fetch error while preserving any existing cache."""
        with self._cache_lock:
            self._last_error = error

