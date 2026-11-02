"""Thread-safe in-memory LRU store for scenario runs."""

from __future__ import annotations

import collections
import threading
from typing import Any

from .schemas import ScenarioResult


class ScenarioStore:
    """Thread-safe scenario cache retaining up to `capacity` executed scenarios."""

    def __init__(self, capacity: int = 50) -> None:
        self.capacity = capacity
        self._lock = threading.Lock()
        self._store: collections.OrderedDict[str, ScenarioResult] = collections.OrderedDict()

    def put(self, result: ScenarioResult) -> None:
        with self._lock:
            if result.scenario_id in self._store:
                self._store.move_to_end(result.scenario_id)
            self._store[result.scenario_id] = result
            if len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def get(self, scenario_id: str) -> ScenarioResult | None:
        with self._lock:
            res = self._store.get(scenario_id)
            if res:
                self._store.move_to_end(scenario_id)
            return res

    def list_all(self) -> list[ScenarioResult]:
        with self._lock:
            return list(self._store.values())


scenario_store = ScenarioStore()

