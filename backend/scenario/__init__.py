"""What-If scenario analysis engine package."""

from .engine import ScenarioEngine
from .schemas import (
    CompactGridData,
    GridSummaryMetrics,
    ScenarioCompareRequest,
    ScenarioCompareResult,
    ScenarioResult,
    ScenarioRunRequest,
    ScenarioVariableSummary,
)
from .spatial import get_spatial_mask, get_supported_states
from .storage import scenario_store

__all__ = [
    "ScenarioEngine",
    "ScenarioRunRequest",
    "ScenarioResult",
    "ScenarioCompareRequest",
    "ScenarioCompareResult",
    "ScenarioVariableSummary",
    "GridSummaryMetrics",
    "CompactGridData",
    "get_spatial_mask",
    "get_supported_states",
    "scenario_store",
]

