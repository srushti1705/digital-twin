/**
 * API client and types for What-If climate scenario perturbations.
 */

const configuredApiUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const API_BASE_URL = configuredApiUrl
  ? configuredApiUrl.replace(/\/+$/, "")
  : "http://localhost:8000";

export type ScenarioRunRequest = {
  scenario_name: string;
  geographic_scope: string; // 'all_india' or ISO state code like 'IN-MH'
  baseline_source: "historical" | "prediction";
  baseline_date: string;
  variable_selection: "rainfall" | "temperature" | "both";
  rainfall_change_percent: number;
  tmax_delta: number;
  tmin_delta: number;
  duration_days?: number;
  description?: string;
};

export type GridSummaryMetrics = {
  mean: number | null;
  min: number | null;
  max: number | null;
  std: number | null;
  valid_cell_count: number;
  missing_cell_count: number;
};

export type ScenarioVariableSummary = {
  variable: string;
  units: string;
  baseline: GridSummaryMetrics;
  scenario: GridSummaryMetrics;
  delta_mean: number | null;
  delta_percent: number | null;
  affected_cells_count: number;
};

export type CompactGridData = {
  variable: string;
  units: string;
  spatial_resolution_degrees: number;
  grid_shape: [number, number];
  latitudes: number[];
  longitudes: number[];
  values: (number | null)[][];
  valid_cell_count: number;
  missing_cell_count: number;
};

export type ScenarioResult = {
  scenario_id: string;
  created_at_utc: string;
  parameters: ScenarioRunRequest;
  geographic_scope_name: string;
  scientific_disclaimer: string;
  mathematical_assumptions: string[];
  warnings: string[];
  rainfall_summary: ScenarioVariableSummary | null;
  tmax_summary: ScenarioVariableSummary | null;
  tmin_summary: ScenarioVariableSummary | null;
  rainfall_baseline_grid: CompactGridData | null;
  rainfall_scenario_grid: CompactGridData | null;
  rainfall_difference_grid: CompactGridData | null;
  tmax_baseline_grid: CompactGridData | null;
  tmax_scenario_grid: CompactGridData | null;
  tmax_difference_grid: CompactGridData | null;
  tmin_baseline_grid: CompactGridData | null;
  tmin_scenario_grid: CompactGridData | null;
  tmin_difference_grid: CompactGridData | null;
};

export type ScenarioScope = {
  code: string;
  name: string;
};

export type ScenarioCompareRequest = {
  scenario_id_a: string;
  scenario_id_b: string;
};

export type ScenarioCompareResult = {
  scenario_a_id: string;
  scenario_a_name: string;
  scenario_b_id: string;
  scenario_b_name: string;
  rainfall_mean_a: number | null;
  rainfall_mean_b: number | null;
  rainfall_delta_b_minus_a: number | null;
  tmax_mean_a: number | null;
  tmax_mean_b: number | null;
  tmax_delta_b_minus_a: number | null;
  tmin_mean_a: number | null;
  tmin_mean_b: number | null;
  tmin_delta_b_minus_a: number | null;
  summary_text: string;
};

async function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload &&
      typeof (payload as { detail: unknown }).detail === "string"
        ? (payload as { detail: string }).detail
        : `Scenario API request failed with HTTP ${response.status}.`;
    throw new Error(detail);
  }
  return payload as T;
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { signal });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload &&
      typeof (payload as { detail: unknown }).detail === "string"
        ? (payload as { detail: string }).detail
        : `Scenario API request failed with HTTP ${response.status}.`;
    throw new Error(detail);
  }
  return payload as T;
}

export async function fetchScenarioScopes(signal?: AbortSignal): Promise<ScenarioScope[]> {
  return getJson<ScenarioScope[]>("/api/scenarios/scopes", signal);
}

export async function runClimateScenario(
  request: ScenarioRunRequest,
  signal?: AbortSignal,
): Promise<ScenarioResult> {
  return postJson<ScenarioResult>("/api/scenarios/run", request, signal);
}

export async function fetchScenarioResult(
  scenarioId: string,
  signal?: AbortSignal,
): Promise<ScenarioResult> {
  return getJson<ScenarioResult>(`/api/scenarios/${scenarioId}`, signal);
}

export async function compareClimateScenarios(
  request: ScenarioCompareRequest,
  signal?: AbortSignal,
): Promise<ScenarioCompareResult> {
  return postJson<ScenarioCompareResult>("/api/scenarios/compare", request, signal);
}

