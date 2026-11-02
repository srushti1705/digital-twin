/**
 * API client and types for live and near-real-time climate telemetry & satellite layers.
 */

const configuredApiUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const API_BASE_URL = configuredApiUrl
  ? configuredApiUrl.replace(/\/+$/, "")
  : "http://localhost:8000";

export type ProviderStatus = {
  provider_id: string;
  name: string;
  description: string;
  configured: boolean;
  available: boolean;
  data_status: "fresh" | "stale" | "cached" | "credentials_required" | "unavailable";
  requires_credentials: boolean;
  credential_env_vars: string[];
  documentation_url: string;
  last_fetched_utc: string | null;
  last_observation_utc: string | null;
  freshness_seconds: number | null;
  error_message: string | null;
  attribution: string;
};

export type StationWeatherRecord = {
  location_id: string;
  location_name: string;
  state: string;
  state_code: string;
  latitude: number;
  longitude: number;
  elevation_m: number | null;
  source: string;
  source_product: string;
  is_observation: boolean;
  data_status: "fresh" | "stale" | "cached" | "credentials_required" | "unavailable";
  observation_time_utc: string;
  fetched_at_utc: string;
  temperature_c: number | null;
  apparent_temperature_c: number | null;
  rainfall_mm: number | null;
  relative_humidity_percent: number | null;
  wind_speed_kmh: number | null;
  surface_pressure_hpa: number | null;
  weather_condition: string | null;
  weather_code: number | null;
};

export type SatelliteLayerInfo = {
  layer_id: string;
  name: string;
  description: string;
  category: "true_color" | "clouds" | "precipitation" | "thermal";
  service_type: "wms" | "wmts";
  service_url: string;
  layer_name: string;
  format: string;
  time_support: boolean;
  default_date: string;
  date_start: string | null;
  date_end: string | null;
  attribution: string;
  legend_url: string | null;
  opacity_default: number;
};

export type LiveCoverageInfo = {
  coverage_extent: {
    min_latitude: number;
    max_latitude: number;
    min_longitude: number;
    max_longitude: number;
  };
  supported_locations_count: number;
  active_providers_count: number;
  latest_update_utc: string | null;
  providers: ProviderStatus[];
};

export type SatelliteImageryMeta = {
  layer_id: string;
  layer_name: string;
  service_url: string;
  service_type: string;
  format: string;
  time: string;
  opacity: number;
  attribution: string;
  legend_url: string | null;
  wms_parameters: {
    service: string;
    version: string;
    request: string;
    layers: string;
    format: string;
    transparent: string;
    time: string;
  };
};

async function fetchJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { signal });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload &&
      typeof (payload as { detail: unknown }).detail === "string"
        ? (payload as { detail: string }).detail
        : `Live API request failed with HTTP ${response.status}.`;
    throw new Error(detail);
  }
  return payload as T;
}

export async function fetchLiveStatus(signal?: AbortSignal): Promise<ProviderStatus[]> {
  return fetchJson<ProviderStatus[]>("/api/live/status", signal);
}

export async function fetchLiveWeather(
  stateCode?: string,
  search?: string,
  signal?: AbortSignal,
): Promise<StationWeatherRecord[]> {
  const query = new URLSearchParams();
  if (stateCode) query.set("state_code", stateCode);
  if (search) query.set("search", search);
  const qStr = query.toString();
  return fetchJson<StationWeatherRecord[]>(`/api/live/weather${qStr ? `?${qStr}` : ""}`, signal);
}

export async function fetchSatelliteLayers(signal?: AbortSignal): Promise<SatelliteLayerInfo[]> {
  return fetchJson<SatelliteLayerInfo[]>("/api/live/satellite/layers", signal);
}

export async function fetchSatelliteImageryMeta(
  layerId: string,
  observationDate?: string,
  signal?: AbortSignal,
): Promise<SatelliteImageryMeta> {
  const query = new URLSearchParams({ layer_id: layerId });
  if (observationDate) query.set("observation_date", observationDate);
  return fetchJson<SatelliteImageryMeta>(`/api/live/satellite/imagery?${query}`, signal);
}

export async function fetchLiveCoverage(signal?: AbortSignal): Promise<LiveCoverageInfo> {
  return fetchJson<LiveCoverageInfo>("/api/live/coverage", signal);
}
