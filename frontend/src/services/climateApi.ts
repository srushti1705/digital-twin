const configuredApiUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const API_BASE_URL = configuredApiUrl
  ? configuredApiUrl.replace(/\/+$/, "")
  : "http://localhost:8000";

export type ApiHealth = {
  status: string;
};

export type ImdRainfallDates = {
  available_dates: string[];
  source: string;
  product: string;
  units: string;
  spatial_resolution_degrees: number;
};

export type ImdRainfallGrid = {
  source: string;
  product: string;
  observation_date: string;
  units: string;
  spatial_resolution_degrees: number | null;
  grid_shape: [number, number];
  latitudes: number[];
  longitudes: number[];
  values: (number | null)[][];
  valid_cell_count: number;
  missing_cell_count: number;
  filename: string;
  source_url: string;
};

export type ImdRainfallCell = {
  source: string;
  product: string;
  observation_date: string;
  latitude: number;
  longitude: number;
  rainfall: number | null;
  units: string;
  spatial_resolution_degrees: number | null;
  missing: boolean;
  filename: string;
  source_url: string;
};

export type ImdTemperatureGrid = {
  source: string;
  product: string;
  variable: "Tmax" | "Tmin";
  observation_date: string;
  units: "°C";
  spatial_resolution_degrees: number;
  grid_shape: [number, number];
  latitudes: number[];
  longitudes: number[];
  values: (number | null)[][];
  valid_cell_count: number;
  missing_cell_count: number;
  filename: string;
  source_url: string;
};

export type ImdTemperatureCell = {
  source: string;
  product: string;
  variable: "Tmax" | "Tmin";
  observation_date: string;
  latitude: number;
  longitude: number;
  temperature: number | null;
  units: "°C";
  spatial_resolution_degrees: number;
  missing: boolean;
  filename: string;
  source_url: string;
};

export type ImdTemperatureDates = {
  source: string;
  product: string;
  variable: "Tmax" | "Tmin";
  units: "°C";
  spatial_resolution_degrees: number;
  available_dates: string[];
  source_url: string;
};

export type NasaRainfallAvailability = {
  source: string;
  layer: string;
  product: string;
  product_version: string;
  units: string;
  spatial_resolution: string;
  visualization_only: boolean;
  queryable_measurements: boolean;
  observation_date_start: string;
  observation_date_end: string;
  available_time_intervals: string[];
  imagery_service: string;
  legend_url: string | null;
  attribution: string;
};

async function fetchPayload<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { signal });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload &&
      typeof payload.detail === "string"
        ? payload.detail
        : `Climate API request failed with HTTP ${response.status}.`;
    throw new Error(detail);
  }
  if (typeof payload !== "object" || payload === null) {
    throw new Error("Climate API response has an unexpected format.");
  }
  return payload as T;
}

export async function fetchApiHealth(signal: AbortSignal): Promise<ApiHealth> {
  const response = await fetch(`${API_BASE_URL}/health`, { signal });
  if (!response.ok) {
    throw new Error(`Climate API health check failed with HTTP ${response.status}.`);
  }

  const payload: unknown = await response.json();
  if (
    typeof payload !== "object" ||
    payload === null ||
    !("status" in payload) ||
    typeof payload.status !== "string"
  ) {
    throw new Error("Climate API health response has an unexpected format.");
  }

  return { status: payload.status };
}

export async function fetchImdRainfallDates(
  signal?: AbortSignal,
): Promise<ImdRainfallDates> {
  const payload = await fetchPayload<ImdRainfallDates>(
    "/api/climate/rainfall/imd/dates",
    signal,
  );
  if (
    !Array.isArray(payload.available_dates) ||
    !payload.available_dates.every((value) => typeof value === "string") ||
    typeof payload.spatial_resolution_degrees !== "number"
  ) {
    throw new Error("IMD rainfall date response has an unexpected format.");
  }
  return payload;
}

export async function fetchImdRainfallGrid(
  observationDate: string,
  signal?: AbortSignal,
): Promise<ImdRainfallGrid> {
  const query = new URLSearchParams({ date: observationDate });
  const payload = await fetchPayload<ImdRainfallGrid>(
    `/api/climate/rainfall/imd?${query}`,
    signal,
  );
  if (
    !Array.isArray(payload.latitudes) ||
    !Array.isArray(payload.longitudes) ||
    !Array.isArray(payload.values) ||
    !Array.isArray(payload.grid_shape) ||
    payload.values.length !== payload.latitudes.length ||
    payload.values.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== payload.longitudes.length ||
        row.some((value) => value !== null && typeof value !== "number"),
    ) ||
    typeof payload.observation_date !== "string"
  ) {
    throw new Error("IMD rainfall grid response has an unexpected format.");
  }
  return payload;
}

export async function fetchImdRainfallCell(
  observationDate: string,
  latitude: number,
  longitude: number,
  signal?: AbortSignal,
): Promise<ImdRainfallCell> {
  const query = new URLSearchParams({
    date: observationDate,
    latitude: String(latitude),
    longitude: String(longitude),
  });
  const payload = await fetchPayload<ImdRainfallCell>(
    `/api/climate/rainfall/imd/cell?${query}`,
    signal,
  );
  if (
    typeof payload.latitude !== "number" ||
    typeof payload.longitude !== "number" ||
    (payload.rainfall !== null && typeof payload.rainfall !== "number") ||
    typeof payload.missing !== "boolean"
  ) {
    throw new Error("IMD rainfall cell response has an unexpected format.");
  }
  return payload;
}

export async function fetchImdTemperatureDates(
  variable: "tmax" | "tmin",
  signal?: AbortSignal,
): Promise<ImdTemperatureDates> {
  const payload = await fetchPayload<ImdTemperatureDates>(
    `/api/climate/temperature/${variable}/dates`,
    signal,
  );
  if (
    !Array.isArray(payload.available_dates) ||
    !payload.available_dates.every((value) => typeof value === "string") ||
    payload.spatial_resolution_degrees !== 1
  ) {
    throw new Error(`IMD ${variable} date response has an unexpected format.`);
  }
  return payload;
}

export async function fetchImdTemperatureGrid(
  variable: "tmax" | "tmin",
  observationDate: string,
  signal?: AbortSignal,
): Promise<ImdTemperatureGrid> {
  const query = new URLSearchParams({ date: observationDate });
  const payload = await fetchPayload<ImdTemperatureGrid>(
    `/api/climate/temperature/${variable}?${query}`,
    signal,
  );
  if (
    !Array.isArray(payload.latitudes) ||
    !Array.isArray(payload.longitudes) ||
    !Array.isArray(payload.values) ||
    payload.values.length !== payload.latitudes.length ||
    payload.values.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== payload.longitudes.length ||
        row.some((value) => value !== null && typeof value !== "number"),
    ) ||
    payload.variable !== (variable === "tmax" ? "Tmax" : "Tmin")
  ) {
    throw new Error(`IMD ${variable} grid response has an unexpected format.`);
  }
  return payload;
}

export async function fetchImdTemperatureCell(
  variable: "tmax" | "tmin",
  observationDate: string,
  latitude: number,
  longitude: number,
  signal?: AbortSignal,
): Promise<ImdTemperatureCell> {
  const query = new URLSearchParams({
    date: observationDate,
    latitude: String(latitude),
    longitude: String(longitude),
  });
  const payload = await fetchPayload<ImdTemperatureCell>(
    `/api/climate/temperature/${variable}/cell?${query}`,
    signal,
  );
  if (
    typeof payload.latitude !== "number" ||
    typeof payload.longitude !== "number" ||
    (payload.temperature !== null && typeof payload.temperature !== "number") ||
    typeof payload.missing !== "boolean"
  ) {
    throw new Error(`IMD ${variable} cell response has an unexpected format.`);
  }
  return payload;
}

export async function fetchNasaRainfallAvailability(
  signal?: AbortSignal,
): Promise<NasaRainfallAvailability> {
  const payload = await fetchPayload<NasaRainfallAvailability>(
    "/api/climate/rainfall/nasa/dates",
    signal,
  );
  if (
    typeof payload.observation_date_start !== "string" ||
    typeof payload.observation_date_end !== "string" ||
    !Array.isArray(payload.available_time_intervals) ||
    (payload.legend_url !== null && typeof payload.legend_url !== "string") ||
    payload.visualization_only !== true ||
    payload.queryable_measurements !== false
  ) {
    throw new Error("NASA GIBS availability response has an unexpected format.");
  }
  return payload;
}
