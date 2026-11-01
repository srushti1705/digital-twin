type PhotonFeature = {
  geometry?: {
    coordinates?: unknown;
  };
  properties?: Record<string, unknown>;
};

export type GeocodedPlace = {
  label: string;
  name: string;
  region: string;
  longitude: number;
  latitude: number;
  kind: string;
  extent?: [number, number, number, number];
  id?: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readExtent(value: unknown): GeocodedPlace["extent"] {
  if (!Array.isArray(value) || value.length !== 4) return undefined;
  const extent = value.map(Number);
  if (
    !extent.every(Number.isFinite) ||
    extent[0] < -180 ||
    extent[2] > 180 ||
    extent[1] < -90 ||
    extent[3] > 90 ||
    extent[0] > extent[2] ||
    extent[1] > extent[3]
  ) {
    return undefined;
  }
  return extent as [number, number, number, number];
}

function toPlace(feature: PhotonFeature): GeocodedPlace | undefined {
  const coordinates = feature.geometry?.coordinates;
  const properties = feature.properties;
  if (
    !Array.isArray(coordinates) ||
    coordinates.length < 2 ||
    !properties
  ) {
    return undefined;
  }

  const longitude = Number(coordinates[0]);
  const latitude = Number(coordinates[1]);
  if (
    !Number.isFinite(longitude) ||
    !Number.isFinite(latitude) ||
    longitude < -180 ||
    longitude > 180 ||
    latitude < -90 ||
    latitude > 90
  ) {
    return undefined;
  }

  const nameValue = properties.name ?? properties.city ?? properties.country;
  const name = typeof nameValue === "string" ? nameValue : undefined;
  if (!name) return undefined;

  const regionParts = [properties.state, properties.country].filter(
    (part): part is string => typeof part === "string" && part.length > 0 && part !== name,
  );
  const region = regionParts.join(", ") || "Geographic location";
  const label = [name, ...regionParts].join(", ");
  const kind =
    typeof properties.type === "string"
      ? properties.type
      : typeof properties.osm_value === "string"
        ? properties.osm_value
        : "place";
  const id =
    typeof properties.osm_id === "number" || typeof properties.osm_id === "string"
      ? String(properties.osm_id)
      : undefined;

  return {
    label,
    name,
    region,
    longitude,
    latitude,
    kind,
    extent: readExtent(properties.extent),
    id,
  };
}

export async function searchPlaces(
  query: string,
  signal: AbortSignal,
): Promise<GeocodedPlace[]> {
  const url = new URL("https://photon.komoot.io/api/");
  url.searchParams.set("q", query);
  url.searchParams.set("limit", "6");

  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Place search failed with HTTP ${response.status}.`);
  }

  const payload: unknown = await response.json();
  if (!isRecord(payload) || !Array.isArray(payload.features)) {
    throw new Error("Place search returned an unexpected response.");
  }

  return payload.features
    .map((feature) => (isRecord(feature) ? toPlace(feature as PhotonFeature) : undefined))
    .filter((place): place is GeocodedPlace => place !== undefined);
}
