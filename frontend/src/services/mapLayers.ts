import * as Cesium from "cesium";
import { INDIAN_CITIES, STATE_ANCHORS } from "../data/indianPlaces";

/**
 * Original Cesium 3D Earth Satellite Imagery Provider (Esri World Imagery).
 * High-resolution realistic satellite imagery with no street maps, road tiles, or watermarks.
 */
export const ORIGINAL_EARTH_IMAGERY_PROVIDER = new Cesium.UrlTemplateImageryProvider({
  url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  credit: "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
  maximumLevel: 19,
});

/**
 * Creates custom Cesium DataSources for State Name Labels and City/Town labels.
 * Designed as elegant, floating 3D geographical annotations over the satellite Earth.
 * Utilizes native Cesium DistanceDisplayCondition for zero-overhead, GPU-accelerated LOD.
 */
export function buildGeographicLabelsDataSources(): {
  stateLabelsSource: Cesium.CustomDataSource;
  cityLabelsSource: Cesium.CustomDataSource;
} {
  const stateLabelsSource = new Cesium.CustomDataSource("stateLabelsDataSource");
  const cityLabelsSource = new Cesium.CustomDataSource("cityLabelsDataSource");

  // 1. State & Union Territory Floating Labels
  // Major states visible at India view. Compact states & UTs (e.g. Delhi, Goa, Chandigarh, NE states)
  // appear upon zooming into regional view (< 2.4M) to avoid visual overcrowding.
  for (const anchor of STATE_ANCHORS) {
    const isCompact = Boolean(anchor.isCompact);
    const nearDistance = isCompact ? 50_000 : 200_000;
    const farDistance = isCompact ? 2_400_000 : 7_500_000;
    const fontSize = isCompact ? 9.5 : 10.5;
    const fontWeight = isCompact ? "500" : "600";

    stateLabelsSource.entities.add({
      id: `state-label-${anchor.code}`,
      name: anchor.name,
      position: Cesium.Cartesian3.fromDegrees(anchor.lon, anchor.lat),
      properties: new Cesium.PropertyBag({
        kind: "state-label",
        iso_code: anchor.code,
        state_name: anchor.name,
        region: anchor.region,
        is_union_territory: anchor.isUnionTerritory,
      }),
      label: {
        text: anchor.name,
        font: `${fontWeight} ${fontSize}px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`,
        fillColor: Cesium.Color.fromCssColorString("#f1f5f9"),
        outlineColor: Cesium.Color.fromCssColorString("#03101c"),
        outlineWidth: 2.5,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        showBackground: true,
        backgroundColor: Cesium.Color.fromCssColorString("#03101c").withAlpha(0.55),
        backgroundPadding: new Cesium.Cartesian2(5, 2.5),
        verticalOrigin: Cesium.VerticalOrigin.CENTER,
        horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
        distanceDisplayCondition: new Cesium.DistanceDisplayCondition(nearDistance, farDistance),
        scaleByDistance: new Cesium.NearFarScalar(300_000, 1.0, 6_000_000, 0.75),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  }

  // 2. City & Town Floating Markers + Labels (LOD Hierarchy: Tier 1, Tier 2, Tier 3)
  for (const city of INDIAN_CITIES) {
    const nearDistance = 10_000;
    let farDistance = 1_800_000; // Tier 3 default
    let dotPixelSize = 3.5;
    let fontSize = 9;

    if (city.tier === 1) {
      farDistance = 4_500_000;
      dotPixelSize = 4.5;
      fontSize = 10;
    } else if (city.tier === 2) {
      farDistance = 2_800_000;
      dotPixelSize = 4.0;
      fontSize = 9.5;
    }

    cityLabelsSource.entities.add({
      id: `city-marker-${city.name.toLowerCase().replace(/\s+/g, "-")}`,
      name: city.name,
      position: Cesium.Cartesian3.fromDegrees(city.lon, city.lat),
      properties: new Cesium.PropertyBag({
        kind: "city-marker",
        city_name: city.name,
        state: city.state,
        tier: city.tier,
      }),
      point: {
        pixelSize: dotPixelSize,
        color: Cesium.Color.fromCssColorString("#38bdf8").withAlpha(0.95),
        outlineColor: Cesium.Color.fromCssColorString("#020617"),
        outlineWidth: 1.5,
        distanceDisplayCondition: new Cesium.DistanceDisplayCondition(nearDistance, farDistance),
        scaleByDistance: new Cesium.NearFarScalar(100_000, 1.0, 3_500_000, 0.8),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: city.name,
        font: `500 ${fontSize}px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`,
        fillColor: Cesium.Color.fromCssColorString("#e2e8f0"),
        outlineColor: Cesium.Color.fromCssColorString("#020617"),
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        showBackground: false,
        pixelOffset: new Cesium.Cartesian2(0, -9),
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
        distanceDisplayCondition: new Cesium.DistanceDisplayCondition(nearDistance, farDistance),
        scaleByDistance: new Cesium.NearFarScalar(100_000, 1.0, 3_500_000, 0.8),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  }

  return { stateLabelsSource, cityLabelsSource };
}

/**
 * Applies subtle, clean outlines following real state shapes on the 3D Earth.
 */
export function styleBoundaryEntities(
  source: Cesium.GeoJsonDataSource,
  selectedCode?: string,
) {
  const values = source.entities.values;
  for (let i = 0; i < values.length; i += 1) {
    const entity = values[i];
    if (!entity.polygon) continue;

    const properties = entity.properties?.getValue(Cesium.JulianDate.now());
    const code =
      properties?.iso_code ??
      properties?.shapeGroup ??
      properties?.shapeISO ??
      properties?.STATE_CODE ??
      entity.id;
    const isSelected = selectedCode ? code === selectedCode : false;

    entity.polygon.material = new Cesium.ColorMaterialProperty(
      (isSelected
        ? Cesium.Color.fromCssColorString("#38bdf8")
        : Cesium.Color.fromCssColorString("#38bdf8")
      ).withAlpha(isSelected ? 0.18 : 0.015),
    );
    entity.polygon.outlineColor = new Cesium.ConstantProperty(
      (isSelected
        ? Cesium.Color.fromCssColorString("#67e8f9")
        : Cesium.Color.fromCssColorString("#38bdf8")
      ).withAlpha(isSelected ? 0.95 : 0.45),
    );
    entity.polygon.outlineWidth = new Cesium.ConstantProperty(isSelected ? 2 : 1);
  }
}

/**
 * Creates a Cesium WebMapServiceImageryProvider for NASA GIBS satellite products.
 */
export function createGibsSatelliteProvider(
  layerName: string,
  timeDate: string,
  format = "image/png",
): Cesium.WebMapServiceImageryProvider {
  return new Cesium.WebMapServiceImageryProvider({
    url: "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
    layers: layerName,
    parameters: {
      service: "WMS",
      version: "1.1.1",
      request: "GetMap",
      transparent: "true",
      format,
      time: timeDate,
    },
    credit: "NASA Global Imagery Browse Services (GIBS)",
  });
}

/** Diverging color scale for rainfall differences (Deficit: Amber/Red; Surplus: Cyan/Royal Blue). */
export function rainfallDiffColor(delta: number): number[] {
  if (Math.abs(delta) < 0.2) return [0, 0, 0, 0];
  if (delta < 0) {
    const intensity = Math.min(1.0, Math.abs(delta) / 25.0);
    // Interpolate from light orange to deep crimson
    return [
      Math.round(230 + 25 * intensity),
      Math.round(140 * (1 - intensity * 0.7)),
      Math.round(50 * (1 - intensity * 0.7)),
      Math.round(180 + 55 * intensity),
    ];
  }
  const intensity = Math.min(1.0, delta / 25.0);
  // Interpolate from bright cyan to deep ocean blue
  return [
    Math.round(50 * (1 - intensity * 0.5)),
    Math.round(180 + 40 * (1 - intensity)),
    Math.round(240),
    Math.round(180 + 55 * intensity),
  ];
}

/** Diverging color scale for temperature differences (Cooling: Cyan/Blue; Warming: Yellow/Crimson). */
export function temperatureDiffColor(delta: number): number[] {
  if (Math.abs(delta) < 0.1) return [0, 0, 0, 0];
  if (delta < 0) {
    const intensity = Math.min(1.0, Math.abs(delta) / 5.0);
    return [
      Math.round(40 * (1 - intensity * 0.5)),
      Math.round(130 + 100 * (1 - intensity)),
      Math.round(245),
      Math.round(180 + 55 * intensity),
    ];
  }
  const intensity = Math.min(1.0, delta / 5.0);
  return [
    Math.round(245),
    Math.round(160 * (1 - intensity * 0.8)),
    Math.round(40 * (1 - intensity * 0.8)),
    Math.round(180 + 55 * intensity),
  ];
}
