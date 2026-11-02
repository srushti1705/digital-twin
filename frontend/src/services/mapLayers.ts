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

  // 2. City & Town Floating Annotations with 3 Zoom-Level Visibility Tiers:
  // - Tier 1 (Metros): visible from India view (100km to 5,500,000m)
  // - Tier 2 (Major Regional Hubs & Capitals: Mumbai, Pune, Nashik, Nagpur, Solapur): visible at State view (40km to 2,400,000m)
  // - Tier 3 (Towns & secondary cities): visible at Closer zoom (1km to 850,000m)
  for (const place of INDIAN_CITIES) {
    const nearDistance = place.tier === 1 ? 80_000 : place.tier === 2 ? 30_000 : 1_000;
    const farDistance =
      place.tier === 1 ? 5_500_000 : place.tier === 2 ? 2_400_000 : 850_000;
    const pointColor =
      place.tier === 1
        ? Cesium.Color.fromCssColorString("#38bdf8")
        : place.tier === 2
          ? Cesium.Color.fromCssColorString("#7dd3fc")
          : Cesium.Color.fromCssColorString("#94a3b8");
    const pointSize = place.tier === 1 ? 3.8 : place.tier === 2 ? 3.0 : 2.4;
    const fontSize = place.tier === 1 ? 10 : place.tier === 2 ? 9.5 : 8.5;
    const fontWeight = place.tier === 1 ? "600" : "500";
    const textColor =
      place.tier === 1 ? "#f8fafc" : place.tier === 2 ? "#e2e8f0" : "#94a3b8";

    cityLabelsSource.entities.add({
      id: `city-label-${place.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
      name: place.name,
      position: Cesium.Cartesian3.fromDegrees(place.lon, place.lat),
      properties: new Cesium.PropertyBag({
        kind: "city-label",
        cityName: place.name,
        state_name: place.state,
        tier: place.tier,
        latitude: place.lat,
        longitude: place.lon,
      }),
      point: {
        pixelSize: pointSize,
        color: pointColor,
        outlineColor: Cesium.Color.fromCssColorString("#03101c"),
        outlineWidth: 1.2,
        distanceDisplayCondition: new Cesium.DistanceDisplayCondition(nearDistance, farDistance),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: place.name,
        font: `${fontWeight} ${fontSize}px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`,
        fillColor: Cesium.Color.fromCssColorString(textColor),
        outlineColor: Cesium.Color.fromCssColorString("#03101c"),
        outlineWidth: 2.2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, -7),
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
