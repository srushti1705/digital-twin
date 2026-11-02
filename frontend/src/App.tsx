import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  CalendarDays,
  Cloud,
  Compass,
  Crosshair,
  Droplets,
  Globe2,
  Info,
  Layers3,
  Search,
  Thermometer,
  X,
} from "lucide-react";
import { CameraFlyTo, ImageryLayer, Viewer } from "resium";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import "./App.css";
import {
  ORIGINAL_EARTH_IMAGERY_PROVIDER,
  buildGeographicLabelsDataSources,
  createGibsSatelliteProvider,
  rainfallDiffColor,
  styleBoundaryEntities,
  temperatureDiffColor,
} from "./services/mapLayers";
import { fetchApiHealth } from "./services/climateApi";
import { searchPlaces } from "./services/geocoding";
import type { GeocodedPlace } from "./services/geocoding";
import {
  fetchImdRainfallCell,
  fetchImdRainfallDates,
  fetchImdRainfallGrid,
  fetchImdTemperatureCell,
  fetchImdTemperatureDates,
  fetchImdTemperatureGrid,
  fetchNasaRainfallAvailability,
  fetchPredictionCatalog,
  fetchPredictionCell,
  fetchPredictionGrid,
} from "./services/climateApi";
import type {
  ImdRainfallCell,
  ImdRainfallGrid,
  ImdTemperatureCell,
  ImdTemperatureGrid,
  NasaRainfallAvailability,
  PredictionCatalog,
  PredictionGrid,
  PredictionVariable,
} from "./services/climateApi";
import {
  fetchLiveStatus,
  fetchLiveWeather,
  fetchSatelliteLayers,
} from "./services/liveClimateApi";
import type {
  ProviderStatus,
  SatelliteLayerInfo,
  StationWeatherRecord,
} from "./services/liveClimateApi";
import {
  fetchScenarioScopes,
  runClimateScenario,
} from "./services/scenarioApi";
import type {
  CompactGridData,
  ScenarioResult,
  ScenarioScope,
} from "./services/scenarioApi";
import { LiveTelemetryBar } from "./components/LiveTelemetryBar";
import { WhatIfScenarioDock } from "./components/WhatIfScenarioDock";

const INDIA_VIEW = { longitude: 81.8, latitude: 22.8, height: 4_950_000 };
const MAHARASHTRA_VIEW = { longitude: 76.2, latitude: 19.3, height: 1_150_000 };
const GLOBAL_VIEW = { longitude: 80, latitude: 20, height: 14_000_000 };
const STATE_BOUNDARY_MAX_HEIGHT = 7_000_000;

type Location = { code: string; name: string; region: string };
type ViewMode = "historical" | "live" | "forecast" | "whatif";
type ViewerHandle = { cesiumElement?: Cesium.Viewer };
type ApiStatus = "checking" | "healthy" | "unavailable";
type RainfallSource = "none" | "imd" | "nasa";
type TemperatureSource = "none" | "tmax" | "tmin";
type RainfallCellStatus = "idle" | "loading" | "unavailable" | "error";

type SelectedPredictionCell = {
  latitude: number;
  longitude: number;
  value: number | null;
  missing: boolean;
};

const modeDescriptions: Record<ViewMode, { title: string; description: string }> = {
  historical: {
    title: "Historical observations",
    description: "Explore recorded climate observations across India.",
  },
  live: {
    title: "Latest available conditions",
    description: "View the latest available weather data and satellite imagery. Model-derived conditions are identified separately from station observations.",
  },
  forecast: {
    title: "AI prediction",
    description: "Explore one-day-ahead AI estimates from available model dates. Predictions are not observations or official forecasts.",
  },
  whatif: {
    title: "What-if simulation",
    description: "Test hypothetical rainfall and temperature changes against a selected baseline.",
  },
};

const unavailableClimateLayers = [
  {
    label: "INSAT-3D Cloud Imagery",
    source: "ISRO MOSDAC / IMD georeferenced satellite feed is not currently connected.",
    sourceUrl: "https://www.mosdac.gov.in/",
    sourceLabel: "MOSDAC data catalog",
    icon: Cloud,
  },
  {
    label: "Cyclone Tracking",
    source: "RSMC New Delhi cyclone track and intensity bulletin feed is not connected.",
    sourceUrl: "https://rsmcnewdelhi.imd.gov.in/",
    sourceLabel: "RSMC New Delhi",
    icon: Compass,
  },
];

function setupPanDrag(viewer: Cesium.Viewer) {
  const controls = viewer.scene.screenSpaceCameraController;
  controls.enableRotate = false;
  controls.enableTilt = false;
  controls.enableLook = false;
  controls.enableTranslate = false;
  controls.enableZoom = false;
  controls.zoomEventTypes = Cesium.CameraEventType.PINCH;

  const canvas = viewer.scene.canvas;
  let panPosition: Cesium.Cartesian2 | undefined;
  const zoomAtCursor = (event: WheelEvent) => {
    event.preventDefault();
    const camera = viewer.camera;
    const currentHeight = camera.positionCartographic.height;
    const minHeight = 20_000;
    const maxHeight = 20_000_000;
    const zoomFactor = event.deltaY < 0 ? 0.78 : 1.28;
    const nextHeight = Cesium.Math.clamp(currentHeight * zoomFactor, minHeight, maxHeight);
    const distance = currentHeight - nextHeight;
    if (distance === 0) return;

    const bounds = canvas.getBoundingClientRect();
    const pointer = new Cesium.Cartesian2(
      event.clientX - bounds.left,
      event.clientY - bounds.top,
    );
    const anchorBefore = camera.pickEllipsoid(pointer, viewer.scene.globe.ellipsoid);
    camera.zoomIn(distance);
    if (!anchorBefore) {
      viewer.scene.requestRender();
      return;
    }
    const anchorAfter = camera.pickEllipsoid(pointer, viewer.scene.globe.ellipsoid);
    if (!anchorAfter) {
      viewer.scene.requestRender();
      return;
    }
    const correction = Cesium.Cartesian3.subtract(
      anchorBefore,
      anchorAfter,
      new Cesium.Cartesian3(),
    );
    camera.position = Cesium.Cartesian3.add(camera.position, correction, camera.position);
    viewer.scene.requestRender();
  };

  const startPan = (event: MouseEvent) => {
    if (event.button !== 0) return;
    panPosition = new Cesium.Cartesian2(event.clientX, event.clientY);
  };
  const movePan = (event: MouseEvent) => {
    if (!panPosition) return;
    const deltaX = event.clientX - panPosition.x;
    const deltaY = event.clientY - panPosition.y;
    panPosition = new Cesium.Cartesian2(event.clientX, event.clientY);
    const unitsPerPixel =
      Math.max(viewer.camera.positionCartographic.height, 100) /
      Math.max(canvas.clientHeight, 1);
    viewer.camera.moveRight(-deltaX * unitsPerPixel);
    viewer.camera.moveUp(deltaY * unitsPerPixel);
    viewer.scene.requestRender();
  };
  const endPan = () => {
    panPosition = undefined;
  };
  canvas.addEventListener("wheel", zoomAtCursor, { passive: false });
  canvas.addEventListener("mousedown", startPan);
  window.addEventListener("mousemove", movePan);
  window.addEventListener("mouseup", endPan);
  window.addEventListener("blur", endPan);
  return () => {
    canvas.removeEventListener("wheel", zoomAtCursor);
    canvas.removeEventListener("mousedown", startPan);
    window.removeEventListener("mousemove", movePan);
    window.removeEventListener("mouseup", endPan);
    window.removeEventListener("blur", endPan);
  };
}

const rainfallColorStops = [
  { value: 0, color: [0, 0, 0, 0] },
  { value: 1, color: [40, 124, 255, 190] },
  { value: 5, color: [42, 211, 240, 200] },
  { value: 15, color: [42, 195, 115, 210] },
  { value: 30, color: [239, 223, 73, 220] },
  { value: 60, color: [249, 146, 43, 230] },
  { value: 120, color: [228, 62, 49, 235] },
  { value: 250, color: [165, 34, 86, 240] },
];

function rainfallColor(value: number) {
  if (value <= 0) return rainfallColorStops[0].color;
  for (let index = 1; index < rainfallColorStops.length; index += 1) {
    const upper = rainfallColorStops[index];
    const lower = rainfallColorStops[index - 1];
    if (value > upper.value) continue;
    const ratio = (value - lower.value) / (upper.value - lower.value);
    return lower.color.map((channel, channelIndex) =>
      Math.round(channel + (upper.color[channelIndex] - channel) * ratio),
    );
  }
  return rainfallColorStops[rainfallColorStops.length - 1].color;
}

const temperatureColorStops = [
  { value: -10, color: [35, 91, 214, 218] },
  { value: 0, color: [45, 193, 230, 218] },
  { value: 15, color: [54, 194, 125, 218] },
  { value: 25, color: [231, 217, 70, 224] },
  { value: 35, color: [243, 136, 49, 230] },
  { value: 45, color: [204, 48, 47, 235] },
];

function temperatureColor(value: number) {
  for (let index = 1; index < temperatureColorStops.length; index += 1) {
    const upper = temperatureColorStops[index];
    const lower = temperatureColorStops[index - 1];
    if (value > upper.value) continue;
    const ratio = Math.max(
      0,
      Math.min(1, (value - lower.value) / (upper.value - lower.value)),
    );
    return lower.color.map((channel, channelIndex) =>
      Math.round(channel + (upper.color[channelIndex] - channel) * ratio),
    );
  }
  return temperatureColorStops[temperatureColorStops.length - 1].color;
}

function isNasaDateAvailable(
  observationDate: string,
  intervals: readonly string[],
) {
  return intervals.some((interval) => {
    const [start, end, period] = interval.split("/");
    if (!start || !end) return interval === observationDate;
    return period === "P1D" && observationDate >= start && observationDate <= end;
  });
}

const rainfallRasterCache = new WeakMap<ImdRainfallGrid, string>();
const temperatureRasterCache = new WeakMap<ImdTemperatureGrid, string>();
const predictionRasterCache = new WeakMap<PredictionGrid, string>();
const scenarioRasterCache = new WeakMap<CompactGridData, Map<string, string>>();

function createRainfallRaster(grid: ImdRainfallGrid) {
  const cachedRaster = rainfallRasterCache.get(grid);
  if (cachedRaster) return cachedRaster;

  const width = grid.longitudes.length;
  const height = grid.latitudes.length;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser could not create the rainfall raster canvas.");

  const image = context.createImageData(width, height);
  grid.values.forEach((row, latitudeIndex) => {
    row.forEach((value, longitudeIndex) => {
      if (value === null) return;
      const color = rainfallColor(value);
      const northUpRow = height - latitudeIndex - 1;
      const pixelOffset = (northUpRow * width + longitudeIndex) * 4;
      image.data[pixelOffset] = color[0];
      image.data[pixelOffset + 1] = color[1];
      image.data[pixelOffset + 2] = color[2];
      image.data[pixelOffset + 3] = color[3];
    });
  });
  context.putImageData(image, 0, 0);
  const raster = canvas.toDataURL("image/png");
  rainfallRasterCache.set(grid, raster);
  return raster;
}

function createTemperatureRaster(grid: ImdTemperatureGrid) {
  const cachedRaster = temperatureRasterCache.get(grid);
  if (cachedRaster) return cachedRaster;

  const width = grid.longitudes.length;
  const height = grid.latitudes.length;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser could not create the temperature raster canvas.");

  const image = context.createImageData(width, height);
  grid.values.forEach((row, latitudeIndex) => {
    row.forEach((value, longitudeIndex) => {
      if (value === null) return;
      const color = temperatureColor(value);
      const northUpRow = height - latitudeIndex - 1;
      const pixelOffset = (northUpRow * width + longitudeIndex) * 4;
      image.data[pixelOffset] = color[0];
      image.data[pixelOffset + 1] = color[1];
      image.data[pixelOffset + 2] = color[2];
      image.data[pixelOffset + 3] = color[3];
    });
  });
  context.putImageData(image, 0, 0);
  const raster = canvas.toDataURL("image/png");
  temperatureRasterCache.set(grid, raster);
  return raster;
}

function createPredictionRaster(grid: PredictionGrid) {
  const cachedRaster = predictionRasterCache.get(grid);
  if (cachedRaster) return cachedRaster;

  const width = grid.longitudes.length;
  const height = grid.latitudes.length;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser could not create the climate prediction raster.");

  const image = context.createImageData(width, height);
  grid.values.forEach((row, latitudeIndex) => {
    row.forEach((value, longitudeIndex) => {
      if (value === null) return;
      const color = grid.variable === "rainfall" ? rainfallColor(value) : temperatureColor(value);
      const northUpRow = height - latitudeIndex - 1;
      const pixelOffset = (northUpRow * width + longitudeIndex) * 4;
      image.data[pixelOffset] = color[0];
      image.data[pixelOffset + 1] = color[1];
      image.data[pixelOffset + 2] = color[2];
      image.data[pixelOffset + 3] = color[3];
    });
  });
  context.putImageData(image, 0, 0);
  const raster = canvas.toDataURL("image/png");
  predictionRasterCache.set(grid, raster);
  return raster;
}

function createScenarioCompactRaster(
  grid: CompactGridData,
  isDifference: boolean,
  isRainfall: boolean,
) {
  const cacheKey = `${isDifference ? "difference" : "value"}:${isRainfall ? "rainfall" : "temperature"}`;
  const cachedRaster = scenarioRasterCache.get(grid)?.get(cacheKey);
  if (cachedRaster) return cachedRaster;

  const width = grid.longitudes.length;
  const height = grid.latitudes.length;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not create scenario raster canvas.");

  const image = context.createImageData(width, height);
  grid.values.forEach((row, latitudeIndex) => {
    row.forEach((value, longitudeIndex) => {
      if (value === null) return;
      const color = isDifference
        ? (isRainfall ? rainfallDiffColor(value) : temperatureDiffColor(value))
        : (isRainfall ? rainfallColor(value) : temperatureColor(value));
      const northUpRow = height - latitudeIndex - 1;
      const pixelOffset = (northUpRow * width + longitudeIndex) * 4;
      image.data[pixelOffset] = color[0];
      image.data[pixelOffset + 1] = color[1];
      image.data[pixelOffset + 2] = color[2];
      image.data[pixelOffset + 3] = color[3];
    });
  });
  context.putImageData(image, 0, 0);
  const raster = canvas.toDataURL("image/png");
  const cachedRasters = scenarioRasterCache.get(grid) ?? new Map<string, string>();
  cachedRasters.set(cacheKey, raster);
  scenarioRasterCache.set(grid, cachedRasters);
  return raster;
}

function configureViewerScene(viewer: Cesium.Viewer) {
  viewer.scene.globe.baseColor = Cesium.Color.BLACK;
  viewer.scene.backgroundColor = Cesium.Color.BLACK;
  viewer.scene.globe.enableLighting = false;
  viewer.scene.globe.showGroundAtmosphere = true;
  if (viewer.scene.skyAtmosphere) {
    viewer.scene.skyAtmosphere.show = true;
    viewer.scene.skyAtmosphere.brightnessShift = 0.05;
    viewer.scene.skyAtmosphere.saturationShift = 0.1;
  }
  viewer.scene.globe.atmosphereLightIntensity = 10.0;
  viewer.scene.globe.depthTestAgainstTerrain = false;
}

const initialCameraDestination = Cesium.Cartesian3.fromDegrees(
  INDIA_VIEW.longitude,
  INDIA_VIEW.latitude,
  INDIA_VIEW.height,
);
const initialCameraOrientation = {
  heading: 0,
  pitch: Cesium.Math.toRadians(-88),
  roll: 0,
};

const ClimateGlobe = memo(function ClimateGlobe({
  onViewerReady,
}: {
  onViewerReady: (handle: ViewerHandle | null) => void;
}) {
  return (
    <div className="globe-canvas" aria-label="Interactive Cesium satellite globe">
      <Viewer
        ref={onViewerReady}
        full
        baseLayer={false}
        animation={false}
        baseLayerPicker={false}
        timeline={false}
        geocoder={false}
        homeButton={false}
        fullscreenButton={false}
        sceneModePicker={false}
        navigationHelpButton={false}
        infoBox={false}
        selectionIndicator={false}
        requestRenderMode
        maximumRenderTimeChange={Number.POSITIVE_INFINITY}
        useBrowserRecommendedResolution
      >
        <ImageryLayer imageryProvider={ORIGINAL_EARTH_IMAGERY_PROVIDER} />
        <CameraFlyTo
          destination={initialCameraDestination}
          orientation={initialCameraOrientation}
          duration={0}
          once
        />
      </Viewer>
    </div>
  );
});

export default function App() {
  const viewerRef = useRef<ViewerHandle | null>(null);
  const boundarySourceRef = useRef<Cesium.GeoJsonDataSource | null>(null);
  const stateLabelsSourceRef = useRef<Cesium.CustomDataSource | null>(null);
  const cityLabelsSourceRef = useRef<Cesium.CustomDataSource | null>(null);
  const rainfallLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const rainfallSourceRef = useRef<RainfallSource>("none");
  const temperatureSourceRef = useRef<TemperatureSource>("none");
  const viewModeRef = useRef<ViewMode>("historical");
  const predictionGridRef = useRef<PredictionGrid | null>(null);
  const rainfallDateRef = useRef("");
  const rainfallCellControllerRef = useRef<AbortController | null>(null);
  const boundaryPreferenceRef = useRef(true);
  const stateLabelsPreferenceRef = useRef(true);
  const cityLabelsPreferenceRef = useRef(true);
  const searchControllerRef = useRef<AbortController | null>(null);

  const [viewerReady, setViewerReady] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<Location>({
    code: "IN",
    name: "India",
    region: "Indian subcontinent",
  });
  const selectedLocationRef = useRef(selectedLocation);
  const [boundariesVisible, setBoundariesVisible] = useState(true);
  const [stateLabelsVisible, setStateLabelsVisible] = useState(true);
  const [cityLabelsVisible, setCityLabelsVisible] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(() => window.innerWidth > 900);
  const [infoOpen, setInfoOpen] = useState(() => window.innerWidth > 640);
  const [viewMode, setViewMode] = useState<ViewMode>("historical");
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [apiStatusMessage, setApiStatusMessage] = useState("Checking climate API");

  // Historical state
  const [rainfallSource, setRainfallSource] = useState<RainfallSource>("none");
  const [temperatureSource, setTemperatureSource] = useState<TemperatureSource>("none");
  const [rainfallDate, setRainfallDate] = useState("");
  const [imdDates, setImdDates] = useState<string[]>([]);
  const [rainfallGrid, setRainfallGrid] = useState<ImdRainfallGrid | null>(null);
  const [temperatureDates, setTemperatureDates] = useState<Record<"tmax" | "tmin", string[]>>({
    tmax: [],
    tmin: [],
  });
  const [temperatureGrid, setTemperatureGrid] = useState<ImdTemperatureGrid | null>(null);
  const [temperatureLoading, setTemperatureLoading] = useState(false);
  const [temperatureError, setTemperatureError] = useState("");
  const [rainfallLoading, setRainfallLoading] = useState(false);
  const [rainfallError, setRainfallError] = useState("");
  const [nasaAvailability, setNasaAvailability] = useState<NasaRainfallAvailability | null>(null);
  const [nasaAvailabilityStatus, setNasaAvailabilityStatus] = useState<"checking" | "available" | "unavailable">("checking");
  const [selectedRainfallCell, setSelectedRainfallCell] = useState<ImdRainfallCell | null>(null);
  const [selectedTemperatureCell, setSelectedTemperatureCell] = useState<ImdTemperatureCell | null>(null);
  const [rainfallCellStatus, setRainfallCellStatus] = useState<RainfallCellStatus>("idle");
  const [rainfallCellError, setRainfallCellError] = useState("");

  // AI Prediction state
  const [predictionCatalog, setPredictionCatalog] = useState<PredictionCatalog | null>(null);
  const [predictionCatalogError, setPredictionCatalogError] = useState("");
  const [predictionVariable, setPredictionVariable] = useState<PredictionVariable>("rainfall");
  const [predictionInitializationDate, setPredictionInitializationDate] = useState("");
  const [predictionGrid, setPredictionGrid] = useState<PredictionGrid | null>(null);
  const [predictionError, setPredictionError] = useState("");
  const [predictionRequested, setPredictionRequested] = useState(false);
  const [predictionRunAt, setPredictionRunAt] = useState<string | null>(null);
  const [selectedPredictionCell, setSelectedPredictionCell] = useState<SelectedPredictionCell | null>(null);

  // Live Mode state
  const [liveWeatherRecords, setLiveWeatherRecords] = useState<StationWeatherRecord[]>([]);
  const [liveWeatherLoading, setLiveWeatherLoading] = useState(false);
  const [liveWeatherError, setLiveWeatherError] = useState("");
  const [liveProviders, setLiveProviders] = useState<ProviderStatus[]>([]);
  const [liveMetadataError, setLiveMetadataError] = useState("");
  const [selectedLiveStation, setSelectedLiveStation] = useState<StationWeatherRecord | null>(null);
  const [satelliteOverlayActive, setSatelliteOverlayActive] = useState(false);
  const [satelliteLayersList, setSatelliteLayersList] = useState<SatelliteLayerInfo[]>([]);
  const [selectedSatelliteLayerId, setSelectedSatelliteLayerId] = useState("true_color_terra");
  const [satelliteDate, setSatelliteDate] = useState("");

  // What-If Scenario state
  const [scenarioScopesList, setScenarioScopesList] = useState<ScenarioScope[]>([
    { code: "all_india", name: "All India (Nationwide Grid)" },
  ]);
  const [scenarioScopesError, setScenarioScopesError] = useState("");
  const [scenarioScope, setScenarioScope] = useState("all_india");
  const [scenarioBaselineSource, setScenarioBaselineSource] = useState<"historical" | "prediction">("historical");
  const [scenarioBaselineDate, setScenarioBaselineDate] = useState("2025-07-15");
  const [scenarioVarSelection, setScenarioVarSelection] = useState<"rainfall" | "temperature" | "both">("rainfall");
  const [rainfallChangePct, setRainfallChangePct] = useState(20);
  const [tmaxDelta, setTmaxDelta] = useState(2.0);
  const [tminDelta, setTminDelta] = useState(1.5);
  const [scenarioRunning, setScenarioRunning] = useState(false);
  const [scenarioError, setScenarioError] = useState("");
  const [scenarioResult, setScenarioResult] = useState<ScenarioResult | null>(null);
  const [scenarioViewLayer, setScenarioViewLayer] = useState<"scenario" | "baseline" | "difference">("scenario");
  const [scenarioActiveVar, setScenarioActiveVar] = useState<"rainfall" | "tmax" | "tmin">("rainfall");

  // Place search
  const [placeQuery, setPlaceQuery] = useState("");
  const [placeResults, setPlaceResults] = useState<GeocodedPlace[]>([]);
  const [placeSearchError, setPlaceSearchError] = useState("");
  const [placeSearchBusy, setPlaceSearchBusy] = useState(false);

  const selectedPredictionModel = predictionCatalog?.models.find(
    (model) => model.variable === predictionVariable,
  );
  const predictionLoading = Boolean(
    viewMode === "forecast" &&
      predictionRequested &&
      selectedPredictionModel?.available &&
      predictionInitializationDate &&
      !predictionGrid &&
      !predictionError,
  );
  const predictionInitializationDates =
    selectedPredictionModel?.available_initialization_dates ?? [];
  const scenarioTargetGrid = useMemo(() => {
    if (!scenarioResult) return null;
    if (scenarioActiveVar === "rainfall") {
      return scenarioViewLayer === "difference"
        ? scenarioResult.rainfall_difference_grid
        : scenarioViewLayer === "baseline"
          ? scenarioResult.rainfall_baseline_grid
          : scenarioResult.rainfall_scenario_grid;
    }
    if (scenarioActiveVar === "tmax") {
      return scenarioViewLayer === "difference"
        ? scenarioResult.tmax_difference_grid
        : scenarioViewLayer === "baseline"
          ? scenarioResult.tmax_baseline_grid
          : scenarioResult.tmax_scenario_grid;
    }
    return scenarioViewLayer === "difference"
      ? scenarioResult.tmin_difference_grid
      : scenarioViewLayer === "baseline"
        ? scenarioResult.tmin_baseline_grid
        : scenarioResult.tmin_scenario_grid;
  }, [scenarioActiveVar, scenarioResult, scenarioViewLayer]);
  const activeSatelliteLayer = useMemo(
    () =>
      viewMode === "live" && satelliteOverlayActive
        ? satelliteLayersList.find((layer) => layer.layer_id === selectedSatelliteLayerId) ?? null
        : null,
    [satelliteLayersList, selectedSatelliteLayerId, satelliteOverlayActive, viewMode],
  );
  const activeScenarioGrid = viewMode === "whatif" ? scenarioTargetGrid : null;
  const activePredictionGrid = viewMode === "forecast" ? predictionGrid : null;
  const activeRainfallGrid =
    viewMode === "historical" && rainfallSource === "imd" ? rainfallGrid : null;
  const activeTemperatureGrid =
    viewMode === "historical" && temperatureSource !== "none" ? temperatureGrid : null;
  const activeNasaAvailability =
    viewMode === "historical" &&
    rainfallSource === "nasa" &&
    nasaAvailability &&
    isNasaDateAvailable(rainfallDate, nasaAvailability.available_time_intervals)
      ? nasaAvailability
      : null;
  const activeSatelliteDate = activeSatelliteLayer
    ? satelliteDate || activeSatelliteLayer.default_date
    : null;
  const activeHistoricalNasaDate = activeNasaAvailability ? rainfallDate : null;
  const activeScenarioIsDifference =
    viewMode === "whatif" && scenarioViewLayer === "difference";
  const activeScenarioIsRainfall =
    viewMode === "whatif" && scenarioActiveVar === "rainfall";

  useEffect(() => {
    const closeLayersOnNarrowViewport = () => {
      if (window.innerWidth <= 900) setSettingsOpen(false);
    };
    window.addEventListener("resize", closeLayersOnNarrowViewport);
    return () => window.removeEventListener("resize", closeLayersOnNarrowViewport);
  }, []);

  useEffect(() => {
    rainfallSourceRef.current = rainfallSource;
    temperatureSourceRef.current = temperatureSource;
    viewModeRef.current = viewMode;
    predictionGridRef.current = predictionGrid;
    rainfallDateRef.current = rainfallDate;
  }, [predictionGrid, rainfallDate, rainfallSource, temperatureSource, viewMode]);

  const setViewerRef = useCallback((handle: ViewerHandle | null) => {
    viewerRef.current = handle;
    const viewer = handle?.cesiumElement;
    if (viewer) {
      configureViewerScene(viewer);
      setViewerReady(true);
    }
  }, []);

  const flyToIndia = useCallback(() => {
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewer) return;
    setSelectedLocation({
      code: "IN",
      name: "India",
      region: "Indian subcontinent",
    });
    setInfoOpen(false);
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(
        INDIA_VIEW.longitude,
        INDIA_VIEW.latitude,
        INDIA_VIEW.height,
      ),
      orientation: {
        heading: 0,
        pitch: Cesium.Math.toRadians(-88),
        roll: 0,
      },
      duration: 1.8,
    });
  }, []);

  const flyToMaharashtra = useCallback(() => {
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewer) return;
    setSelectedLocation({
      code: "IN-MH",
      name: "Maharashtra",
      region: "Western India",
    });
    setInfoOpen(false);
    const source = boundarySourceRef.current;
    const state = source?.entities.values.find((entity) => {
      const properties = entity.properties?.getValue(Cesium.JulianDate.now());
      const code = properties?.iso_code ?? properties?.STATE_CODE ?? entity.id;
      return code === "IN-MH";
    });
    if (state) {
      void viewer.flyTo(state, {
        duration: 1.8,
        offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-75), 1_150_000),
      });
      return;
    }
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(
        MAHARASHTRA_VIEW.longitude,
        MAHARASHTRA_VIEW.latitude,
        MAHARASHTRA_VIEW.height,
      ),
      orientation: {
        heading: 0,
        pitch: Cesium.Math.toRadians(-75),
        roll: 0,
      },
      duration: 1.8,
    });
  }, []);

  const flyToGlobalView = useCallback(() => {
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewer) return;
    setInfoOpen(false);
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(
        GLOBAL_VIEW.longitude,
        GLOBAL_VIEW.latitude,
        GLOBAL_VIEW.height,
      ),
      orientation: {
        heading: 0,
        pitch: Cesium.Math.toRadians(-88),
        roll: 0,
      },
      duration: 2.2,
    });
  }, []);

  const selectRainfallSource = useCallback(
    (source: RainfallSource) => {
      rainfallCellControllerRef.current?.abort();
      setSelectedRainfallCell(null);
      setSelectedTemperatureCell(null);
      setRainfallCellError("");
      setRainfallCellStatus("idle");
      setRainfallError("");
      setTemperatureError("");
      setRainfallGrid(null);
      setTemperatureGrid(null);
      setRainfallLoading(source === "imd" && viewMode === "historical");
      setTemperatureLoading(false);
      setRainfallSource(source);
      setTemperatureSource("none");
      if (source === "imd" && imdDates.length > 0 && !imdDates.includes(rainfallDate)) {
        setRainfallDate(imdDates[imdDates.length - 1]);
      } else if (source === "nasa" && nasaAvailability) {
        setRainfallDate(nasaAvailability.observation_date_end);
      }
    },
    [imdDates, nasaAvailability, rainfallDate, viewMode],
  );

  const selectTemperatureSource = useCallback(
    (source: TemperatureSource) => {
      rainfallCellControllerRef.current?.abort();
      setSelectedRainfallCell(null);
      setSelectedTemperatureCell(null);
      setRainfallCellError("");
      setRainfallCellStatus("idle");
      setRainfallError("");
      setTemperatureError("");
      setRainfallGrid(null);
      setTemperatureGrid(null);
      setRainfallLoading(false);
      setTemperatureLoading(source !== "none" && viewMode === "historical");
      setRainfallSource("none");
      setTemperatureSource(source);
      if (source !== "none") {
        const availableDates = temperatureDates[source];
        if (availableDates.length > 0 && !availableDates.includes(rainfallDate)) {
          setRainfallDate(availableDates[availableDates.length - 1]);
        }
      }
    },
    [rainfallDate, temperatureDates, viewMode],
  );

  const selectPredictionVariable = useCallback(
    (variable: PredictionVariable) => {
      setPredictionVariable(variable);
      setPredictionGrid(null);
      setSelectedPredictionCell(null);
      setPredictionError("");
      setPredictionRequested(false);
      setPredictionRunAt(null);
      const model = predictionCatalog?.models.find((candidate) => candidate.variable === variable);
      const dates = model?.available_initialization_dates ?? [];
      if (dates.length > 0) {
        setPredictionInitializationDate(dates[dates.length - 1]);
      }
    },
    [predictionCatalog],
  );

  // Setup viewer and scene
  useEffect(() => {
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewer) return;

    let cancelled = false;
    const cleanupPan = setupPanDrag(viewer);
    const updateBoundaryVisibility = () => {
      const height = viewer.camera.positionCartographic.height;
      const shouldShow = boundaryPreferenceRef.current && height <= STATE_BOUNDARY_MAX_HEIGHT;
      if (boundarySourceRef.current && boundarySourceRef.current.show !== shouldShow) {
        boundarySourceRef.current.show = shouldShow;
        viewer.scene.requestRender();
      }
    };
    viewer.camera.changed.addEventListener(updateBoundaryVisibility);
    updateBoundaryVisibility();

    // Custom annotations data sources
    const { stateLabelsSource, cityLabelsSource } = buildGeographicLabelsDataSources();
    stateLabelsSourceRef.current = stateLabelsSource;
    cityLabelsSourceRef.current = cityLabelsSource;
    void Promise.all([
      viewer.dataSources.add(stateLabelsSource),
      viewer.dataSources.add(cityLabelsSource),
    ]).catch((error: unknown) => {
      if (!cancelled) console.error("Geographic label initialization failed:", error);
    });

    // Boundaries GeoJSON
    void Cesium.GeoJsonDataSource.load("/india-states.geojson", {
      stroke: Cesium.Color.fromCssColorString("#38bdf8").withAlpha(0.45),
      fill: Cesium.Color.fromCssColorString("#38bdf8").withAlpha(0.015),
      strokeWidth: 1,
      clampToGround: false,
    })
      .then((source) => {
        if (cancelled) {
          source.entities.removeAll();
          return;
        }
        boundarySourceRef.current = source;
        source.show = boundaryPreferenceRef.current;
        styleBoundaryEntities(source, selectedLocationRef.current.code);
        return viewer.dataSources.add(source);
      })
      .then(() => {
        if (cancelled) return;
        viewer.scene.requestRender();
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        console.error("Boundary load failure:", error);
      });

    // Screen-space interaction handler
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction(
      (movement: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
        const position = viewer.camera.pickEllipsoid(
          movement.position,
          viewer.scene.globe.ellipsoid,
        );
        if (!position) return;
        const cartographic = Cesium.Cartographic.fromCartesian(position);
        const latitude = Cesium.Math.toDegrees(cartographic.latitude);
        const longitude = Cesium.Math.toDegrees(cartographic.longitude);

        const picked = viewer.scene.pick(movement.position);
        if (Cesium.defined(picked) && picked.id) {
          const entity = picked.id as Cesium.Entity;
          const props = entity.properties?.getValue(Cesium.JulianDate.now());
          const code = props?.iso_code ?? props?.STATE_CODE ?? entity.id;
          const name = props?.state_name ?? props?.ST_NM ?? entity.name ?? "Indian Region";
          const region = props?.REGION ?? props?.region ?? "Indian subcontinent";
          setSelectedLocation({ code: String(code), name: String(name), region: String(region) });
          setSettingsOpen(false);
          setInfoOpen(true);
        }

        // Forecast Mode inspection
        if (viewModeRef.current === "forecast") {
          const grid = predictionGridRef.current;
          if (!grid) return;
          const latMin = grid.latitudes[0] - grid.spatial_resolution_degrees / 2;
          const latMax = grid.latitudes[grid.latitudes.length - 1] + grid.spatial_resolution_degrees / 2;
          const lonMin = grid.longitudes[0] - grid.spatial_resolution_degrees / 2;
          const lonMax = grid.longitudes[grid.longitudes.length - 1] + grid.spatial_resolution_degrees / 2;
          if (latitude < latMin || latitude > latMax || longitude < lonMin || longitude > lonMax) {
            setRainfallCellError("This location is outside the forecast grid.");
            setRainfallCellStatus("error");
            return;
          }
          const controller = new AbortController();
          rainfallCellControllerRef.current?.abort();
          rainfallCellControllerRef.current = controller;
          setRainfallCellStatus("loading");
          void fetchPredictionCell(
            grid.variable,
            grid.initialization_date,
            latitude,
            longitude,
            controller.signal,
          )
            .then((cell: { latitude: number; longitude: number; value: number | null; missing: boolean }) => {
              setSelectedPredictionCell({
                latitude: cell.latitude,
                longitude: cell.longitude,
                value: cell.value,
                missing: cell.missing,
              });
              setRainfallCellStatus("idle");
              setSettingsOpen(false);
              setInfoOpen(true);
            })
            .catch((err: unknown) => {
              if (controller.signal.aborted) return;
              setRainfallCellError(err instanceof Error ? err.message : "Could not read forecast cell.");
              setRainfallCellStatus("error");
            });
          return;
        }

        // Historical Mode inspection
        if (viewModeRef.current === "historical") {
          if (rainfallSourceRef.current === "imd" && rainfallDateRef.current) {
            const controller = new AbortController();
            rainfallCellControllerRef.current?.abort();
            rainfallCellControllerRef.current = controller;
            setRainfallCellStatus("loading");
            void fetchImdRainfallCell(rainfallDateRef.current, latitude, longitude, controller.signal)
              .then((cell) => {
                setSelectedRainfallCell(cell);
                setRainfallCellStatus("idle");
                setSettingsOpen(false);
                setInfoOpen(true);
              })
              .catch((err) => {
                if (controller.signal.aborted) return;
                setRainfallCellError(err instanceof Error ? err.message : "Could not inspect cell");
                setRainfallCellStatus("error");
              });
          } else if (temperatureSourceRef.current !== "none" && rainfallDateRef.current) {
            const controller = new AbortController();
            rainfallCellControllerRef.current?.abort();
            rainfallCellControllerRef.current = controller;
            setRainfallCellStatus("loading");
            void fetchImdTemperatureCell(
              temperatureSourceRef.current,
              rainfallDateRef.current,
              latitude,
              longitude,
              controller.signal,
            )
              .then((cell) => {
                setSelectedTemperatureCell(cell);
                setRainfallCellStatus("idle");
                setSettingsOpen(false);
                setInfoOpen(true);
              })
              .catch((err) => {
                if (controller.signal.aborted) return;
                setRainfallCellError(err instanceof Error ? err.message : "Could not inspect cell");
                setRainfallCellStatus("error");
              });
          }
        }
      },
      Cesium.ScreenSpaceEventType.LEFT_CLICK,
    );

    return () => {
      cancelled = true;
      cleanupPan();
      viewer.camera.changed.removeEventListener(updateBoundaryVisibility);
      handler.destroy();
      if (!viewer.isDestroyed()) {
        for (const source of [
          boundarySourceRef.current,
          stateLabelsSourceRef.current,
          cityLabelsSourceRef.current,
        ]) {
          if (source && viewer.dataSources.contains(source)) {
            void viewer.dataSources.remove(source, true);
          }
        }
      }
      boundarySourceRef.current = null;
      stateLabelsSourceRef.current = null;
      cityLabelsSourceRef.current = null;
    };
  }, [viewerReady]);

  // Initial load of background services
  useEffect(() => {
    const controller = new AbortController();
    void fetchApiHealth(controller.signal)
      .then((health) => {
        setApiStatus(health.status === "healthy" ? "healthy" : "unavailable");
        setApiStatusMessage(health.status === "healthy" ? "Climate API connected" : "API degraded");
      })
      .catch(() => {
        setApiStatus("unavailable");
        setApiStatusMessage("Backend API unreachable");
      });

    void fetchImdRainfallDates(controller.signal)
      .then((payload) => {
        setImdDates(payload.available_dates);
        if (payload.available_dates.length > 0) {
          setRainfallDate(payload.available_dates[payload.available_dates.length - 1]);
          setScenarioBaselineDate(payload.available_dates[payload.available_dates.length - 1]);
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setRainfallError(
            error instanceof Error ? error.message : "Could not load available IMD rainfall dates.",
          );
        }
      });

    void fetchImdTemperatureDates("tmax", controller.signal).then((payload) => {
      setTemperatureDates((prev) => ({ ...prev, tmax: payload.available_dates }));
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setTemperatureError(
          error instanceof Error ? error.message : "Could not load available IMD Tmax dates.",
        );
      }
    });

    void fetchImdTemperatureDates("tmin", controller.signal).then((payload) => {
      setTemperatureDates((prev) => ({ ...prev, tmin: payload.available_dates }));
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setTemperatureError(
          error instanceof Error ? error.message : "Could not load available IMD Tmin dates.",
        );
      }
    });

    void fetchNasaRainfallAvailability(controller.signal).then((avail: NasaRainfallAvailability) => {
      setNasaAvailability(avail);
      setNasaAvailabilityStatus("available");
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setNasaAvailabilityStatus("unavailable");
      setRainfallError(
        error instanceof Error ? error.message : "NASA GIBS rainfall imagery availability could not be checked.",
      );
    });

    void fetchPredictionCatalog(controller.signal).then((cat) => {
      setPredictionCatalog(cat);
      const rainModel = cat.models.find((m) => m.variable === "rainfall");
      if (rainModel?.available_initialization_dates?.length) {
        setPredictionInitializationDate(
          rainModel.available_initialization_dates[rainModel.available_initialization_dates.length - 1],
        );
      }
    }).catch((err) => setPredictionCatalogError(err instanceof Error ? err.message : "Model catalog error"));

    // Live data metadata
    void fetchLiveStatus(controller.signal).then(setLiveProviders).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setLiveMetadataError(
          error instanceof Error ? error.message : "Live provider status could not be loaded.",
        );
      }
    });
    void fetchSatelliteLayers(controller.signal).then((layers) => {
      setSatelliteLayersList(layers);
      if (layers.length > 0) {
        setSatelliteDate((currentDate) => currentDate || layers[0].default_date);
      }
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setLiveMetadataError(
          error instanceof Error ? error.message : "Satellite layer information could not be loaded.",
        );
      }
    });
    void fetchScenarioScopes(controller.signal).then(setScenarioScopesList).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setScenarioScopesError(
          error instanceof Error ? error.message : "Scenario geographic scopes could not be loaded.",
        );
      }
    });

    return () => controller.abort();
  }, []);

  // Refresh live weather every five minutes and cancel outstanding requests on mode changes.
  useEffect(() => {
    if (viewMode !== "live") return;

    let activeController: AbortController | null = null;
    const refreshLiveWeather = async () => {
      if (activeController) return;
      const controller = new AbortController();
      activeController = controller;
      setLiveWeatherLoading(true);
      setLiveWeatherError("");
      try {
        const records = await fetchLiveWeather(undefined, undefined, controller.signal);
        setLiveWeatherRecords(records);
        setSelectedLiveStation((selected) =>
          selected
            ? records.find((record) => record.location_id === selected.location_id) ?? null
            : null,
        );
        try {
          setLiveProviders(await fetchLiveStatus(controller.signal));
          setLiveMetadataError("");
        } catch (error: unknown) {
          if (!controller.signal.aborted) {
            setLiveMetadataError(
              error instanceof Error ? error.message : "Live provider status could not be refreshed.",
            );
          }
        }
      } catch (error: unknown) {
        if (!controller.signal.aborted) {
          setLiveWeatherError(
            error instanceof Error ? error.message : "Could not load the latest available weather data.",
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setLiveWeatherLoading(false);
          activeController = null;
        }
      }
    };

    void refreshLiveWeather();
    const refreshTimer = window.setInterval(() => {
      void refreshLiveWeather();
    }, 5 * 60 * 1000);
    return () => {
      window.clearInterval(refreshTimer);
      activeController?.abort();
    };
  }, [viewMode]);

  // Historical Grid fetching
  useEffect(() => {
    if (viewMode !== "historical") return;
    if (rainfallSource === "imd" && rainfallDate) {
      const controller = new AbortController();
      void fetchImdRainfallGrid(rainfallDate, controller.signal)
        .then((grid) => {
          setRainfallGrid(grid);
          setRainfallLoading(false);
        })
        .catch((err) => {
          if (controller.signal.aborted) return;
          setRainfallError(err instanceof Error ? err.message : "Could not load IMD rainfall grid.");
          setRainfallLoading(false);
        });
      return () => controller.abort();
    }
    if ((temperatureSource === "tmax" || temperatureSource === "tmin") && rainfallDate) {
      const controller = new AbortController();
      void fetchImdTemperatureGrid(temperatureSource, rainfallDate, controller.signal)
        .then((grid) => {
          setTemperatureGrid(grid);
          setTemperatureLoading(false);
        })
        .catch((err) => {
          if (controller.signal.aborted) return;
          setTemperatureError(err instanceof Error ? err.message : "Could not load IMD temperature grid.");
          setTemperatureLoading(false);
        });
      return () => controller.abort();
    }
  }, [rainfallDate, rainfallSource, temperatureSource, viewMode]);

  // AI Prediction Grid fetching
  useEffect(() => {
    if (viewMode !== "forecast" || !predictionRequested) return;
    if (!selectedPredictionModel?.available || !predictionInitializationDate) return;
    const controller = new AbortController();
    void fetchPredictionGrid(predictionVariable, predictionInitializationDate, controller.signal)
      .then((grid) => {
        setPredictionGrid(grid);
        setPredictionRunAt(new Date().toISOString());
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setPredictionError(err instanceof Error ? err.message : "Failed to load forecast grid.");
        setPredictionRequested(false);
      });
    return () => controller.abort();
  }, [
    predictionInitializationDate,
    predictionRequested,
    predictionVariable,
    selectedPredictionModel,
    viewMode,
  ]);

  // Main Cesium Raster Overlay Manager (handles Historical, Live Satellite, Prediction, and What-If)
  useEffect(() => {
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewerReady || !viewer) return;

    let cancelled = false;
    let addedLayer: Cesium.ImageryLayer | undefined;

    const previousLayer = rainfallLayerRef.current;
    if (previousLayer) {
      viewer.imageryLayers.remove(previousLayer, true);
      rainfallLayerRef.current = null;
    }

    const applyLayer = async () => {
      // 1. Live Satellite Overlay Mode
      if (viewMode === "live" && activeSatelliteLayer && activeSatelliteDate) {
        const provider = createGibsSatelliteProvider(
          activeSatelliteLayer.layer_name,
          activeSatelliteDate,
          activeSatelliteLayer.format,
        );
        addedLayer = viewer.imageryLayers.addImageryProvider(provider);
        addedLayer.alpha = activeSatelliteLayer.opacity_default || 0.75;
      }

      // 2. What-If Scenario Mode
      else if (viewMode === "whatif" && activeScenarioGrid) {
          const image = createScenarioCompactRaster(
            activeScenarioGrid,
            activeScenarioIsDifference,
            activeScenarioIsRainfall,
          );
          const res = activeScenarioGrid.spatial_resolution_degrees;
          const provider = await Cesium.SingleTileImageryProvider.fromUrl(image, {
            rectangle: Cesium.Rectangle.fromDegrees(
              activeScenarioGrid.longitudes[0] - res / 2,
              activeScenarioGrid.latitudes[0] - res / 2,
              activeScenarioGrid.longitudes[activeScenarioGrid.longitudes.length - 1] + res / 2,
              activeScenarioGrid.latitudes[activeScenarioGrid.latitudes.length - 1] + res / 2,
            ),
          });
          if (cancelled) return;
          addedLayer = viewer.imageryLayers.addImageryProvider(provider);
          addedLayer.alpha = 0.84;
      }

      // 3. AI Prediction Mode
      else if (viewMode === "forecast" && activePredictionGrid) {
        const image = createPredictionRaster(activePredictionGrid);
        const resolution = activePredictionGrid.spatial_resolution_degrees;
        const provider = await Cesium.SingleTileImageryProvider.fromUrl(image, {
          rectangle: Cesium.Rectangle.fromDegrees(
            activePredictionGrid.longitudes[0] - resolution / 2,
            activePredictionGrid.latitudes[0] - resolution / 2,
            activePredictionGrid.longitudes[activePredictionGrid.longitudes.length - 1] + resolution / 2,
            activePredictionGrid.latitudes[activePredictionGrid.latitudes.length - 1] + resolution / 2,
          ),
        });
        if (cancelled) return;
        addedLayer = viewer.imageryLayers.addImageryProvider(provider);
        addedLayer.alpha = 0.84;
      }

      // 4. Historical Observation Mode
      else if (viewMode === "historical") {
        if (activeRainfallGrid) {
          const image = createRainfallRaster(activeRainfallGrid);
          const provider = await Cesium.SingleTileImageryProvider.fromUrl(image, {
            rectangle: Cesium.Rectangle.fromDegrees(66.375, 6.375, 100.125, 38.625),
          });
          if (cancelled) return;
          addedLayer = viewer.imageryLayers.addImageryProvider(provider);
          addedLayer.alpha = 0.84;
        } else if (activeTemperatureGrid) {
          const image = createTemperatureRaster(activeTemperatureGrid);
          const provider = await Cesium.SingleTileImageryProvider.fromUrl(image, {
            rectangle: Cesium.Rectangle.fromDegrees(67, 7, 98, 38),
          });
          if (cancelled) return;
          addedLayer = viewer.imageryLayers.addImageryProvider(provider);
          addedLayer.alpha = 0.82;
        } else if (
          activeNasaAvailability &&
          activeHistoricalNasaDate
        ) {
          const provider = new Cesium.WebMapServiceImageryProvider({
            url: activeNasaAvailability.imagery_service,
            layers: activeNasaAvailability.layer,
            parameters: {
              format: "image/png",
              transparent: "true",
              time: activeHistoricalNasaDate,
            },
            tilingScheme: new Cesium.GeographicTilingScheme(),
            enablePickFeatures: false,
            credit: activeNasaAvailability.attribution,
          });
          addedLayer = viewer.imageryLayers.addImageryProvider(provider);
          addedLayer.alpha = 0.88;
        }
      }

      if (addedLayer) {
        rainfallLayerRef.current = addedLayer;
        viewer.scene.requestRender();
      }
    };

    void applyLayer().catch((err: unknown) => {
      if (cancelled) return;
      console.error("Imagery layer application error:", err);
    });

    return () => {
      cancelled = true;
      if (addedLayer) viewer.imageryLayers.remove(addedLayer, true);
      if (rainfallLayerRef.current === addedLayer) rainfallLayerRef.current = null;
    };
  }, [
    viewMode,
    activeSatelliteLayer,
    activeSatelliteDate,
    activeScenarioGrid,
    activeScenarioIsDifference,
    activeScenarioIsRainfall,
    activePredictionGrid,
    activeRainfallGrid,
    activeTemperatureGrid,
    activeNasaAvailability,
    activeHistoricalNasaDate,
    viewerReady,
  ]);

  // Boundaries & Labels toggle listeners
  useEffect(() => {
    boundaryPreferenceRef.current = boundariesVisible;
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewer) return;
    const height = viewer.camera.positionCartographic.height;
    if (boundarySourceRef.current) {
      boundarySourceRef.current.show = boundariesVisible && height <= STATE_BOUNDARY_MAX_HEIGHT;
      viewer.scene.requestRender();
    }
  }, [boundariesVisible]);

  useEffect(() => {
    stateLabelsPreferenceRef.current = stateLabelsVisible;
    if (stateLabelsSourceRef.current) {
      stateLabelsSourceRef.current.show = stateLabelsVisible;
      viewerRef.current?.cesiumElement?.scene.requestRender();
    }
  }, [stateLabelsVisible]);

  useEffect(() => {
    cityLabelsPreferenceRef.current = cityLabelsVisible;
    if (cityLabelsSourceRef.current) {
      cityLabelsSourceRef.current.show = cityLabelsVisible;
      viewerRef.current?.cesiumElement?.scene.requestRender();
    }
  }, [cityLabelsVisible]);

  // Place search submit
  const submitPlaceSearch = async (event: FormEvent) => {
    event.preventDefault();
    if (!placeQuery.trim()) return;
    searchControllerRef.current?.abort();
    const controller = new AbortController();
    searchControllerRef.current = controller;
    setPlaceSearchBusy(true);
    setPlaceSearchError("");
    try {
      const results = await searchPlaces(placeQuery, controller.signal);
      setPlaceResults(results);
      if (results.length === 0) setPlaceSearchError("No places matching that name were found.");
    } catch (err: unknown) {
      if (controller.signal.aborted) return;
      setPlaceSearchError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setPlaceSearchBusy(false);
    }
  };

  const flyToPlace = (place: GeocodedPlace) => {
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewer) return;
    setSelectedLocation({ code: place.id ?? "SEARCH", name: place.name, region: place.region });
    setSettingsOpen(false);
    setInfoOpen(true);
    setPlaceResults([]);
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(place.longitude, place.latitude, 450_000),
      orientation: { heading: 0, pitch: Cesium.Math.toRadians(-75), roll: 0 },
      duration: 2.0,
    });
  };

  const handleSelectLiveStation = useCallback((st: StationWeatherRecord) => {
    setSelectedLiveStation(st);
    setSettingsOpen(false);
    setInfoOpen(true);
    setSelectedLocation({
      code: st.state_code,
      name: st.location_name,
      region: st.state,
    });
    setInfoOpen(true);
    const viewer = viewerRef.current?.cesiumElement;
    if (viewer) {
      viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(st.longitude, st.latitude, 850_000),
        orientation: { heading: 0, pitch: Cesium.Math.toRadians(-75), roll: 0 },
        duration: 1.6,
      });
    }
  }, []);

  const handleRunScenario = useCallback(async () => {
    setScenarioRunning(true);
    setScenarioError("");
    setScenarioResult(null);
    try {
      const res = await runClimateScenario({
        scenario_name: `${scenarioScope === "all_india" ? "All-India" : scenarioScope} Perturbation`,
        geographic_scope: scenarioScope,
        baseline_source: scenarioBaselineSource,
        baseline_date: scenarioBaselineDate,
        variable_selection: scenarioVarSelection,
        rainfall_change_percent: rainfallChangePct,
        tmax_delta: tmaxDelta,
        tmin_delta: tminDelta,
      });
      setScenarioResult(res);
      if (scenarioVarSelection === "temperature") {
        setScenarioActiveVar("tmax");
      } else {
        setScenarioActiveVar("rainfall");
      }
    } catch (err: unknown) {
      setScenarioError(err instanceof Error ? err.message : "Failed to run What-If scenario.");
    } finally {
      setScenarioRunning(false);
    }
  }, [
    scenarioScope,
    scenarioBaselineSource,
    scenarioBaselineDate,
    scenarioVarSelection,
    rainfallChangePct,
    tmaxDelta,
    tminDelta,
  ]);

  const handleResetScenario = useCallback(() => {
    setRainfallChangePct(0);
    setTmaxDelta(0);
    setTminDelta(0);
    setScenarioResult(null);
    setScenarioError("");
  }, []);

  return (
    <main
      className={`earth-app mode-${viewMode}${infoOpen || settingsOpen ? " side-panel-open" : ""}${settingsOpen ? " layers-open" : ""}`}
    >
      <ClimateGlobe onViewerReady={setViewerRef} />

      {/* Top Header */}
      <header className="top-overlay">
        <div className="brand-lockup">
          <span className="brand-mark"><span /></span>
          <div>
            <p className="brand-name">INDIA CLIMATE TWIN</p>
            <p className="location-line">
              {selectedLocation.name}
              <span> / {selectedLocation.region}</span>
            </p>
          </div>
        </div>

        <div className="top-right-controls">
          <div className="mode-switch" aria-label="Application mode">
            <button
              type="button"
              aria-pressed={viewMode === "historical"}
              className={viewMode === "historical" ? "selected" : ""}
              onClick={() => {
                setRainfallGrid(null);
                setTemperatureGrid(null);
                setPredictionGrid(null);
                setSelectedPredictionCell(null);
                setPredictionError("");
                setPredictionRequested(false);
                setPredictionRunAt(null);
                setRainfallLoading(rainfallSource === "imd");
                setTemperatureLoading(temperatureSource !== "none");
                setSettingsOpen(false);
                setInfoOpen(true);
                setViewMode("historical");
              }}
            >
              Historical
            </button>
            <button
              type="button"
              aria-pressed={viewMode === "live"}
              className={viewMode === "live" ? "selected" : ""}
              onClick={() => {
                setRainfallGrid(null);
                setTemperatureGrid(null);
                setPredictionGrid(null);
                setSelectedPredictionCell(null);
                setPredictionError("");
                setPredictionRequested(false);
                setPredictionRunAt(null);
                setSettingsOpen(false);
                setInfoOpen(true);
                setViewMode("live");
              }}
            >
              Live
            </button>
            <button
              type="button"
              aria-pressed={viewMode === "forecast"}
              className={viewMode === "forecast" ? "selected" : ""}
              onClick={() => {
                setRainfallGrid(null);
                setTemperatureGrid(null);
                setPredictionGrid(null);
                setSelectedPredictionCell(null);
                setPredictionError("");
                setPredictionRequested(false);
                setPredictionRunAt(null);
                setRainfallLoading(false);
                setTemperatureLoading(false);
                const dates = selectedPredictionModel?.available_initialization_dates ?? [];
                if (dates.length > 0) {
                  setPredictionInitializationDate(dates[dates.length - 1]);
                }
                setSettingsOpen(false);
                setInfoOpen(true);
                setViewMode("forecast");
              }}
            >
              Prediction
            </button>
            <button
              type="button"
              aria-pressed={viewMode === "whatif"}
              className={viewMode === "whatif" ? "selected" : ""}
              onClick={() => {
                setRainfallGrid(null);
                setTemperatureGrid(null);
                setPredictionGrid(null);
                setSelectedPredictionCell(null);
                setPredictionError("");
                setPredictionRequested(false);
                setPredictionRunAt(null);
                setSettingsOpen(false);
                setInfoOpen(true);
                setViewMode("whatif");
              }}
            >
              What-If
            </button>
          </div>
          <p className="active-mode-description">{modeDescriptions[viewMode].description}</p>

          <div className="top-status" title={apiStatusMessage}>
            <div className={`data-status api-status ${apiStatus}`}>
              <span className="status-dot" />
              <span>
                {apiStatus === "checking"
                  ? "Checking API"
                  : apiStatus === "healthy"
                    ? "API Reachable"
                    : "API Offline"}
              </span>
              <small>
                {viewMode === "live"
                  ? liveWeatherError
                    ? "Latest weather refresh failed"
                    : liveWeatherLoading && liveWeatherRecords.length === 0
                      ? "Loading latest available data"
                      : liveWeatherRecords.length > 0
                        ? `Fetched ${new Date(liveWeatherRecords[0].fetched_at_utc).toLocaleTimeString()}`
                        : "No live weather records available"
                  : viewMode === "whatif"
                    ? "Interactive Simulation"
                    : imdDates.length > 0
                    ? `IMD 2025 Grids (${imdDates.length} days)`
                    : "Checking national datasets"}
              </small>
            </div>
          </div>
        </div>
      </header>

      {/* Map toolbar */}
      <div className="map-toolbar" aria-label="Map controls">
        <form className="place-search" role="search" onSubmit={submitPlaceSearch}>
          <Search size={16} aria-hidden="true" />
          <input
            type="search"
            value={placeQuery}
            aria-label="Search a place on Earth"
            placeholder="Search anywhere"
            onChange={(event) => {
              setPlaceQuery(event.target.value);
              setPlaceSearchError("");
              setPlaceResults([]);
            }}
          />
          <button type="submit" aria-label="Search places" disabled={placeSearchBusy}>
            {placeSearchBusy ? "…" : "Go"}
          </button>
          {(placeResults.length > 0 || placeSearchError) && (
            <div className="place-search-results" aria-live="polite">
              {placeSearchError && <p className="search-message">{placeSearchError}</p>}
              {placeResults.map((place) => (
                <button
                  type="button"
                  className="place-result"
                  key={`${place.id ?? place.label}-${place.longitude}-${place.latitude}`}
                  onClick={() => flyToPlace(place)}
                >
                  <strong>{place.name}</strong>
                  <small>{place.region}</small>
                </button>
              ))}
              <p className="search-attribution">Place search: Photon · © OpenStreetMap contributors</p>
            </div>
          )}
        </form>

        <div className="map-toolbar-actions">
          <button
            type="button"
            className={`toolbar-button${settingsOpen ? " active" : ""}`}
            aria-label="Map layers"
            aria-expanded={settingsOpen}
            onClick={() => {
              setInfoOpen(false);
              setSettingsOpen((open) => !open);
            }}
          >
            <Layers3 size={17} />
            <span>Layers</span>
          </button>
          <button
            type="button"
            className={`toolbar-button${infoOpen ? " active" : ""}`}
            aria-label="Location and mode details"
            aria-expanded={infoOpen}
            onClick={() => {
              setSettingsOpen(false);
              setInfoOpen((open) => !open);
            }}
          >
            <Info size={16} />
            <span>Details</span>
          </button>
        </div>
      </div>

      {/* Navigation rail */}
      <nav className={`navigation-rail navigation-rail-${viewMode}`} aria-label="Map navigation">
        <button type="button" className="nav-action" aria-label="Return to India" title="India" onClick={flyToIndia}>
          <Compass size={17} />
        </button>
        <button type="button" className="nav-action" aria-label="Go to Maharashtra" title="Maharashtra" onClick={flyToMaharashtra}>
          <Crosshair size={17} />
        </button>
        <button type="button" className="nav-action" aria-label="Go to global view" title="Global view" onClick={flyToGlobalView}>
          <Globe2 size={17} />
        </button>
      </nav>

      {/* Info panel */}
      {infoOpen && (
        <aside className="weather-panel" aria-label="Selected location information">
          <div className="panel-title-row">
            <div>
                <p className="eyebrow">{modeDescriptions[viewMode].title}</p>
              <h1>{selectedLocation.name}</h1>
            </div>
            <button
              type="button"
              className="icon-button panel-close"
              aria-label="Close location information"
              onClick={() => setInfoOpen(false)}
            >
              <X size={16} />
            </button>
          </div>
          <p className="location-caption">{selectedLocation.region} · {selectedLocation.code}</p>
          <p className="mode-description">{modeDescriptions[viewMode].description}</p>

          {/* Mode specific info contents */}
          {viewMode === "live" ? (
            <div className="climate-value-panel" aria-live="polite">
              {liveWeatherError && (
                <p className="rainfall-error">
                  Latest weather data could not be refreshed: {liveWeatherError}
                </p>
              )}
              {selectedLiveStation ? (
                <>
                  <strong style={{ fontSize: 20 }}>
                    {selectedLiveStation.temperature_c !== null
                      ? `${selectedLiveStation.temperature_c.toFixed(1)} °C`
                      : "Temperature unavailable"}
                  </strong>
                  <span>{selectedLiveStation.weather_condition || "Condition unavailable"}</span>
                  <span className="data-kind-label">
                    {selectedLiveStation.is_observation
                      ? "Station observation"
                      : "Model-derived analysis · not a station observation"}
                    {" · "}
                    {selectedLiveStation.data_status}
                  </span>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginTop: 8, fontSize: 11 }}>
                    <div>Rainfall: <strong>{selectedLiveStation.rainfall_mm !== null ? `${selectedLiveStation.rainfall_mm} mm` : "Unavailable"}</strong></div>
                    <div>Humidity: <strong>{selectedLiveStation.relative_humidity_percent !== null ? `${selectedLiveStation.relative_humidity_percent}%` : "—"}</strong></div>
                    <div>Wind: <strong>{selectedLiveStation.wind_speed_kmh !== null ? `${selectedLiveStation.wind_speed_kmh} km/h` : "—"}</strong></div>
                    <div>Pressure: <strong>{selectedLiveStation.surface_pressure_hpa !== null ? `${selectedLiveStation.surface_pressure_hpa} hPa` : "—"}</strong></div>
                  </div>
                  <small style={{ marginTop: 8, display: "block" }}>
                    Source: {selectedLiveStation.source_product}
                    <br />
                    {selectedLiveStation.is_observation ? "Observation time" : "Data time"} (UTC):{" "}
                    {selectedLiveStation.observation_time_utc}
                  </small>
                </>
              ) : (
                <span>
                  {liveWeatherLoading
                    ? "Loading latest available weather data…"
                    : liveWeatherRecords.length > 0
                      ? "Select a station in the live strip to inspect its latest available record."
                      : "No live weather records are currently available from the configured providers."}
                </span>
              )}
            </div>
          ) : viewMode === "whatif" ? (
            <div className="climate-value-panel" aria-live="polite">
              <span>
                Choose a geographic scope, baseline, and variables; run the simulation to compare the baseline, scenario, and mapped difference.
              </span>
            </div>
          ) : viewMode === "forecast" ? (
            <div className="climate-value-panel" aria-live="polite">
              {predictionLoading && <span>Generating one-day model prediction…</span>}
              {predictionError && <span className="rainfall-error">{predictionError}</span>}
              {!predictionGrid && !predictionLoading && !predictionError && (
                <span>
                  Select a model variable and an available initialization date, then run the prediction.
                </span>
              )}
              {!predictionGrid && (
                <button
                  type="button"
                  className="prediction-run-btn"
                  onClick={() => {
                    setPredictionError("");
                    setPredictionRunAt(null);
                    setSelectedPredictionCell(null);
                    setPredictionRequested(true);
                  }}
                  disabled={
                    !selectedPredictionModel?.available ||
                    !predictionInitializationDate ||
                    predictionLoading
                  }
                >
                  {predictionLoading ? "Running prediction…" : "Run 1-day prediction"}
                </button>
              )}
              {predictionGrid && (
                <>
                  <span className="data-kind-label">AI model estimate · not an observation</span>
                  {selectedPredictionCell ? (
                    <strong>
                      {selectedPredictionCell.value === null
                        ? "No valid forecast"
                        : `${selectedPredictionCell.value.toFixed(1)} ${predictionGrid.units}`}
                    </strong>
                  ) : (
                    <strong>{predictionGrid.forecast_date} · {predictionGrid.units}</strong>
                  )}
                  <span>
                    {predictionGrid.variable.toUpperCase()} · {predictionGrid.model_name}
                    <br />
                    One-day forecast from {predictionGrid.initialization_date}
                  </span>
                  {predictionGrid.forecast_date < new Date().toISOString().slice(0, 10) && (
                    <small className="prediction-retrospective-note">
                      Target date is in the past relative to today; this is a retrospective model estimate, not a future forecast.
                    </small>
                  )}
                  {selectedPredictionCell ? (
                    <small>
                      Forecast cell {selectedPredictionCell.latitude.toFixed(2)}°N, {selectedPredictionCell.longitude.toFixed(2)}°E
                    </small>
                  ) : (
                    <small>Click the globe to inspect the nearest forecast grid cell.</small>
                  )}
                  <small>Source: {predictionGrid.source}</small>
                  {predictionGrid.evaluation?.test_metrics && (
                    <small>
                      Held-out test metrics: MAE {predictionGrid.evaluation.test_metrics.MAE.toFixed(2)} {predictionGrid.units},
                      {" "}RMSE {predictionGrid.evaluation.test_metrics.RMSE.toFixed(2)} {predictionGrid.units}
                      {" "}({predictionGrid.evaluation.test_metrics.sample_count} samples)
                    </small>
                  )}
                  {predictionRunAt && (
                    <small>Prediction generated: {new Date(predictionRunAt).toLocaleString()}</small>
                  )}
                </>
              )}
            </div>
          ) : temperatureSource !== "none" ? (
            <div className="climate-value-panel" aria-live="polite">
              {temperatureLoading && <span>Reading IMD temperature grid cell…</span>}
              {selectedTemperatureCell && (
                <>
                  <strong>
                    {selectedTemperatureCell.temperature === null
                      ? "Missing observation"
                      : `${selectedTemperatureCell.temperature.toFixed(1)} °C`}
                  </strong>
                  <span>{selectedTemperatureCell.observation_date} · IMD {selectedTemperatureCell.variable}</span>
                  <small>Grid cell {selectedTemperatureCell.latitude.toFixed(1)}°N, {selectedTemperatureCell.longitude.toFixed(1)}°E</small>
                </>
              )}
              {!selectedTemperatureCell && !temperatureLoading && "Click the globe to inspect the nearest 1° temperature cell."}
            </div>
          ) : rainfallSource === "imd" ? (
            <div className="climate-value-panel" aria-live="polite">
              {rainfallCellStatus === "loading" && <span>Reading IMD grid cell…</span>}
              {rainfallCellStatus === "error" && <span className="rainfall-error">{rainfallCellError}</span>}
              {selectedRainfallCell && (
                <>
                  <strong>
                    {selectedRainfallCell.rainfall === null
                      ? "Missing observation"
                      : `${selectedRainfallCell.rainfall.toFixed(1)} mm/day`}
                  </strong>
                  <span>{selectedRainfallCell.observation_date} · IMD 0.25° grid</span>
                  <small>Grid cell {selectedRainfallCell.latitude.toFixed(2)}°N, {selectedRainfallCell.longitude.toFixed(2)}°E</small>
                </>
              )}
              {!selectedRainfallCell && rainfallCellStatus === "idle" && "Click the globe to inspect the nearest IMD grid cell."}
            </div>
          ) : (
            <div className="climate-unavailable">Select a climate layer to inspect coordinates.</div>
          )}
          {viewMode === "whatif" && scenarioScopesError && (
            <p className="scenario-scope-error">{scenarioScopesError}</p>
          )}
          {viewMode === "whatif" && (
            <WhatIfScenarioDock
              scopes={scenarioScopesList}
              selectedScope={scenarioScope}
              onSelectScope={setScenarioScope}
              baselineSource={scenarioBaselineSource}
              onSelectBaselineSource={setScenarioBaselineSource}
              baselineDate={scenarioBaselineDate}
              onSelectBaselineDate={setScenarioBaselineDate}
              variableSelection={scenarioVarSelection}
              onSelectVariableSelection={setScenarioVarSelection}
              rainfallChangePct={rainfallChangePct}
              onChangeRainfallChangePct={setRainfallChangePct}
              tmaxDelta={tmaxDelta}
              onChangeTmaxDelta={setTmaxDelta}
              tminDelta={tminDelta}
              onChangeTminDelta={setTminDelta}
              running={scenarioRunning}
              error={scenarioError}
              onRunScenario={handleRunScenario}
              onReset={handleResetScenario}
              result={scenarioResult}
              viewLayer={scenarioViewLayer}
              onSelectViewLayer={setScenarioViewLayer}
              activeVariable={scenarioActiveVar}
              onSelectActiveVariable={setScenarioActiveVar}
            />
          )}
        </aside>
      )}

      {/* Settings popover */}
      {settingsOpen && (
        <section className="settings-popover" aria-label="Map layers">
          <div className="popover-heading">
            <strong>Map layers & detail</strong>
            <button
              type="button"
              className="icon-button"
              aria-label="Close layers"
              onClick={() => setSettingsOpen(false)}
            >
              <X size={15} />
            </button>
          </div>

          <p className="layer-group-label">Geographical overlays</p>
          <label className="layer-item">
            <input
              type="checkbox"
              checked={boundariesVisible}
              onChange={(event) => setBoundariesVisible(event.target.checked)}
            />
            <span className="layer-swatch boundary-swatch" />
            <span>State & UT boundaries (real borders)</span>
          </label>
          <label className="layer-item">
            <input
              type="checkbox"
              checked={stateLabelsVisible}
              onChange={(event) => setStateLabelsVisible(event.target.checked)}
            />
            <span className="layer-swatch state-label-swatch" />
            <span>State & UT names</span>
          </label>
          <label className="layer-item">
            <input
              type="checkbox"
              checked={cityLabelsVisible}
              onChange={(event) => setCityLabelsVisible(event.target.checked)}
            />
            <span className="layer-swatch city-label-swatch" />
            <span>Cities & towns (zoom-adaptive)</span>
          </label>

          {/* Mode specific layers */}
          {viewMode === "live" ? (
            <>
              <p className="layer-group-label">NASA Satellite Imagery Overlay</p>
              <p className="source-note">
                Satellite imagery is a visual layer; it is not a numerical rainfall observation.
              </p>
              <label className="layer-item">
                <input
                  type="checkbox"
                  checked={satelliteOverlayActive}
                  onChange={(e) => setSatelliteOverlayActive(e.target.checked)}
                />
                <span className="layer-swatch" style={{ background: "#38bdf8" }} />
                <span>Show NASA satellite layer overlay</span>
              </label>

              {satelliteOverlayActive && (
                <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 6 }}>
                  <select
                    className="whatif-select"
                    value={selectedSatelliteLayerId}
                    onChange={(e) => {
                      setSelectedSatelliteLayerId(e.target.value);
                      const layer = satelliteLayersList.find((item) => item.layer_id === e.target.value);
                      if (layer) setSatelliteDate(layer.default_date);
                    }}
                  >
                    {satelliteLayersList.map((l) => (
                      <option key={l.layer_id} value={l.layer_id}>
                        {l.name}
                      </option>
                    ))}
                  </select>
                  <input
                    type="date"
                    className="whatif-input"
                    value={satelliteDate}
                    onChange={(e) => setSatelliteDate(e.target.value)}
                  />
                </div>
              )}
              {satelliteLayersList.length === 0 && (
                <p className="source-note">
                  Satellite layer metadata is unavailable; no imagery overlay can be selected.
                </p>
              )}

              <p className="layer-group-label" style={{ marginTop: 12 }}>Provider Ingestion Status</p>
              {liveMetadataError && <p className="rainfall-error">{liveMetadataError}</p>}
              <div style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 10 }}>
                {liveProviders.map((p) => (
                  <div key={p.provider_id} style={{ display: "flex", justifyContent: "space-between", color: "rgba(255,255,255,0.7)" }}>
                    <span>{p.name.split("(")[0]}</span>
                    <span style={{ color: p.available ? "#4ade80" : "#fbbf24" }}>
                      {p.data_status === "credentials_required" ? "Gated (Registration req.)" : p.data_status}
                    </span>
                  </div>
                ))}
              </div>
            </>
          ) : viewMode === "forecast" ? (
            <>
              <p className="layer-group-label">One-day prediction</p>
              <div className="rainfall-layer-options">
                {(predictionCatalog?.models ?? []).map((model) => {
                  const Icon = model.variable === "rainfall" ? Droplets : Thermometer;
                  return (
                    <button
                      type="button"
                      key={model.variable}
                      className={`rainfall-layer-option${predictionVariable === model.variable ? " selected" : ""}`}
                      aria-pressed={predictionVariable === model.variable}
                      disabled={!model.available}
                      onClick={() => selectPredictionVariable(model.variable)}
                    >
                      <Icon size={15} aria-hidden="true" />
                      <span>{model.label}</span>
                      <small>
                        {model.available && model.test_metrics
                          ? `${model.model_name} · native ${model.spatial_resolution_degrees}° · ${model.units} · test RMSE ${model.test_metrics.RMSE.toFixed(2)}`
                          : model.error ?? "No trained model installed"}
                      </small>
                    </button>
                  );
                })}
              </div>
              {predictionCatalogError && (
                <p className="rainfall-error">{predictionCatalogError}</p>
              )}
            </>
          ) : viewMode === "historical" ? (
            <>
              <p className="layer-group-label">Climate fields</p>
              <div className="rainfall-layer-options">
                <button
                  type="button"
                  className={`rainfall-layer-option${temperatureSource === "tmax" ? " selected" : ""}`}
                  aria-pressed={temperatureSource === "tmax"}
                  disabled={temperatureDates.tmax.length === 0}
                  onClick={() => selectTemperatureSource(temperatureSource === "tmax" ? "none" : "tmax")}
                >
                  <Thermometer size={15} aria-hidden="true" />
                  <span>Maximum temperature</span>
                  <small>
                    {temperatureDates.tmax.length > 0
                      ? `IMD · 1° · °C · ${temperatureDates.tmax.length} dates`
                      : "No local IMD Tmax file"}
                  </small>
                </button>
                <button
                  type="button"
                  className={`rainfall-layer-option${temperatureSource === "tmin" ? " selected" : ""}`}
                  aria-pressed={temperatureSource === "tmin"}
                  disabled={temperatureDates.tmin.length === 0}
                  onClick={() => selectTemperatureSource(temperatureSource === "tmin" ? "none" : "tmin")}
                >
                  <Thermometer size={15} aria-hidden="true" />
                  <span>Minimum temperature</span>
                  <small>
                    {temperatureDates.tmin.length > 0
                      ? `IMD · 1° · °C · ${temperatureDates.tmin.length} dates`
                      : "No local IMD Tmin file"}
                  </small>
                </button>
                <button
                  type="button"
                  className={`rainfall-layer-option${rainfallSource === "imd" ? " selected" : ""}`}
                  aria-pressed={rainfallSource === "imd"}
                  disabled={imdDates.length === 0}
                  onClick={() => selectRainfallSource(rainfallSource === "imd" ? "none" : "imd")}
                >
                  <Droplets size={15} aria-hidden="true" />
                  <span>IMD rainfall</span>
                  <small>
                    {imdDates.length > 0
                      ? `0.25° · mm/day · ${imdDates.length} dates`
                      : rainfallError || "No local IMD file"}
                  </small>
                </button>
                <button
                  type="button"
                  className={`rainfall-layer-option${rainfallSource === "nasa" ? " selected" : ""}`}
                  aria-pressed={rainfallSource === "nasa"}
                  disabled={nasaAvailabilityStatus !== "available"}
                  onClick={() => selectRainfallSource(rainfallSource === "nasa" ? "none" : "nasa")}
                >
                  <Cloud size={15} aria-hidden="true" />
                  <span>NASA IMERG imagery</span>
                  <small>
                    {nasaAvailabilityStatus === "available" && nasaAvailability
                      ? `GIBS image · ${nasaAvailability.observation_date_end}`
                      : "NASA GIBS unavailable"}
                  </small>
                </button>
              </div>

              {rainfallLoading && <p className="source-note">Loading IMD rainfall observation…</p>}
              {rainfallError && <p className="rainfall-error">{rainfallError}</p>}
              {temperatureError && <p className="rainfall-error">{temperatureError}</p>}
              {rainfallSource === "nasa" && (
                <p className="source-note">
                  NASA GIBS satellite imagery is visual-only and does not provide queryable rainfall grid values.
                </p>
              )}

              <div className="unavailable-climate-layers">
                {unavailableClimateLayers.map(({ label, source, sourceUrl, sourceLabel, icon: Icon }) => (
                  <div className="unavailable-climate-item" key={label}>
                    <button type="button" className="unavailable-climate-layer" disabled title={`${label} unavailable: ${source}`}>
                      <Icon size={15} aria-hidden="true" />
                      <span>{label}</span>
                      <small>{source}</small>
                    </button>
                    <a className="unavailable-climate-reference" href={sourceUrl} target="_blank" rel="noreferrer">
                      {sourceLabel}
                    </a>
                  </div>
                ))}
              </div>
            </>
          ) : null}

          <div className="geo-attribution">
            Globe: Esri World Imagery (realistic 3D Earth) · Boundaries: geoBoundaries IND-ADM1 (CC BY 4.0) · Telemetry: IMD & Open-Meteo · Satellite: NASA GIBS.
          </div>
        </section>
      )}

      {/* Mode Bottom Dock / Controls */}
      {viewMode === "live" && (
        <LiveTelemetryBar
          records={liveWeatherRecords}
          loading={liveWeatherLoading}
          error={liveWeatherError}
          providers={liveProviders}
          selectedStationId={selectedLiveStation?.location_id ?? null}
          onSelectStation={handleSelectLiveStation}
        />
      )}

      {(viewMode === "historical" || viewMode === "forecast") && (
        <footer className="timeline-control" aria-label="Climate observation timeline">
          <div className="timeline-status">
            <CalendarDays size={15} aria-hidden="true" />
            <span>{viewMode === "forecast" ? "Model prediction" : "Historical observations"}</span>
            <small>
              {viewMode === "forecast"
                ? predictionInitializationDates.length > 0
                  ? `${predictionInitializationDates.length} initialization dates`
                  : "No model dates available"
                : rainfallSource === "imd"
                ? `${imdDates.length} IMD dates`
                : temperatureSource !== "none"
                  ? `${temperatureDates[temperatureSource].length} IMD dates`
                : rainfallSource === "nasa" && nasaAvailability
                  ? "NASA GIBS imagery"
                  : "Select a rainfall source"}
            </small>
          </div>
          <input
            aria-label={viewMode === "forecast" ? "Select model date" : "Select observation date"}
            type="date"
            min={
              viewMode === "forecast"
                ? predictionInitializationDates[0]
                : rainfallSource === "imd"
                ? imdDates[0]
                : temperatureSource !== "none"
                  ? temperatureDates[temperatureSource][0]
                : rainfallSource === "nasa"
                  ? nasaAvailability?.observation_date_start
                  : undefined
            }
            max={
              viewMode === "forecast"
                ? predictionInitializationDates[predictionInitializationDates.length - 1]
                : rainfallSource === "imd"
                ? imdDates[imdDates.length - 1]
                : temperatureSource !== "none"
                  ? temperatureDates[temperatureSource][temperatureDates[temperatureSource].length - 1]
                : rainfallSource === "nasa"
                  ? nasaAvailability?.observation_date_end
                  : undefined
            }
            value={viewMode === "forecast" ? predictionInitializationDate : rainfallDate}
            disabled={
              viewMode === "forecast"
                ? predictionInitializationDates.length === 0
                : (temperatureSource !== "none" && temperatureDates[temperatureSource].length === 0) ||
                  (rainfallSource === "none" && temperatureSource === "none") ||
                  (rainfallSource === "imd" && imdDates.length === 0)
            }
            onChange={(event) => {
              const nextDate = event.target.value;
              if (viewMode === "forecast") {
                if (!predictionInitializationDates.includes(nextDate)) {
                  setPredictionError("Choose an initialization date available for this model.");
                  return;
                }
                setPredictionGrid(null);
                setSelectedPredictionCell(null);
                setPredictionRequested(false);
                setPredictionRunAt(null);
                setPredictionError("");
                setPredictionInitializationDate(nextDate);
                return;
              }
              if (
                rainfallSource === "nasa" &&
                nasaAvailability &&
                !isNasaDateAvailable(nextDate, nasaAvailability.available_time_intervals)
              ) {
                setRainfallError("NASA GIBS does not advertise an image for that date.");
                return;
              }
              setRainfallDate(nextDate);
              setRainfallGrid(null);
              setTemperatureGrid(null);
              setRainfallError("");
              setTemperatureError("");
              setRainfallLoading(rainfallSource === "imd");
              setTemperatureLoading(temperatureSource !== "none");
              setSelectedRainfallCell(null);
              setSelectedTemperatureCell(null);
            }}
          />
          <span className="timeline-source">
            {viewMode === "forecast"
              ? `${selectedPredictionModel?.model_name ?? "No model"} · 1-day prediction`
              : rainfallSource === "imd"
              ? "IMD daily observations"
              : temperatureSource !== "none"
                ? `IMD ${temperatureSource === "tmax" ? "Tmax" : "Tmin"} daily observations`
              : rainfallSource === "nasa"
                ? "NASA GIBS daily imagery"
                : "No source selected"}
          </span>
        </footer>
      )}

      {/* Legends */}
      {viewMode === "whatif" && scenarioResult && (
        <aside className="rainfall-legend" aria-label="What-If layer legend">
          <strong>
            {scenarioViewLayer === "difference"
              ? `${scenarioActiveVar === "rainfall" ? "Rainfall" : "Temperature"} Difference (Scenario - Baseline)`
              : `${scenarioActiveVar === "rainfall" ? "Rainfall" : "Temperature"} (${scenarioViewLayer})`}
          </strong>
          {scenarioViewLayer === "difference" ? (
            <>
              <div className="diff-legend-gradient" />
              <div className="diff-legend-labels">
                <span>{scenarioActiveVar === "rainfall" ? "-25 mm (Deficit)" : "-4 °C (Cooling)"}</span>
                <span>0</span>
                <span>{scenarioActiveVar === "rainfall" ? "+25 mm (Surplus)" : "+4 °C (Warming)"}</span>
              </div>
            </>
          ) : scenarioActiveVar === "rainfall" ? (
            <div className="rainfall-legend-scale" aria-hidden="true">
              {rainfallColorStops.slice(1).map((stop) => (
                <div key={stop.value} className="legend-step">
                  <span
                    className="legend-color"
                    style={{ background: `rgb(${stop.color[0]}, ${stop.color[1]}, ${stop.color[2]})` }}
                  />
                  <span>{stop.value} mm</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="rainfall-legend-scale" aria-hidden="true">
              {temperatureColorStops.map((stop) => (
                <div key={stop.value} className="legend-step">
                  <span
                    className="legend-color"
                    style={{ background: `rgb(${stop.color[0]}, ${stop.color[1]}, ${stop.color[2]})` }}
                  />
                  <span>{stop.value} °C</span>
                </div>
              ))}
            </div>
          )}
          <small>
            Scope: {scenarioResult.geographic_scope_name} · Baseline: {scenarioResult.parameters.baseline_date}
          </small>
        </aside>
      )}

      {(viewMode === "historical" || viewMode === "forecast") &&
        ((viewMode === "historical" && (rainfallSource !== "none" || temperatureSource !== "none")) ||
          (viewMode === "forecast" && predictionGrid !== null)) && (
          <aside className="rainfall-legend" aria-label="Climate layer legend">
            <strong>
              {viewMode === "forecast" && predictionGrid
                ? `${predictionGrid.variable === "rainfall" ? "Rainfall" : "Temperature"} prediction · ${predictionGrid.units}`
                : temperatureSource !== "none"
                ? `IMD ${temperatureSource === "tmax" ? "Tmax" : "Tmin"} · °C`
                : "IMD daily rainfall · mm/day"}
            </strong>

            <div className="rainfall-legend-scale" aria-hidden="true">
              {(viewMode === "forecast" && predictionGrid?.variable !== "rainfall") ||
              (viewMode === "historical" && temperatureSource !== "none")
                ? temperatureColorStops.map((stop) => (
                    <div key={stop.value} className="legend-step">
                      <span
                        className="legend-color"
                        style={{ background: `rgb(${stop.color[0]}, ${stop.color[1]}, ${stop.color[2]})` }}
                      />
                      <span>{stop.value} °C</span>
                    </div>
                  ))
                : rainfallColorStops.slice(1).map((stop) => (
                    <div key={stop.value} className="legend-step">
                      <span
                        className="legend-color"
                        style={{ background: `rgb(${stop.color[0]}, ${stop.color[1]}, ${stop.color[2]})` }}
                      />
                      <span>{stop.value} mm</span>
                    </div>
                  ))}
            </div>
          </aside>
        )}
    </main>
  );
}
