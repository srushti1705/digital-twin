import { memo, useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  CalendarDays,
  Cloud,
  Compass,
  Crosshair,
  Droplets,
  Globe2,
  Layers3,
  Search,
  Thermometer,
  X,
} from "lucide-react";
import { CameraFlyTo, ImageryLayer, Viewer } from "resium";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import "./App.css";
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
} from "./services/climateApi";
import type {
  ImdRainfallCell,
  ImdRainfallGrid,
  ImdTemperatureCell,
  ImdTemperatureGrid,
  NasaRainfallAvailability,
} from "./services/climateApi";

const INDIA_VIEW = { longitude: 78.9, latitude: 20.5, height: 3_600_000 };
const MAHARASHTRA_VIEW = { longitude: 75.7, latitude: 19.2, height: 850_000 };
const GLOBAL_VIEW = { longitude: 15, latitude: 15, height: 12_500_000 };
const STATE_BOUNDARY_MAX_HEIGHT = 7_000_000;

type Location = { code: string; name: string; region: string };
type ViewMode = "historical" | "forecast";
type ViewerHandle = { cesiumElement?: Cesium.Viewer };
type ApiStatus = "checking" | "healthy" | "unavailable";
type RainfallSource = "none" | "imd" | "nasa";
type TemperatureSource = "none" | "tmax" | "tmin";
type RainfallCellStatus = "idle" | "loading" | "unavailable" | "error";

const unavailableClimateLayers = [
  { label: "Satellite clouds", source: "No satellite feed connected", icon: Cloud },
  { label: "Cyclone information", source: "No official feed connected", icon: Crosshair },
] as const;

let indiaBoundariesPromise: Promise<Cesium.GeoJsonDataSource> | undefined;

function loadIndiaBoundaries() {
  if (!indiaBoundariesPromise) {
    indiaBoundariesPromise = Cesium.GeoJsonDataSource.load("/india-states.geojson").catch(
      (error: unknown) => {
        indiaBoundariesPromise = undefined;
        throw error;
      },
    );
  }
  return indiaBoundariesPromise;
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}

function configureViewerScene(viewer: Cesium.Viewer) {
  const controls = viewer.scene.screenSpaceCameraController;
  viewer.scene.globe.enableLighting = true;
  viewer.scene.globe.showGroundAtmosphere = true;
  viewer.scene.globe.depthTestAgainstTerrain = false;
  viewer.scene.fog.enabled = true;
  if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = true;
  if (viewer.scene.skyBox) viewer.scene.skyBox.show = false;
  viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#03101c");
  controls.enableInputs = true;
  controls.minimumZoomDistance = 500;
  controls.maximumZoomDistance = 30_000_000;
  controls.zoomFactor = 3;
  controls.inertiaSpin = 0.85;
  controls.inertiaTranslate = 0.85;
  controls.inertiaZoom = 0.8;
  controls.rotateEventTypes = Cesium.CameraEventType.LEFT_DRAG;
  controls.tiltEventTypes = Cesium.CameraEventType.RIGHT_DRAG;
  controls.translateEventTypes = [];
  controls.zoomEventTypes = Cesium.CameraEventType.PINCH;

  const canvas = viewer.scene.canvas;
  let panPosition: Cesium.Cartesian2 | undefined;
  const zoomAtCursor = (event: WheelEvent) => {
    event.preventDefault();
    const camera = viewer.camera;
    const currentHeight = camera.positionCartographic.height;
    const zoomFactor = Math.exp(event.deltaY * 0.001);
    const targetHeight = Cesium.Math.clamp(
      currentHeight * zoomFactor,
      controls.minimumZoomDistance,
      controls.maximumZoomDistance,
    );
    const distance = Math.abs(targetHeight - currentHeight);
    if (distance === 0) return;

    const bounds = canvas.getBoundingClientRect();
    const pointer = new Cesium.Cartesian2(
      event.clientX - bounds.left,
      event.clientY - bounds.top,
    );
    const anchorBefore = camera.pickEllipsoid(pointer, viewer.scene.globe.ellipsoid);
    if (targetHeight > currentHeight) camera.zoomOut(distance);
    else camera.zoomIn(distance);

    if (anchorBefore) {
      const anchorAfter = camera.pickEllipsoid(pointer, viewer.scene.globe.ellipsoid);
      if (anchorAfter) {
        const correction = Cesium.Cartesian3.subtract(
          anchorBefore,
          anchorAfter,
          new Cesium.Cartesian3(),
        );
        const correctionDistance = Cesium.Cartesian3.magnitude(correction);
        if (correctionDistance > 0) {
          camera.move(
            Cesium.Cartesian3.normalize(correction, correction),
            correctionDistance,
          );
        }
      }
    }
    viewer.scene.requestRender();
  };
  const startPan = (event: MouseEvent) => {
    if (event.button !== 1 && !(event.button === 0 && event.shiftKey)) return;
    panPosition = new Cesium.Cartesian2(event.clientX, event.clientY);
    event.preventDefault();
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

function formatCoordinate(value: number, positive: string, negative: string) {
  return `${Math.abs(value).toFixed(2)}°${value < 0 ? negative : positive}`;
}

const imageryProvider = new Cesium.UrlTemplateImageryProvider({
  url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  credit: "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
});

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

function createRainfallRaster(grid: ImdRainfallGrid) {
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
  return canvas.toDataURL("image/png");
}

function createTemperatureRaster(grid: ImdTemperatureGrid) {
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
  return canvas.toDataURL("image/png");
}

const initialCameraDestination = Cesium.Cartesian3.fromDegrees(
  INDIA_VIEW.longitude,
  INDIA_VIEW.latitude,
  INDIA_VIEW.height,
);
const initialCameraOrientation = {
  heading: 0,
  pitch: Cesium.Math.toRadians(-78),
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
        <ImageryLayer imageryProvider={imageryProvider} />
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
  const rainfallLayerRef = useRef<Cesium.ImageryLayer | null>(null);
  const rainfallSourceRef = useRef<RainfallSource>("none");
  const temperatureSourceRef = useRef<TemperatureSource>("none");
  const rainfallDateRef = useRef("");
  const rainfallCellControllerRef = useRef<AbortController | null>(null);
  const boundaryPreferenceRef = useRef(true);
  const searchControllerRef = useRef<AbortController | null>(null);
  const [viewerReady, setViewerReady] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<Location>({
    code: "IN",
    name: "India",
    region: "Indian subcontinent",
  });
  const [boundaryStatus, setBoundaryStatus] = useState("Loading boundaries");
  const [boundariesVisible, setBoundariesVisible] = useState(true);
  const [imageryVisible, setImageryVisible] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("historical");
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [apiStatusMessage, setApiStatusMessage] = useState("Checking climate API");
  const [rainfallSource, setRainfallSource] = useState<RainfallSource>("none");
  const [temperatureSource, setTemperatureSource] =
    useState<TemperatureSource>("none");
  const [rainfallDate, setRainfallDate] = useState("");
  const [imdDates, setImdDates] = useState<string[]>([]);
  const [rainfallGrid, setRainfallGrid] = useState<ImdRainfallGrid | null>(null);
  const [temperatureDates, setTemperatureDates] = useState<
    Record<"tmax" | "tmin", string[]>
  >({ tmax: [], tmin: [] });
  const [temperatureGrid, setTemperatureGrid] =
    useState<ImdTemperatureGrid | null>(null);
  const [temperatureLoading, setTemperatureLoading] = useState(false);
  const [temperatureError, setTemperatureError] = useState("");
  const [rainfallLoading, setRainfallLoading] = useState(false);
  const [rainfallError, setRainfallError] = useState("");
  const [nasaAvailability, setNasaAvailability] =
    useState<NasaRainfallAvailability | null>(null);
  const [nasaAvailabilityStatus, setNasaAvailabilityStatus] =
    useState<"checking" | "available" | "unavailable">("checking");
  const [selectedRainfallCell, setSelectedRainfallCell] =
    useState<ImdRainfallCell | null>(null);
  const [selectedTemperatureCell, setSelectedTemperatureCell] =
    useState<ImdTemperatureCell | null>(null);
  const [rainfallCellStatus, setRainfallCellStatus] =
    useState<RainfallCellStatus>("idle");
  const [rainfallCellError, setRainfallCellError] = useState("");
  const [placeQuery, setPlaceQuery] = useState("");
  const [placeResults, setPlaceResults] = useState<GeocodedPlace[]>([]);
  const [placeSearchError, setPlaceSearchError] = useState("");
  const [placeSearchBusy, setPlaceSearchBusy] = useState(false);

  useEffect(() => {
    rainfallSourceRef.current = rainfallSource;
    temperatureSourceRef.current = temperatureSource;
    rainfallDateRef.current = rainfallDate;
  }, [rainfallDate, rainfallSource, temperatureSource]);

  const setViewerRef = useCallback((handle: ViewerHandle | null) => {
    viewerRef.current = handle;
    if (handle?.cesiumElement) setViewerReady(true);
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
        pitch: Cesium.Math.toRadians(-78),
        roll: 0,
      },
      duration: 1.8,
    });
  }, []);

  const flyToGlobalView = useCallback(() => {
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewer) return;
    setSelectedLocation({ code: "WORLD", name: "World", region: "Global view" });
    setInfoOpen(false);
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(
        GLOBAL_VIEW.longitude,
        GLOBAL_VIEW.latitude,
        GLOBAL_VIEW.height,
      ),
      orientation: {
        heading: 0,
        pitch: Cesium.Math.toRadians(-90),
        roll: 0,
      },
      duration: 1.8,
    });
  }, []);

  const flyToMaharashtra = useCallback(() => {
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewer) return;
    const state = boundarySourceRef.current?.entities.values.find((entity) => {
      const properties = entity.properties?.getValue(Cesium.JulianDate.now());
      return properties?.iso_code === "IN-MH" || properties?.STATE_CODE === "IN-MH";
    });

    setSelectedLocation({
      code: "IN-MH",
      name: "Maharashtra",
      region: "West",
    });
    setInfoOpen(true);
    if (state) {
      void viewer.flyTo(state, {
        duration: 1.8,
        offset: new Cesium.HeadingPitchRange(0, -0.75, 850_000),
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
        pitch: Cesium.Math.toRadians(-65),
        roll: 0,
      },
      duration: 1.8,
    });
  }, []);

  const flyToPlace = useCallback((place: GeocodedPlace) => {
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewer) return;

    if (place.extent) {
      viewer.camera.flyTo({
        destination: Cesium.Rectangle.fromDegrees(...place.extent),
        duration: 1.8,
      });
    } else {
      const placeKind = place.kind.toLowerCase();
      const height = placeKind.includes("country")
        ? 4_000_000
        : placeKind.includes("state") || placeKind.includes("region")
          ? 1_200_000
          : 300_000;
      viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(
          place.longitude,
          place.latitude,
          height,
        ),
        orientation: {
          heading: 0,
          pitch: Cesium.Math.toRadians(-62),
          roll: 0,
        },
        duration: 1.8,
      });
    }

    setSelectedLocation({
      code: place.id ? `OSM ${place.id}` : "Search result",
      name: place.name,
      region: place.region,
    });
    setInfoOpen(true);
    setPlaceQuery(place.label);
    setPlaceResults([]);
    setPlaceSearchError("");
  }, []);

  const inspectCoordinates = useCallback(
    (latitude: number, longitude: number, preserveSelectedLocation = false) => {
      if (!preserveSelectedLocation) {
        setSelectedLocation({
          code: `${latitude.toFixed(2)}, ${longitude.toFixed(2)}`,
          name: `${formatCoordinate(latitude, "N", "S")}, ${formatCoordinate(longitude, "E", "W")}`,
          region: "Map coordinates",
        });
      }
      setInfoOpen(true);
      setSelectedRainfallCell(null);
      setSelectedTemperatureCell(null);
      setRainfallCellError("");
      rainfallCellControllerRef.current?.abort();

      const source = rainfallSourceRef.current;
      const temperature = temperatureSourceRef.current;
      const observationDate = rainfallDateRef.current;
      if (!observationDate) {
        setRainfallCellStatus("unavailable");
        return;
      }

      const controller = new AbortController();
      rainfallCellControllerRef.current = controller;
      setRainfallCellStatus("loading");
      const cellRequest =
        temperature === "tmax" || temperature === "tmin"
          ? fetchImdTemperatureCell(
              temperature,
              observationDate,
              latitude,
              longitude,
              controller.signal,
            ).then((cell) => {
              if (rainfallCellControllerRef.current !== controller) return;
              setSelectedTemperatureCell(cell);
              setRainfallCellStatus("idle");
            })
          : source === "imd"
            ? fetchImdRainfallCell(
                observationDate,
                latitude,
                longitude,
                controller.signal,
              ).then((cell) => {
                if (rainfallCellControllerRef.current !== controller) return;
                setSelectedRainfallCell(cell);
                setRainfallCellStatus("idle");
              })
            : Promise.resolve().then(() => setRainfallCellStatus("unavailable"));
      void cellRequest
        .catch((error: unknown) => {
          if (isAbortError(error)) return;
          if (rainfallCellControllerRef.current !== controller) return;
          setRainfallCellError(
            error instanceof Error ? error.message : "Could not retrieve this grid cell.",
          );
          setRainfallCellStatus("error");
        });
    },
    [],
  );

  const selectRainfallSource = useCallback(
    (source: RainfallSource) => {
      rainfallCellControllerRef.current?.abort();
      setSelectedRainfallCell(null);
      setRainfallCellStatus("idle");
      setRainfallCellError("");
      setRainfallError("");
      setTemperatureError("");
      setRainfallGrid(null);
      setTemperatureGrid(null);
      setRainfallLoading(source === "imd" && viewMode === "historical");
      setTemperatureLoading(false);
      setRainfallSource(source);
      setTemperatureSource("none");
      setSelectedTemperatureCell(null);
      if (source === "imd" && imdDates.length > 0) {
        setRainfallDate((current) =>
          imdDates.includes(current) ? current : imdDates[imdDates.length - 1],
        );
      } else if (source === "nasa" && nasaAvailability) {
        setRainfallDate(nasaAvailability.observation_date_end);
      }
    },
    [imdDates, nasaAvailability, viewMode],
  );

  const selectTemperatureSource = useCallback(
    (source: TemperatureSource) => {
      rainfallCellControllerRef.current?.abort();
      setSelectedRainfallCell(null);
      setSelectedTemperatureCell(null);
      setRainfallCellStatus("idle");
      setRainfallCellError("");
      setRainfallError("");
      setTemperatureError("");
      setRainfallGrid(null);
      setTemperatureGrid(null);
      setRainfallLoading(false);
      setTemperatureLoading(source !== "none" && viewMode === "historical");
      setRainfallSource("none");
      setTemperatureSource(source);
      if (source !== "none") {
        const dates = temperatureDates[source];
        if (dates.length > 0) {
          setRainfallDate((current) =>
            dates.includes(current) ? current : dates[dates.length - 1],
          );
        }
      }
    },
    [temperatureDates, viewMode],
  );

  const submitPlaceSearch = useCallback(async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const query = placeQuery.trim();
    if (query.length < 2) {
      setPlaceResults([]);
      setPlaceSearchError("Enter at least two characters to search.");
      return;
    }

    searchControllerRef.current?.abort();
    const controller = new AbortController();
    searchControllerRef.current = controller;
    setPlaceSearchBusy(true);
    setPlaceSearchError("");

    try {
      const results = await searchPlaces(query, controller.signal);
      setPlaceResults(results);
      if (results.length === 0) setPlaceSearchError("No matching places found.");
    } catch (error) {
      if (isAbortError(error)) return;
      console.error("Place search failed:", error);
      setPlaceSearchError(
        error instanceof Error ? error.message : "Place search failed unexpectedly.",
      );
    } finally {
      if (searchControllerRef.current === controller) {
        searchControllerRef.current = null;
        setPlaceSearchBusy(false);
      }
    }
  }, [placeQuery]);

  useEffect(() => {
    const viewer = viewerRef.current?.cesiumElement;
    if (!viewerReady || !viewer) return;

    const removeCursorZoom = configureViewerScene(viewer);

    let active = true;
    let removeSelectionListener: (() => void) | undefined;
    let removeCameraListener: (() => void) | undefined;
    let coordinatePicker: Cesium.ScreenSpaceEventHandler | undefined;

    const loadBoundaries = async () => {
      try {
        const source = await loadIndiaBoundaries();
        if (!active) return;
        if (!viewer.dataSources.contains(source)) {
          await viewer.dataSources.add(source);
        }
        boundarySourceRef.current = source;
        source.entities.values.forEach((entity) => {
          if (!entity.polygon) return;
          entity.polygon.material = new Cesium.ColorMaterialProperty(
            Cesium.Color.fromCssColorString("#b6dce5").withAlpha(0.015),
          );
          entity.polygon.outline = new Cesium.ConstantProperty(true);
          entity.polygon.outlineColor = new Cesium.ConstantProperty(
            Cesium.Color.fromCssColorString("#b6dce5").withAlpha(0.5),
          );
          entity.polygon.outlineWidth = new Cesium.ConstantProperty(1);
        });

        const onSelectionChanged = (selected: Cesium.Entity | undefined) => {
          if (!selected?.properties) return;
          const properties = selected.properties.getValue(Cesium.JulianDate.now());
          const code = properties?.iso_code ?? properties?.STATE_CODE ?? selected.id;
          const name =
            properties?.state_name ??
            properties?.STATE_NAME ??
            properties?.ST_NM ??
            "Selected state";
          const region = properties?.REGION ?? "India";
          if (typeof code === "string") {
            setSelectedLocation({
              code,
              name: String(name),
              region: String(region),
            });
            setInfoOpen(true);
          }
        };
        viewer.selectedEntityChanged.addEventListener(onSelectionChanged);
        removeSelectionListener = () =>
          viewer.selectedEntityChanged.removeEventListener(onSelectionChanged);
        const updateBoundaryScale = () => {
          source.show =
            boundaryPreferenceRef.current &&
            viewer.camera.positionCartographic.height <= STATE_BOUNDARY_MAX_HEIGHT;
          viewer.scene.requestRender();
        };
        viewer.camera.moveEnd.addEventListener(updateBoundaryScale);
        removeCameraListener = () =>
          viewer.camera.moveEnd.removeEventListener(updateBoundaryScale);

        coordinatePicker = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
        coordinatePicker.setInputAction(
          (movement: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
            const position = viewer.camera.pickEllipsoid(
              movement.position,
              viewer.scene.globe.ellipsoid,
            );
            if (!position) return;
            const coordinate = Cesium.Cartographic.fromCartesian(position);
            const latitude = Cesium.Math.toDegrees(coordinate.latitude);
            const longitude = Cesium.Math.toDegrees(coordinate.longitude);
            const picked = viewer.scene.pick(movement.position);
            inspectCoordinates(
              latitude,
              longitude,
              picked?.id instanceof Cesium.Entity,
            );
          },
          Cesium.ScreenSpaceEventType.LEFT_CLICK,
        );
        coordinatePicker.setInputAction(
          (movement: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
            const position = viewer.camera.pickEllipsoid(
              movement.position,
              viewer.scene.globe.ellipsoid,
            );
            if (!position) return;
            const coordinate = Cesium.Cartographic.fromCartesian(position);
            const latitude = Cesium.Math.toDegrees(coordinate.latitude);
            const longitude = Cesium.Math.toDegrees(coordinate.longitude);
            inspectCoordinates(latitude, longitude);
            const height = Cesium.Math.clamp(
              viewer.camera.positionCartographic.height * 0.45,
              120_000,
              2_500_000,
            );
            viewer.camera.flyTo({
              destination: Cesium.Cartesian3.fromDegrees(longitude, latitude, height),
              orientation: {
                heading: viewer.camera.heading,
                pitch: Cesium.Math.toRadians(-65),
                roll: 0,
              },
              duration: 1.2,
            });
          },
          Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK,
        );
        viewer.screenSpaceEventHandler.removeInputAction(
          Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK,
        );

        updateBoundaryScale();
        viewer.scene.requestRender();
        setBoundaryStatus("State boundaries ready");
      } catch (error) {
        console.error("Failed to load India state boundaries:", error);
        if (active) setBoundaryStatus("Boundary data unavailable");
      }
    };

    void loadBoundaries();
    return () => {
      active = false;
      removeSelectionListener?.();
      removeCameraListener?.();
      coordinatePicker?.destroy();
      removeCursorZoom();
    };
  }, [inspectCoordinates, viewerReady]);

  useEffect(() => {
    const viewer = viewerRef.current?.cesiumElement;
    const layer = viewer?.imageryLayers.get(0);
    if (viewer && layer) {
      layer.show = imageryVisible;
      viewer.scene.requestRender();
    }
  }, [imageryVisible, viewerReady]);

  useEffect(() => {
    boundaryPreferenceRef.current = boundariesVisible;
    const viewer = viewerRef.current?.cesiumElement;
    if (viewer && boundarySourceRef.current) {
      boundarySourceRef.current.show =
        boundariesVisible &&
        viewer.camera.positionCartographic.height <= STATE_BOUNDARY_MAX_HEIGHT;
      viewer.scene.requestRender();
    }
  }, [boundariesVisible, boundaryStatus]);

  useEffect(() => {
    const source = boundarySourceRef.current;
    if (!source) return;
    source.entities.values.forEach((entity) => {
      if (!entity.polygon) return;
      const properties = entity.properties?.getValue(Cesium.JulianDate.now());
      const code = properties?.iso_code ?? properties?.STATE_CODE ?? entity.id;
      const selected = code === selectedLocation.code;
      entity.polygon.material = new Cesium.ColorMaterialProperty(
        (selected ? Cesium.Color.fromCssColorString("#65e4ef") : Cesium.Color.fromCssColorString("#b6dce5"))
          .withAlpha(selected ? 0.18 : 0.015),
      );
      entity.polygon.outlineColor = new Cesium.ConstantProperty(
        (selected ? Cesium.Color.fromCssColorString("#8ff2f6") : Cesium.Color.fromCssColorString("#b6dce5"))
          .withAlpha(selected ? 0.95 : 0.5),
      );
      entity.polygon.outlineWidth = new Cesium.ConstantProperty(selected ? 2 : 1);
    });
    viewerRef.current?.cesiumElement?.scene.requestRender();
  }, [selectedLocation, boundaryStatus]);

  useEffect(() => {
    const controller = new AbortController();
    void fetchApiHealth(controller.signal)
      .then(({ status }) => {
        setApiStatus(status === "healthy" ? "healthy" : "unavailable");
        setApiStatusMessage(
          status === "healthy" ? "Backend health endpoint is healthy" : `Backend status: ${status}`,
        );
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) return;
        console.error("Climate API health check failed:", error);
        setApiStatus("unavailable");
        setApiStatusMessage(
          error instanceof Error ? error.message : "Climate API health check failed.",
        );
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    void fetchImdRainfallDates(controller.signal)
      .then((response) => {
        if (!active) return;
        setImdDates(response.available_dates);
        if (response.available_dates.length === 0) {
          setRainfallError("No IMD rainfall files are installed on the backend.");
          return;
        }
        const latestDate = response.available_dates[response.available_dates.length - 1];
        setRainfallDate(latestDate);
        setRainfallLoading(true);
        setRainfallSource("imd");
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) return;
        console.error("Could not load IMD rainfall dates:", error);
        if (active) {
          setRainfallError(
            error instanceof Error ? error.message : "IMD rainfall dates are unavailable.",
          );
        }
      });
    void fetchNasaRainfallAvailability(controller.signal)
      .then((availability) => {
        if (!active) return;
        setNasaAvailability(availability);
        setNasaAvailabilityStatus("available");
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) return;
        console.error("Could not load NASA GIBS availability:", error);
        if (active) setNasaAvailabilityStatus("unavailable");
      });
    for (const variable of ["tmax", "tmin"] as const) {
      void fetchImdTemperatureDates(variable, controller.signal)
        .then((response) => {
          if (!active) return;
          setTemperatureDates((current) => ({
            ...current,
            [variable]: response.available_dates,
          }));
        })
        .catch((error: unknown) => {
          if (isAbortError(error)) return;
          console.error(`Could not load IMD ${variable} dates:`, error);
          if (active) {
            setTemperatureError(
              error instanceof Error ? error.message : `IMD ${variable} dates are unavailable.`,
            );
          }
        });
    }
    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  useEffect(() => {
    if (
      (temperatureSource !== "tmax" && temperatureSource !== "tmin") ||
      !rainfallDate ||
      viewMode !== "historical"
    ) {
      return;
    }

    const controller = new AbortController();
    void fetchImdTemperatureGrid(temperatureSource, rainfallDate, controller.signal)
      .then((grid) => setTemperatureGrid(grid))
      .catch((error: unknown) => {
        if (isAbortError(error)) return;
        console.error(`Could not load IMD ${temperatureSource} grid:`, error);
        setTemperatureError(
          error instanceof Error
            ? error.message
            : `IMD ${temperatureSource} grid is unavailable.`,
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setTemperatureLoading(false);
      });
    return () => controller.abort();
  }, [rainfallDate, temperatureSource, viewMode]);

  useEffect(() => {
    if (rainfallSource !== "imd" || !rainfallDate || viewMode !== "historical") {
      return;
    }

    const controller = new AbortController();
    void fetchImdRainfallGrid(rainfallDate, controller.signal)
      .then((grid) => setRainfallGrid(grid))
      .catch((error: unknown) => {
        if (isAbortError(error)) return;
        console.error("Could not load IMD rainfall grid:", error);
        setRainfallError(
          error instanceof Error ? error.message : "IMD rainfall grid is unavailable.",
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setRainfallLoading(false);
      });
    return () => controller.abort();
  }, [rainfallDate, rainfallSource, viewMode]);

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

    const addRainfallLayer = async () => {
      if (viewMode !== "historical") return;
      if (rainfallSource === "imd" && rainfallGrid) {
        const image = createRainfallRaster(rainfallGrid);
        const provider = await Cesium.SingleTileImageryProvider.fromUrl(image, {
          rectangle: Cesium.Rectangle.fromDegrees(66.375, 6.375, 100.125, 38.625),
        });
        if (cancelled) return;
        addedLayer = viewer.imageryLayers.addImageryProvider(provider);
        addedLayer.alpha = 0.84;
      } else if (
        (temperatureSource === "tmax" || temperatureSource === "tmin") &&
        temperatureGrid
      ) {
        const image = createTemperatureRaster(temperatureGrid);
        const provider = await Cesium.SingleTileImageryProvider.fromUrl(image, {
          rectangle: Cesium.Rectangle.fromDegrees(67, 7, 98, 38),
        });
        if (cancelled) return;
        addedLayer = viewer.imageryLayers.addImageryProvider(provider);
        addedLayer.alpha = 0.82;
      } else if (
        rainfallSource === "nasa" &&
        nasaAvailability &&
        rainfallDate >= nasaAvailability.observation_date_start &&
        rainfallDate <= nasaAvailability.observation_date_end
      ) {
        const provider = new Cesium.WebMapServiceImageryProvider({
          url: nasaAvailability.imagery_service,
          layers: nasaAvailability.layer,
          parameters: {
            format: "image/png",
            transparent: "true",
            time: rainfallDate,
          },
          tilingScheme: new Cesium.GeographicTilingScheme(),
          enablePickFeatures: false,
          credit: nasaAvailability.attribution,
        });
        addedLayer = viewer.imageryLayers.addImageryProvider(provider);
        addedLayer.alpha = 0.88;
      }
      if (addedLayer) {
        rainfallLayerRef.current = addedLayer;
        viewer.scene.requestRender();
      }
    };

    void addRainfallLayer().catch((error: unknown) => {
      if (cancelled) return;
      console.error("Could not create the Cesium rainfall layer:", error);
      setRainfallError(
        error instanceof Error ? error.message : "Could not display the rainfall layer.",
      );
    });
    return () => {
      cancelled = true;
      if (addedLayer) viewer.imageryLayers.remove(addedLayer, true);
      if (rainfallLayerRef.current === addedLayer) rainfallLayerRef.current = null;
    };
  }, [
    nasaAvailability,
    rainfallDate,
    rainfallGrid,
    rainfallSource,
    temperatureGrid,
    temperatureSource,
    viewerReady,
    viewMode,
  ]);

  useEffect(
    () => () => {
      searchControllerRef.current?.abort();
      rainfallCellControllerRef.current?.abort();
    },
    [],
  );

  return (
    <main className="earth-app">
      <ClimateGlobe onViewerReady={setViewerRef} />

      <header className="top-overlay">
        <div className="brand-lockup">
          <span className="brand-mark"><span /></span>
          <div>
            <p className="brand-name">INDIA CLIMATE TWIN</p>
            <p className="location-line">{selectedLocation.name}<span> / {selectedLocation.region}</span></p>
          </div>
        </div>

        <div className="top-right-controls">
          <div className="mode-switch" aria-label="Application mode">
            <button
              type="button"
              className={viewMode === "historical" ? "selected" : ""}
              onClick={() => {
                setRainfallGrid(null);
                setTemperatureGrid(null);
                setRainfallLoading(rainfallSource === "imd");
                setTemperatureLoading(temperatureSource !== "none");
                setViewMode("historical");
              }}
            >
              Observation
            </button>
            <span />
            <button
              type="button"
              className={viewMode === "forecast" ? "selected" : ""}
              onClick={() => {
                setRainfallGrid(null);
                setTemperatureGrid(null);
                setRainfallLoading(false);
                setTemperatureLoading(false);
                setViewMode("forecast");
              }}
            >
              Prediction
            </button>
          </div>

          <div className="top-status" title={apiStatusMessage}>
            <div className={`data-status api-status ${apiStatus}`}>
              <span className="status-dot" />
              <span>
                {apiStatus === "checking"
                  ? "Checking API"
                  : apiStatus === "healthy"
                    ? "API reachable"
                    : "API unavailable"}
              </span>
              <small>
                {imdDates.length > 0
                  ? `IMD rainfall · ${imdDates[0]}–${imdDates[imdDates.length - 1]}`
                  : "Checking rainfall sources"}
              </small>
            </div>
          </div>
        </div>
      </header>

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
        <button
          type="button"
          className={`toolbar-button${settingsOpen ? " active" : ""}`}
          aria-label="Map layers"
          aria-expanded={settingsOpen}
          onClick={() => setSettingsOpen((open) => !open)}
        >
          <Layers3 size={17} />
          <span>Layers</span>
        </button>
      </div>

      <nav className="navigation-rail" aria-label="Map navigation">
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

      {infoOpen && (
        <aside className="weather-panel" aria-label="Selected location information">
          <div className="panel-title-row">
            <div>
              <p className="eyebrow">Selected location</p>
              <h1>{selectedLocation.name}</h1>
            </div>
            <button type="button" className="icon-button panel-close" aria-label="Close location information" onClick={() => setInfoOpen(false)}>
              <X size={16} />
            </button>
          </div>
          <p className="location-caption">{selectedLocation.region} · {selectedLocation.code}</p>
          {viewMode === "forecast" ? (
            <div className="climate-unavailable">AI prediction outputs are not connected.</div>
          ) : temperatureSource !== "none" ? (
            <div className="climate-value-panel" aria-live="polite">
              {temperatureLoading && <span>Reading IMD temperature grid cell…</span>}
              {rainfallCellStatus === "error" && (
                <span className="rainfall-error">{rainfallCellError}</span>
              )}
              {selectedTemperatureCell && (
                <>
                  <strong>
                    {selectedTemperatureCell.temperature === null
                      ? "Missing observation"
                      : `${selectedTemperatureCell.temperature.toFixed(1)} °C`}
                  </strong>
                  <span>
                    {selectedTemperatureCell.observation_date} · IMD{" "}
                    {selectedTemperatureCell.variable} · native 1° grid
                  </span>
                  <small>
                    Grid cell {selectedTemperatureCell.latitude.toFixed(1)}°N,{" "}
                    {selectedTemperatureCell.longitude.toFixed(1)}°E
                  </small>
                </>
              )}
              {!selectedTemperatureCell && !temperatureLoading &&
                rainfallCellStatus !== "error" &&
                "Click the globe to inspect the nearest 1-degree temperature cell."}
            </div>
          ) : rainfallSource === "imd" ? (
            <div className="climate-value-panel" aria-live="polite">
              {rainfallCellStatus === "loading" && <span>Reading IMD grid cell…</span>}
              {rainfallCellStatus === "error" && (
                <span className="rainfall-error">{rainfallCellError}</span>
              )}
              {selectedRainfallCell && (
                <>
                  <strong>
                    {selectedRainfallCell.rainfall === null
                      ? "Missing observation"
                      : `${selectedRainfallCell.rainfall.toFixed(1)} mm/day`}
                  </strong>
                  <span>
                    {selectedRainfallCell.observation_date} · IMD 0.25° grid
                  </span>
                  <small>
                    Grid cell {selectedRainfallCell.latitude.toFixed(2)}°N,{" "}
                    {selectedRainfallCell.longitude.toFixed(2)}°E
                  </small>
                </>
              )}
              {!selectedRainfallCell &&
                rainfallCellStatus === "unavailable" &&
                "Choose the IMD rainfall layer and click a grid location to inspect a value."}
              {!selectedRainfallCell &&
                rainfallCellStatus === "idle" &&
                "Click the globe to inspect the nearest IMD grid cell."}
            </div>
          ) : rainfallSource === "nasa" ? (
            <div className="climate-value-panel">
              NASA GPM IMERG is shown as a rendered GIBS image only. Numeric rainfall values are
              not queryable from this layer.
            </div>
          ) : (
            <div className="climate-unavailable">
              Select a connected rainfall source to inspect this location.
            </div>
          )}
          <p className="source-note">
            {temperatureSource !== "none"
              ? `Source: India Meteorological Department. Historical daily ${temperatureSource === "tmax" ? "maximum" : "minimum"} temperature; native 1° grid; degrees Celsius. Missing cells are shown transparently.`
              : rainfallSource === "imd"
              ? "Source: India Meteorological Department. Historical daily observation; native 0.25° grid. Missing cells are shown transparently."
              : rainfallSource === "nasa"
                ? `Source: NASA GPM IMERG via GIBS · ${rainfallDate || "no date selected"} · visualization only; GES DISC grid access requires server-side Earthdata authentication.`
                : "No rainfall source is selected. Temperature and cyclone data are not connected."}
          </p>
        </aside>
      )}

      {settingsOpen && (
        <section className="settings-popover" aria-label="Map layers">
          <div className="popover-heading">
            <strong>Map layers</strong>
            <button type="button" className="icon-button" aria-label="Close layers" onClick={() => setSettingsOpen(false)}><X size={15} /></button>
          </div>
          <label className="layer-item">
            <input type="checkbox" checked={imageryVisible} onChange={(event) => setImageryVisible(event.target.checked)} />
            <span className="layer-swatch satellite-swatch" />
            <span>Satellite imagery</span>
          </label>
          <label className="layer-item">
            <input type="checkbox" checked={boundariesVisible} onChange={(event) => setBoundariesVisible(event.target.checked)} />
            <span className="layer-swatch boundary-swatch" />
            <span>India state boundaries</span>
          </label>
          <p className="layer-group-label">Climate fields</p>
          <div className="rainfall-layer-options">
            <button
              type="button"
              className={`rainfall-layer-option${temperatureSource === "tmax" ? " selected" : ""}`}
              aria-pressed={temperatureSource === "tmax"}
              disabled={temperatureDates.tmax.length === 0}
              onClick={() =>
                selectTemperatureSource(temperatureSource === "tmax" ? "none" : "tmax")
              }
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
              onClick={() =>
                selectTemperatureSource(temperatureSource === "tmin" ? "none" : "tmin")
              }
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
              title={
                nasaAvailabilityStatus === "unavailable"
                  ? "NASA GIBS availability could not be retrieved."
                  : "NASA GPM IMERG rendered imagery; grid values are not queryable."
              }
              onClick={() => selectRainfallSource(rainfallSource === "nasa" ? "none" : "nasa")}
            >
              <Cloud size={15} aria-hidden="true" />
              <span>NASA GPM IMERG</span>
              <small>
                {nasaAvailabilityStatus === "checking"
                  ? "Checking NASA GIBS"
                  : nasaAvailabilityStatus === "available" && nasaAvailability
                    ? `GIBS image · ${nasaAvailability.observation_date_end} · not queryable`
                    : "NASA GIBS unavailable"}
              </small>
            </button>
          </div>
          <div className="unavailable-climate-layers">
            {unavailableClimateLayers.map(({ label, source, icon: Icon }) => (
              <button
                type="button"
                className="unavailable-climate-layer"
                key={label}
                disabled
                title={`${label} unavailable: ${source}`}
              >
                <Icon size={15} aria-hidden="true" />
                <span>{label}</span>
                <small>{source}</small>
              </button>
            ))}
          </div>
          {rainfallLoading && <p className="source-note">Loading the selected IMD observation…</p>}
          {temperatureLoading && (
            <p className="source-note">Loading the selected IMD temperature observation…</p>
          )}
          {rainfallError && <p className="rainfall-error">{rainfallError}</p>}
          {temperatureError && <p className="rainfall-error">{temperatureError}</p>}
          {rainfallSource === "imd" && rainfallGrid && (
            <p className="source-note">
              {rainfallGrid.observation_date} · IMD · {rainfallGrid.units} · native{" "}
              {rainfallGrid.spatial_resolution_degrees}° · {rainfallGrid.valid_cell_count} valid
              grid cells
            </p>
          )}
          {rainfallSource === "nasa" && nasaAvailability && (
            <p className="source-note">
              {nasaAvailability.product} · {nasaAvailability.units} · {nasaAvailability.attribution}
            </p>
          )}
          {temperatureSource !== "none" && temperatureGrid && (
            <p className="source-note">
              {temperatureGrid.observation_date} · IMD {temperatureGrid.variable} · °C · native
              1° · {temperatureGrid.valid_cell_count} valid cells
            </p>
          )}
          <p className="source-note">
            Satellite basemap: Esri World Imagery. Provider acquisition timestamp is not supplied.
          </p>
        </section>
      )}

      <footer className="timeline-control" aria-label="Climate observation timeline">
        <div className="timeline-status">
          <CalendarDays size={15} aria-hidden="true" />
          <span>Historical observations</span>
          <small>
            {rainfallSource === "imd"
              ? `${imdDates.length} IMD dates`
              : temperatureSource !== "none"
                ? `${temperatureDates[temperatureSource].length} IMD dates`
              : rainfallSource === "nasa" && nasaAvailability
                ? "NASA GIBS imagery"
                : "Select a rainfall source"}
          </small>
        </div>
        <input
          aria-label={
            temperatureSource !== "none"
              ? "Select temperature observation date"
              : "Select rainfall observation date"
          }
          type="date"
          min={
            rainfallSource === "imd"
              ? imdDates[0]
              : temperatureSource !== "none"
                ? temperatureDates[temperatureSource][0]
              : rainfallSource === "nasa"
                ? nasaAvailability?.observation_date_start
                : undefined
          }
          max={
            rainfallSource === "imd"
              ? imdDates[imdDates.length - 1]
              : temperatureSource !== "none"
                ? temperatureDates[temperatureSource][
                    temperatureDates[temperatureSource].length - 1
                  ]
              : rainfallSource === "nasa"
                ? nasaAvailability?.observation_date_end
                : undefined
          }
          value={rainfallDate}
          disabled={
            (temperatureSource !== "none" &&
              temperatureDates[temperatureSource].length === 0) ||
            (rainfallSource === "none" && temperatureSource === "none") ||
            (rainfallSource === "imd" && imdDates.length === 0) ||
            (rainfallSource === "nasa" && !nasaAvailability)
          }
          onChange={(event) => {
            const nextDate = event.target.value;
            if (
              rainfallSource === "nasa" &&
              nasaAvailability &&
              !isNasaDateAvailable(nextDate, nasaAvailability.available_time_intervals)
            ) {
              setRainfallError(
                "NASA GIBS does not advertise an image for that date. Choose a date within an available interval.",
              );
              return;
            }
            setRainfallError("");
            rainfallCellControllerRef.current?.abort();
            setRainfallDate(nextDate);
            setRainfallGrid(null);
            setTemperatureGrid(null);
            setRainfallLoading(rainfallSource === "imd" && viewMode === "historical");
            setTemperatureLoading(
              temperatureSource !== "none" && viewMode === "historical",
            );
            setSelectedRainfallCell(null);
            setSelectedTemperatureCell(null);
            setRainfallCellError("");
            setRainfallCellStatus("idle");
          }}
        />
        <span className="timeline-source">
          {rainfallSource === "imd"
            ? "IMD daily observations"
            : temperatureSource !== "none"
              ? `IMD ${temperatureSource === "tmax" ? "Tmax" : "Tmin"} daily observations`
            : rainfallSource === "nasa"
              ? "NASA GIBS daily imagery"
              : "No source selected"}
        </span>
      </footer>

      {viewMode === "historical" &&
        (rainfallSource !== "none" || temperatureSource !== "none") && (
        <aside
          className="rainfall-legend"
          aria-label={
            temperatureSource !== "none"
              ? "Temperature layer legend"
              : "Rainfall layer details"
          }
        >
          <strong>
            {temperatureSource !== "none"
              ? `IMD ${temperatureSource === "tmax" ? "Tmax" : "Tmin"} · °C`
              : rainfallSource === "imd"
                ? "IMD rainfall · mm/day"
                : "NASA IMERG · rendered image"}
          </strong>
          {temperatureSource !== "none" ? (
            <>
              <div className="temperature-gradient" />
              <div className="rainfall-ticks">
                <span>-10</span><span>0</span><span>15</span><span>25</span><span>35</span><span>45+ °C</span>
              </div>
              <small>
                {rainfallDate} · native 1° · IMD {temperatureSource === "tmax" ? "Tmax" : "Tmin"}
              </small>
            </>
          ) : rainfallSource === "imd" ? (
            <>
              <div className="rainfall-gradient" />
              <div className="rainfall-ticks">
                <span>0</span><span>1</span><span>5</span><span>15</span><span>30</span>
                <span>60</span><span>120</span><span>250+</span>
              </div>
              <small>{rainfallDate} · native 0.25° · IMD</small>
            </>
          ) : (
            <>
              {nasaAvailability?.legend_url && (
                <img
                  className="nasa-rainfall-legend"
                  src={nasaAvailability.legend_url}
                  alt="NASA GPM IMERG precipitation rate legend in millimeters per hour"
                />
              )}
              <small>
                {rainfallDate} · mm/hr · NASA GIBS visualization only; no numeric cell values
              </small>
            </>
          )}
        </aside>
      )}

      <div className="map-hint">
        Scroll to zoom <span>·</span> Drag to rotate <span>·</span> Right-drag to tilt{" "}
        <span>·</span> Middle/Shift-drag to pan <span>·</span> Double-click to fly
      </div>
    </main>
  );
}
