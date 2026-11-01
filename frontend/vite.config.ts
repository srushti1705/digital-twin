
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

type CesiumPluginFactory = () => import("vite").PluginOption;

function isCesiumPluginFactory(value: unknown): value is CesiumPluginFactory {
  return typeof value === "function";
}

export default defineConfig(async () => {
  const cesiumModule: unknown = await import("vite-plugin-cesium");
  const candidate =
    typeof cesiumModule === "object" && cesiumModule !== null && "default" in cesiumModule
      ? cesiumModule.default
      : cesiumModule;

  if (!isCesiumPluginFactory(candidate)) {
    throw new TypeError("The installed Cesium Vite plugin does not export a plugin factory.");
  }

  return { plugins: [react(), candidate()] };
});
