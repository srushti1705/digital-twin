import React from "react";
import {
  AlertTriangle,
  Droplets,
  Play,
  RotateCcw,
  SlidersHorizontal,
  Thermometer,
} from "lucide-react";
import type { ScenarioResult, ScenarioScope } from "../services/scenarioApi";

type Props = {
  scopes: ScenarioScope[];
  selectedScope: string;
  onSelectScope: (scope: string) => void;
  baselineSource: "historical" | "prediction";
  onSelectBaselineSource: (src: "historical" | "prediction") => void;
  baselineDate: string;
  onSelectBaselineDate: (date: string) => void;
  variableSelection: "rainfall" | "temperature" | "both";
  onSelectVariableSelection: (v: "rainfall" | "temperature" | "both") => void;
  rainfallChangePct: number;
  onChangeRainfallChangePct: (val: number) => void;
  tmaxDelta: number;
  onChangeTmaxDelta: (val: number) => void;
  tminDelta: number;
  onChangeTminDelta: (val: number) => void;
  running: boolean;
  error: string;
  onRunScenario: () => void;
  onReset: () => void;
  result: ScenarioResult | null;
  viewLayer: "scenario" | "baseline" | "difference";
  onSelectViewLayer: (l: "scenario" | "baseline" | "difference") => void;
  activeVariable: "rainfall" | "tmax" | "tmin";
  onSelectActiveVariable: (v: "rainfall" | "tmax" | "tmin") => void;
};

export const WhatIfScenarioDock = React.memo(function WhatIfScenarioDock({
  scopes,
  selectedScope,
  onSelectScope,
  baselineSource,
  onSelectBaselineSource,
  baselineDate,
  onSelectBaselineDate,
  variableSelection,
  onSelectVariableSelection,
  rainfallChangePct,
  onChangeRainfallChangePct,
  tmaxDelta,
  onChangeTmaxDelta,
  tminDelta,
  onChangeTminDelta,
  running,
  error,
  onRunScenario,
  onReset,
  result,
  viewLayer,
  onSelectViewLayer,
  activeVariable,
  onSelectActiveVariable,
}: Props) {
  const hasUnappliedChanges = Boolean(
    result &&
      (
        result.parameters.geographic_scope !== selectedScope ||
        result.parameters.baseline_source !== baselineSource ||
        result.parameters.baseline_date !== baselineDate ||
        result.parameters.variable_selection !== variableSelection ||
        result.parameters.rainfall_change_percent !== rainfallChangePct ||
        result.parameters.tmax_delta !== tmaxDelta ||
        result.parameters.tmin_delta !== tminDelta
      ),
  );

  return (
    <section className="whatif-dock" aria-label="What-If climate scenario parameters and simulation">
      <div className="whatif-dock-header">
        <div className="whatif-dock-title">
          <SlidersHorizontal size={15} />
          <span>Scenario inputs</span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button type="button" className="whatif-reset-btn" onClick={onReset} title="Reset parameters to zero">
            <RotateCcw size={12} />
            <span>Reset</span>
          </button>
          <button
            type="button"
            className="whatif-run-btn"
            onClick={onRunScenario}
            disabled={running}
            aria-busy={running}
          >
            <Play size={13} fill="currentColor" />
            <span>{running ? "Running simulation…" : "Run simulation"}</span>
          </button>
        </div>
      </div>

      {error && <div className="whatif-warning-box">{error}</div>}
      <div className="scenario-workflow" aria-label="Scenario workflow">
        <strong>INPUT</strong><span>→</span><strong>RUN</strong><span>→</span>
        <strong>RESULT</strong><span>→</span><strong>MAP</strong>
      </div>
      <p className="scenario-guidance">
        Rainfall = baseline × (1 + change % / 100). Temperature = baseline + change in °C.
        Rainfall is bounded at 0 mm/day. This is a simplified perturbation, not a physical climate model or forecast.
      </p>

      <div className="whatif-dock-grid">
        {/* 1. Geographic Scope */}
        <div className="whatif-field-group">
          <label className="whatif-label" htmlFor="whatif-scope">Geographic Scope</label>
          <select
            id="whatif-scope"
            className="whatif-select"
            value={selectedScope}
            onChange={(e) => onSelectScope(e.target.value)}
          >
            {scopes.map((s) => (
              <option key={s.code} value={s.code}>
                {s.name}
              </option>
            ))}
          </select>
        </div>

        {/* 2. Baseline Source & Date */}
        <div className="whatif-field-group">
          <label className="whatif-label" htmlFor="whatif-baseline-date">Baseline & Date</label>
          <div style={{ display: "flex", gap: 6 }}>
            <select
              className="whatif-select"
              style={{ flex: 1 }}
              value={baselineSource}
              onChange={(e) => onSelectBaselineSource(e.target.value as "historical" | "prediction")}
            >
              <option value="historical">Historical Observation</option>
              <option value="prediction">AI Model Prediction</option>
            </select>
            <input
              id="whatif-baseline-date"
              type="date"
              className="whatif-input"
              style={{ width: 130 }}
              value={baselineDate}
              onChange={(e) => onSelectBaselineDate(e.target.value)}
            />
          </div>
        </div>

        {/* 3. Variables Selection */}
        <div className="whatif-field-group">
          <label className="whatif-label">Target Variables</label>
          <div className="whatif-layer-toggle">
            <button
              type="button"
              className={variableSelection === "rainfall" ? "active" : ""}
              onClick={() => onSelectVariableSelection("rainfall")}
            >
              Rainfall
            </button>
            <button
              type="button"
              className={variableSelection === "temperature" ? "active" : ""}
              onClick={() => onSelectVariableSelection("temperature")}
            >
              Temperature
            </button>
            <button
              type="button"
              className={variableSelection === "both" ? "active" : ""}
              onClick={() => onSelectVariableSelection("both")}
            >
              Both
            </button>
          </div>
        </div>
      </div>

      {/* Sliders for perturbations */}
      <div className="whatif-dock-grid">
        {variableSelection !== "temperature" && (
          <div className="whatif-slider-container">
            <div className="whatif-slider-row">
              <span className="whatif-label" style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <Droplets size={12} color="#38bdf8" /> Rainfall Shift
              </span>
              <span className="whatif-slider-value">
                {rainfallChangePct > 0 ? `+${rainfallChangePct}%` : `${rainfallChangePct}%`}
              </span>
            </div>
            <input
              type="range"
              className="whatif-range"
              min={-100}
              max={100}
              step={5}
              value={rainfallChangePct}
              onChange={(e) => onChangeRainfallChangePct(Number(e.target.value))}
            />
            <div className="whatif-presets">
              {[-50, -20, 0, 20, 50].map((val) => (
                <button
                  type="button"
                  key={val}
                  className="whatif-preset-btn"
                  onClick={() => onChangeRainfallChangePct(val)}
                >
                  {val > 0 ? `+${val}%` : `${val}%`}
                </button>
              ))}
            </div>
          </div>
        )}

        {variableSelection !== "rainfall" && (
          <>
            <div className="whatif-slider-container">
              <div className="whatif-slider-row">
                <span className="whatif-label" style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <Thermometer size={12} color="#f59e0b" /> Max Temperature (Tmax)
                </span>
                <span className="whatif-slider-value">
                  {tmaxDelta > 0 ? `+${tmaxDelta.toFixed(1)}°C` : `${tmaxDelta.toFixed(1)}°C`}
                </span>
              </div>
              <input
                type="range"
                className="whatif-range"
                min={-5}
                max={5}
                step={0.5}
                value={tmaxDelta}
                onChange={(e) => onChangeTmaxDelta(Number(e.target.value))}
              />
              <div className="whatif-presets">
                {[-2, 0, 1.5, 3, 4.5].map((val) => (
                  <button
                    type="button"
                    key={val}
                    className="whatif-preset-btn"
                    onClick={() => onChangeTmaxDelta(val)}
                  >
                    {val > 0 ? `+${val}°C` : `${val}°C`}
                  </button>
                ))}
              </div>
            </div>

            <div className="whatif-slider-container">
              <div className="whatif-slider-row">
                <span className="whatif-label" style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <Thermometer size={12} color="#38bdf8" /> Min Temperature (Tmin)
                </span>
                <span className="whatif-slider-value">
                  {tminDelta > 0 ? `+${tminDelta.toFixed(1)}°C` : `${tminDelta.toFixed(1)}°C`}
                </span>
              </div>
              <input
                type="range"
                className="whatif-range"
                min={-5}
                max={5}
                step={0.5}
                value={tminDelta}
                onChange={(e) => onChangeTminDelta(Number(e.target.value))}
              />
              <div className="whatif-presets">
                {[-2, 0, 1.5, 3, 4.5].map((val) => (
                  <button
                    type="button"
                    key={val}
                    className="whatif-preset-btn"
                    onClick={() => onChangeTminDelta(val)}
                  >
                    {val > 0 ? `+${val}°C` : `${val}°C`}
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Scenario Output Card */}
      {result && (
        <div className="whatif-results-card">
          {hasUnappliedChanges && (
            <p className="scenario-stale-note">
              Inputs changed since this run. Run the simulation again to update the results and map.
            </p>
          )}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <span className="whatif-label">Globe Raster:</span>
              <div className="whatif-layer-toggle" style={{ margin: 0 }}>
                <button
                  type="button"
                  className={viewLayer === "scenario" ? "active" : ""}
                  onClick={() => onSelectViewLayer("scenario")}
                >
                  Scenario Output
                </button>
                <button
                  type="button"
                  className={viewLayer === "baseline" ? "active" : ""}
                  onClick={() => onSelectViewLayer("baseline")}
                >
                  Baseline
                </button>
                <button
                  type="button"
                  className={viewLayer === "difference" ? "active" : ""}
                  onClick={() => onSelectViewLayer("difference")}
                >
                  Difference (Delta)
                </button>
              </div>
            </div>

            {/* Variable toggle if both available */}
            {(result.rainfall_summary || result.tmax_summary || result.tmin_summary) && (
              <div style={{ display: "flex", gap: 4 }}>
                {result.rainfall_summary && (
                  <button
                    type="button"
                    className={`whatif-preset-btn ${activeVariable === "rainfall" ? "active" : ""}`}
                    onClick={() => onSelectActiveVariable("rainfall")}
                  >
                    Rainfall
                  </button>
                )}
                {result.tmax_summary && (
                  <button
                    type="button"
                    className={`whatif-preset-btn ${activeVariable === "tmax" ? "active" : ""}`}
                    onClick={() => onSelectActiveVariable("tmax")}
                  >
                    Tmax
                  </button>
                )}
                {result.tmin_summary && (
                  <button
                    type="button"
                    className={`whatif-preset-btn ${activeVariable === "tmin" ? "active" : ""}`}
                    onClick={() => onSelectActiveVariable("tmin")}
                  >
                    Tmin
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Metrics summary */}
          <div className="whatif-metrics-row">
            {result.rainfall_summary && (
              <>
                <div className="whatif-metric-box">
                  <div className="label">Baseline Rainfall Mean</div>
                  <div className="val">{result.rainfall_summary.baseline.mean?.toFixed(1) ?? "—"} mm/day</div>
                </div>
                <div className="whatif-metric-box">
                  <div className="label">Scenario Rainfall Mean</div>
                  <div className="val">{result.rainfall_summary.scenario.mean?.toFixed(1) ?? "—"} mm/day</div>
                </div>
                <div className="whatif-metric-box">
                  <div className="label">Net Rainfall Delta</div>
                  <div className={`val ${(result.rainfall_summary.delta_mean ?? 0) >= 0 ? "positive" : "negative"}`}>
                    {result.rainfall_summary.delta_mean !== null
                      ? `${result.rainfall_summary.delta_mean > 0 ? "+" : ""}${result.rainfall_summary.delta_mean.toFixed(1)} mm/day${result.rainfall_summary.delta_percent !== null ? ` (${result.rainfall_summary.delta_percent}%)` : ""}`
                      : "—"}
                  </div>
                </div>
              </>
            )}

            {result.tmax_summary && (
              <>
                <div className="whatif-metric-box">
                  <div className="label">Baseline Tmax Mean</div>
                  <div className="val">{result.tmax_summary.baseline.mean?.toFixed(1) ?? "—"} °C</div>
                </div>
                <div className="whatif-metric-box">
                  <div className="label">Scenario Tmax Mean</div>
                  <div className="val">{result.tmax_summary.scenario.mean?.toFixed(1) ?? "—"} °C</div>
                </div>
                <div className="whatif-metric-box">
                  <div className="label">Net Tmax Delta</div>
                  <div className={`val ${result.tmax_summary.delta_mean === null ? "" : result.tmax_summary.delta_mean >= 0 ? "positive" : "negative"}`}>
                    {result.tmax_summary.delta_mean !== null
                      ? `${result.tmax_summary.delta_mean > 0 ? "+" : ""}${result.tmax_summary.delta_mean.toFixed(2)} °C`
                      : "—"}
                  </div>
                </div>
              </>
            )}

            {result.tmin_summary && (
              <>
                <div className="whatif-metric-box">
                  <div className="label">Baseline Tmin Mean</div>
                  <div className="val">{result.tmin_summary.baseline.mean?.toFixed(1) ?? "—"} °C</div>
                </div>
                <div className="whatif-metric-box">
                  <div className="label">Scenario Tmin Mean</div>
                  <div className="val">{result.tmin_summary.scenario.mean?.toFixed(1) ?? "—"} °C</div>
                </div>
                <div className="whatif-metric-box">
                  <div className="label">Tmin Difference</div>
                  <div className={`val ${result.tmin_summary.delta_mean === null ? "" : result.tmin_summary.delta_mean >= 0 ? "positive" : "negative"}`}>
                    {result.tmin_summary.delta_mean !== null
                      ? `${result.tmin_summary.delta_mean > 0 ? "+" : ""}${result.tmin_summary.delta_mean.toFixed(2)} °C`
                      : "—"}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Warnings */}
          {result.warnings.length > 0 && (
            <div className="whatif-warning-box">
              <AlertTriangle size={15} style={{ flexShrink: 0 }} />
              <div>
                {result.warnings.map((w, idx) => (
                  <p key={idx}>{w}</p>
                ))}
              </div>
            </div>
          )}

          {/* Scientific Disclaimer */}
          <div className="whatif-disclaimer-box">
            <strong>Scientific note:</strong> {result.scientific_disclaimer}
            <br />
            Baseline: {result.parameters.baseline_source} · {result.parameters.baseline_date}
            {" · "}Simulation time: {new Date(result.created_at_utc).toLocaleString()}
          </div>
        </div>
      )}
      {!result && !running && (
        <p className="scenario-empty-state">
          Set the inputs above and run a simulation. The result will appear here and as a separate raster on the globe.
        </p>
      )}
    </section>
  );
});
