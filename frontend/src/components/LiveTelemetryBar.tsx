import React from "react";
import type { ProviderStatus, StationWeatherRecord } from "../services/liveClimateApi";

type Props = {
  records: StationWeatherRecord[];
  loading: boolean;
  error: string;
  providers: ProviderStatus[];
  selectedStationId: string | null;
  onSelectStation: (station: StationWeatherRecord) => void;
};

export const LiveTelemetryBar = React.memo(function LiveTelemetryBar({
  records,
  loading,
  error,
  providers,
  selectedStationId,
  onSelectStation,
}: Props) {
  return (
    <aside className="live-telemetry-strip" aria-label="Live climate telemetry">
      <div className="live-strip-header">
        <div className="live-strip-title">
          <span className="live-pulse-dot" aria-hidden="true" />
          <span>Latest available weather data</span>
        </div>

        <div className="live-providers-badges">
          {providers.map((p) => {
            const isGated = p.data_status === "credentials_required";
            const isActive = p.available && (p.data_status === "fresh" || p.data_status === "cached");
            const status = isGated
              ? "credentials required"
              : p.available
                ? p.data_status
                : "unavailable";
            return (
              <span
                key={p.provider_id}
                className={`provider-badge ${isActive ? "active" : isGated ? "gated" : ""}`}
                title={p.error_message || p.description}
              >
                {p.provider_id === "open_meteo" ? "Open-Meteo NWP" : p.name.split("(")[0].trim()}
                {`: ${status}`}
              </span>
            );
          })}
        </div>
      </div>

      {error && <p className="live-data-error">Latest weather refresh failed: {error}</p>}
      <div className="live-station-scroll" role="list">
        {loading && records.length === 0 ? (
          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.6)", padding: "4px 8px" }}>
            Loading latest available weather data…
          </span>
        ) : records.length === 0 ? (
          <span className="live-empty-state">
            {error ? "No previously loaded records are available." : "No records are currently available from configured weather providers."}
          </span>
        ) : (
          records.map((st) => {
            const isSelected = st.location_id === selectedStationId;
            return (
              <button
                type="button"
                key={st.location_id}
                className={`live-station-pill ${isSelected ? "selected" : ""}`}
                onClick={() => onSelectStation(st)}
                title={`${st.location_name}, ${st.state} · ${st.is_observation ? "Station observation" : "Model-derived analysis"} · ${st.source_product} · ${st.data_status} · ${st.observation_time_utc}`}
              >
                <strong>{st.location_name}</strong>
                <span className="record-kind">{st.is_observation ? "OBS" : "NWP"}</span>
                {st.temperature_c !== null && (
                  <span className="temp">{st.temperature_c.toFixed(1)}°C</span>
                )}
                {st.rainfall_mm !== null && (
                  <span className="rain">{st.rainfall_mm.toFixed(1)} mm</span>
                )}
                {st.data_status !== "fresh" && (
                  <span className="record-status">{st.data_status}</span>
                )}
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
});
