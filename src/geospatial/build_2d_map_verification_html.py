"""
2D GIS Map State-Level Interaction Generator (Phase 6).

Generates an updated interactive HTML map artifact with dedicated state management:
  C:/Users/HP/.gemini/antigravity/brain/87b936bf-bd4d-4d1a-b532-aad8d9e1485d/india_2d_map_verification.html

Features:
- State Hover: Highlights polygon, displays state name & ISO geographic identifier
- State Selection: Click to select, stores selectedStateId in global application state (window.AppState)
- Global Event Dispatcher: Triggers 'stateSelected' CustomEvent on selection change
- Boundary Visibility: State & UT polygon borders remain sharply visible at all times
"""

import os
import json


def generate_map_html(
    geojson_path: str = "data/external/geospatial/india_states/india_states_processed.geojson",
    output_html_path: str = "C:/Users/HP/.gemini/antigravity/brain/87b936bf-bd4d-4d1a-b532-aad8d9e1485d/india_2d_map_verification.html"
):
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    geojson_str = json.dumps(geojson_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>India 2D GIS Map - State Interaction & Boundary Verification</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .state-path {{
      transition: fill 0.15s ease-in-out, fill-opacity 0.15s ease-in-out, stroke-width 0.15s ease-in-out;
      cursor: pointer;
    }}
    .state-path:hover {{
      fill: #2563eb !important;
      fill-opacity: 0.8 !important;
      stroke: #1d4ed8 !important;
      stroke-width: 2.5px !important;
    }}
    .state-path.selected {{
      fill: #dc2626 !important;
      fill-opacity: 0.85 !important;
      stroke: #7f1d1d !important;
      stroke-width: 3.5px !important;
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      transition: opacity 0.1s ease-in-out;
      z-index: 50;
    }}
    #map-svg {{
      user-select: none;
      touch-action: none;
    }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-4 min-h-screen">

  <div class="max-w-7xl mx-auto space-y-4">

    <!-- Header Panel & Application State Status -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-blue-500/10 text-blue-500 border border-blue-500/20">Phase 6 State Interaction</span>
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">GIS Boundaries Active</span>
        </div>
        <h1 class="text-2xl font-bold mt-1 text-[var(--foreground)]">India Map State-Level Interaction</h1>
        <p class="text-sm text-[var(--muted-foreground)] mt-0.5">Hover to highlight state and view geographic identifier. Click to store selection in application state.</p>
      </div>

      <!-- Live Selection Status Banner -->
      <div id="selection-banner" class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-3 text-xs flex items-center gap-3 w-full md:w-auto">
        <div class="w-3 h-3 rounded-full bg-slate-400" id="status-indicator"></div>
        <div>
          <div class="text-[var(--muted-foreground)] font-medium">Selected State ID:</div>
          <div class="font-bold text-sm text-[var(--foreground)] font-mono" id="selected-state-display">None (Click a state)</div>
        </div>
      </div>
    </div>

    <!-- Map & Inspector Layout Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-4">

      <!-- Map Display Container (Cols 1-3) -->
      <div class="lg:col-span-3 bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm relative overflow-hidden flex flex-col items-center justify-center min-h-[680px]">
        
        <!-- Controls Header Bar -->
        <div class="w-full flex items-center justify-between pb-3 mb-2 border-b border-[var(--border)] z-10">
          <div class="flex items-center gap-2">
            <input type="text" id="state-search" placeholder="Search state name or ISO code..." 
                   class="bg-[var(--background)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-xs text-[var(--foreground)] focus:outline-none focus:ring-2 focus:ring-blue-500 w-64">
            <select id="region-filter" class="bg-[var(--background)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-xs text-[var(--foreground)] focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option value="ALL">All Regions (36)</option>
              <option value="North">North</option>
              <option value="South">South</option>
              <option value="East">East</option>
              <option value="West">West</option>
              <option value="Central">Central</option>
              <option value="North-East">North-East</option>
            </select>
          </div>
          <button id="reset-selection-btn" class="bg-slate-700 hover:bg-slate-800 text-white font-medium text-xs px-3 py-1.5 rounded-lg transition-colors">
            Clear Selection
          </button>
        </div>

        <!-- Floating Zoom Buttons -->
        <div class="absolute top-16 right-4 flex flex-col gap-1.5 z-10 bg-[var(--card)]/90 backdrop-blur border border-[var(--border)] rounded-lg p-1 shadow-md">
          <button id="zoom-in" class="w-8 h-8 flex items-center justify-center rounded text-lg font-bold hover:bg-blue-500/10 text-[var(--foreground)]">+</button>
          <button id="zoom-out" class="w-8 h-8 flex items-center justify-center rounded text-lg font-bold hover:bg-blue-500/10 text-[var(--foreground)]">-</button>
        </div>

        <!-- SVG Map Container -->
        <div id="svg-wrapper" class="w-full h-full flex items-center justify-center">
          <svg id="map-svg" viewBox="0 0 900 950" class="w-full max-h-[640px] drop-shadow-md">
            <g id="map-group"></g>
            <g id="labels-group" pointer-events="none"></g>
          </svg>
        </div>

        <!-- Hover Tooltip -->
        <div id="tooltip" class="tooltip opacity-0 bg-slate-900/95 text-white text-xs rounded-lg p-3 shadow-xl backdrop-blur border border-slate-700 pointer-events-none"></div>

        <!-- Map Footer -->
        <div class="w-full flex items-center justify-between pt-3 border-t border-[var(--border)] text-xs text-[var(--muted-foreground)] mt-2">
          <span>Boundary Stroke: <strong>Crisp 1.5px GIS Geometry Borders</strong></span>
          <span>Click Action: <strong>Stores selectedStateId in AppState</strong></span>
        </div>
      </div>

      <!-- State Selection Inspector Panel (Col 4) -->
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm flex flex-col justify-between space-y-4">
        <div>
          <div class="border-b border-[var(--border)] pb-3 mb-4">
            <h2 class="text-lg font-bold text-[var(--foreground)]" id="inspector-title">State Application State</h2>
            <p class="text-xs text-[var(--muted-foreground)]" id="inspector-subtitle">Active state selection payload stored in window.AppState</p>
          </div>

          <div id="inspector-content" class="space-y-3">
            <div class="p-4 bg-[var(--background)] border border-[var(--border)] rounded-lg text-center text-xs text-[var(--muted-foreground)]">
              No state selected. Click any state or UT polygon to update application state.
            </div>
          </div>
        </div>

        <!-- App State JSON Debug Viewer -->
        <div class="border-t border-[var(--border)] pt-3 text-xs space-y-2">
          <div class="font-bold text-[var(--foreground)] flex items-center justify-between">
            <span>window.AppState Payload:</span>
            <span class="text-[10px] text-blue-500 font-mono">Live</span>
          </div>
          <pre id="app-state-json" class="p-2.5 rounded bg-slate-950 text-emerald-400 font-mono text-[10px] overflow-x-auto max-h-36 border border-slate-800"></pre>
        </div>
      </div>

    </div>
  </div>

  <script>
    const geojsonData = {geojson_str};

    // Global Application State Store
    window.AppState = {{
      selectedStateId: null,
      selectedStateName: null,
      selectedStateIsoCode: null,
      selectedStateCategory: null,
      selectedRegion: null,
      selectedCentroid: null,
      selectedBbox: null,
      lastUpdated: new Date().toISOString()
    }};

    const REGION_COLORS = {{
      'North': '#3b82f6',
      'South': '#10b981',
      'East': '#f59e0b',
      'West': '#ec4899',
      'Central': '#8b5cf6',
      'North-East': '#06b6d4'
    }};

    // SVG Projection Mapping (WGS84 EPSG:4326 Lat/Lon -> SVG coordinates)
    const MIN_LON = 66.0, MAX_LON = 100.5;
    const MIN_LAT = 6.0, MAX_LAT = 38.5;
    const SVG_WIDTH = 900, SVG_HEIGHT = 950;

    function projectLon(lon) {{
      return ((lon - MIN_LON) / (MAX_LON - MIN_LON)) * SVG_WIDTH;
    }}

    function projectLat(lat) {{
      return SVG_HEIGHT - (((lat - MIN_LAT) / (MAX_LAT - MIN_LAT)) * SVG_HEIGHT);
    }}

    function ringToPathD(ring) {{
      return ring.map((pt, i) => {{
        const x = projectLon(pt[0]).toFixed(2);
        const y = projectLat(pt[1]).toFixed(2);
        return (i === 0 ? 'M' : 'L') + x + ' ' + y;
      }}).join(' ') + ' Z';
    }}

    function geometryToPathD(geometry) {{
      if (geometry.type === 'Polygon') {{
        return geometry.coordinates.map(ringToPathD).join(' ');
      }} else if (geometry.type === 'MultiPolygon') {{
        return geometry.coordinates.map(poly => poly.map(ringToPathD).join(' ')).join(' ');
      }}
      return '';
    }}

    const mapGroup = document.getElementById('map-group');
    const labelsGroup = document.getElementById('labels-group');
    const tooltip = document.getElementById('tooltip');

    function updateAppStateDisplay() {{
      document.getElementById('app-state-json').textContent = JSON.stringify(window.AppState, null, 2);
      
      const displayEl = document.getElementById('selected-state-display');
      const indicatorEl = document.getElementById('status-indicator');

      if (window.AppState.selectedStateId) {{
        displayEl.textContent = `${{window.AppState.selectedStateName}} (${{window.AppState.selectedStateIsoCode}})`;
        displayEl.className = 'font-bold text-sm text-red-500 font-mono';
        indicatorEl.className = 'w-3 h-3 rounded-full bg-red-500 animate-pulse';
      }} else {{
        displayEl.textContent = 'None (Click a state)';
        displayEl.className = 'font-bold text-sm text-[var(--foreground)] font-mono';
        indicatorEl.className = 'w-3 h-3 rounded-full bg-slate-400';
      }}
    }}

    function renderMap(filterRegion = 'ALL', searchQuery = '') {{
      mapGroup.innerHTML = '';
      labelsGroup.innerHTML = '';

      geojson_data.features.forEach(feature => {{
        const props = feature.properties;
        const stateId = feature.id || props.iso_code || props.STATE_CODE;
        const region = props.REGION || 'Central';
        const name = props.st_nm || props.state_name || props.STATE_NAME;
        const isoCode = props.iso_code || stateId;

        if (filterRegion !== 'ALL' && region !== filterRegion) return;
        if (searchQuery && !name.toLowerCase().includes(searchQuery.toLowerCase()) && !isoCode.toLowerCase().includes(searchQuery.toLowerCase())) return;

        const pathD = geometryToPathD(feature.geometry);
        const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        
        const isSelected = (window.AppState.selectedStateId === stateId);
        const baseColor = REGION_COLORS[region] || '#64748b';

        pathEl.setAttribute('d', pathD);
        pathEl.setAttribute('class', 'state-path' + (isSelected ? ' selected' : ''));
        pathEl.setAttribute('fill', baseColor);
        pathEl.setAttribute('fill-opacity', isSelected ? '0.85' : '0.45');
        pathEl.setAttribute('stroke', isSelected ? '#7f1d1d' : '#1e293b');
        pathEl.setAttribute('stroke-width', isSelected ? '3.5' : '1.5');
        pathEl.setAttribute('data-id', stateId);

        // Hover Behavior: Highlight state + display Name and Geographic Identifier (ISO code)
        pathEl.addEventListener('mouseenter', (e) => {{
          tooltip.style.opacity = '1';
          tooltip.innerHTML = `
            <div class="font-bold text-sm text-blue-300">${{name}}</div>
            <div class="text-[11px] text-slate-300 mt-1 font-mono">Geographic ID (ISO): <strong class="text-emerald-400">${{isoCode}}</strong></div>
            <div class="text-[11px] text-slate-300">Category: <strong>${{props.TYPE || 'State'}}</strong> (${{region}})</div>
            <div class="text-[10px] text-slate-400 mt-1">Click to store selection in AppState</div>
          `;
        }});

        pathEl.addEventListener('mousemove', (e) => {{
          tooltip.style.left = (e.pageX + 15) + 'px';
          tooltip.style.top = (e.pageY - 15) + 'px';
        }});

        pathEl.addEventListener('mouseleave', () => {{
          tooltip.style.opacity = '0';
        }});

        // Click Behavior: Select state + Store state ID in AppState + Emit CustomEvent
        pathEl.addEventListener('click', () => {{
          selectState(feature);
        }});

        mapGroup.appendChild(pathEl);

        // State Labels at Centroids
        if (props.centroid) {{
          const cx = projectLon(props.centroid[1]);
          const cy = projectLat(props.centroid[0]);

          const textEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          textEl.setAttribute('x', cx);
          textEl.setAttribute('y', cy);
          textEl.setAttribute('fill', '#0f172a');
          textEl.setAttribute('font-size', '9.5');
          textEl.setAttribute('font-weight', 'bold');
          textEl.setAttribute('text-anchor', 'middle');
          textEl.setAttribute('dominant-baseline', 'central');
          textEl.textContent = name;

          labelsGroup.appendChild(textEl);
        }}
      }});
    }}

    function selectState(feature) {{
      const props = feature.properties;
      const stateId = feature.id || props.iso_code;

      // Store in Application State Object
      window.AppState.selectedStateId = stateId;
      window.AppState.selectedStateName = props.state_name;
      window.AppState.selectedStateIsoCode = props.iso_code;
      window.AppState.selectedStateCategory = props.TYPE;
      window.AppState.selectedRegion = props.REGION;
      window.AppState.selectedCentroid = props.centroid;
      window.AppState.selectedBbox = props.bbox;
      window.AppState.lastUpdated = new Date().toISOString();

      // Dispatch Custom Event for external listeners
      window.dispatchEvent(new CustomEvent('stateSelected', {{ detail: window.AppState }}));

      updateAppStateDisplay();

      // Re-render map path styling
      document.querySelectorAll('.state-path').forEach(el => {{
        const isThis = (el.getAttribute('data-id') === stateId);
        el.classList.toggle('selected', isThis);
        if (isThis) {{
          el.setAttribute('stroke', '#7f1d1d');
          el.setAttribute('stroke-width', '3.5');
          el.setAttribute('fill-opacity', '0.85');
        }} else {{
          el.setAttribute('stroke', '#1e293b');
          el.setAttribute('stroke-width', '1.5');
          el.setAttribute('fill-opacity', '0.45');
        }}
      }});

      // Populate Inspector Sidebar Panel
      const content = document.getElementById('inspector-content');
      document.getElementById('inspector-title').textContent = props.state_name;
      document.getElementById('inspector-subtitle').textContent = `Stored in window.AppState [${{props.iso_code}}]`;

      content.innerHTML = `
        <div class="space-y-2 text-xs">
          <div class="flex justify-between p-2 rounded bg-red-500/10 border border-red-500/20 text-red-500 font-bold">
            <span>Selected State ID:</span>
            <span class="font-mono">${{stateId}}</span>
          </div>
          <div class="flex justify-between p-2 rounded bg-[var(--background)] border border-[var(--border)]">
            <span class="text-[var(--muted-foreground)]">State / UT Name:</span>
            <span class="font-bold text-[var(--foreground)]">${{props.state_name}}</span>
          </div>
          <div class="flex justify-between p-2 rounded bg-[var(--background)] border border-[var(--border)]">
            <span class="text-[var(--muted-foreground)]">ISO Code:</span>
            <span class="font-mono font-bold text-blue-500">${{props.iso_code}}</span>
          </div>
          <div class="flex justify-between p-2 rounded bg-[var(--background)] border border-[var(--border)]">
            <span class="text-[var(--muted-foreground)]">Category:</span>
            <span class="font-bold text-[var(--foreground)]">${{props.TYPE}}</span>
          </div>
          <div class="flex justify-between p-2 rounded bg-[var(--background)] border border-[var(--border)]">
            <span class="text-[var(--muted-foreground)]">Region:</span>
            <span class="font-bold text-[var(--foreground)]">${{props.REGION}}</span>
          </div>
          <div class="flex justify-between p-2 rounded bg-[var(--background)] border border-[var(--border)]">
            <span class="text-[var(--muted-foreground)]">Centroid Coordinates:</span>
            <span class="font-mono text-emerald-500">${{props.centroid[0]}}°N, ${{props.centroid[1]}}°E</span>
          </div>
        </div>
      `;
    }}

    // Reset Selection
    document.getElementById('reset-selection-btn').addEventListener('click', () => {{
      window.AppState.selectedStateId = null;
      window.AppState.selectedStateName = null;
      window.AppState.selectedStateIsoCode = null;
      window.AppState.selectedStateCategory = null;
      window.AppState.selectedRegion = null;
      window.AppState.selectedCentroid = null;
      window.AppState.selectedBbox = null;
      window.AppState.lastUpdated = new Date().toISOString();

      updateAppStateDisplay();
      renderMap(document.getElementById('region-filter').value, document.getElementById('state-search').value);

      document.getElementById('inspector-title').textContent = 'State Application State';
      document.getElementById('inspector-subtitle').textContent = 'Active state selection payload stored in window.AppState';
      document.getElementById('inspector-content').innerHTML = `
        <div class="p-4 bg-[var(--background)] border border-[var(--border)] rounded-lg text-center text-xs text-[var(--muted-foreground)]">
          No state selected. Click any state or UT polygon to update application state.
        </div>
      `;
    }});

    // Zoom & Pan Controller
    let zoomLevel = 1;
    let panX = 0, panY = 0;
    const mapSvg = document.getElementById('map-svg');

    function updateTransform() {{
      mapSvg.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{zoomLevel}})`;
    }}

    document.getElementById('zoom-in').addEventListener('click', () => {{
      zoomLevel = Math.min(zoomLevel * 1.25, 5);
      updateTransform();
    }});

    document.getElementById('zoom-out').addEventListener('click', () => {{
      zoomLevel = Math.max(zoomLevel / 1.25, 0.8);
      updateTransform();
    }});

    document.getElementById('region-filter').addEventListener('change', (e) => {{
      renderMap(e.target.value, document.getElementById('state-search').value);
    }});

    document.getElementById('state-search').addEventListener('input', (e) => {{
      renderMap(document.getElementById('region-filter').value, e.target.value);
    }});

    // Initial load
    updateAppStateDisplay();
    renderMap();
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[OK] Generated State Interaction Map HTML artifact: '{output_html_path}'.")


if __name__ == "__main__":
    generate_map_html()
