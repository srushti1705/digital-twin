"""
Dynamic 3D State Label Generator & WebGL Digital Twin Map (Phase 6).

Generates an updated Three.js WebGL 3D Digital Twin artifact:
  C:/Users/HP/.gemini/antigravity/brain/87b936bf-bd4d-4d1a-b532-aad8d9e1485d/india_3d_digital_twin_verification.html

Label Features:
- 100% Dynamic Centroid Calculation: Calculates area-weighted center (Cx, Cy) directly from GIS Polygon/MultiPolygon rings without hardcoding coordinates.
- Distance-Aware Zoom Scaling: Fades/scales labels dynamically based on camera distance (prevents overcrowding when zoomed out).
- Billboarding: Labels remain facing camera with crisp dark backdrop pills.
- Independent Boundary Mesh: 3D boundary lines remain 100% decoupled from label projection.
"""

import os
import json


def generate_3d_map_html(
    geojson_path: str = "data/external/geospatial/india_states/india_states_processed.geojson",
    output_html_path: str = "C:/Users/HP/.gemini/antigravity/brain/87b936bf-bd4d-4d1a-b532-aad8d9e1485d/india_3d_digital_twin_verification.html"
):
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    geojson_str = json.dumps(geojson_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>India 3D GIS Digital Twin - Dynamic State Labels</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <style>
    body {{
      margin: 0;
      overflow: hidden;
      background-color: #070a12;
      color: #f8fafc;
      font-family: system-ui, -apple-system, sans-serif;
    }}
    #webgl-canvas {{
      width: 100vw;
      height: 100vh;
      display: block;
    }}
    .ui-panel {{
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.12);
    }}
    .tooltip-3d {{
      position: absolute;
      pointer-events: none;
      transition: opacity 0.12s ease-in-out;
      z-index: 50;
    }}
    .state-label-3d {{
      position: absolute;
      pointer-events: none;
      transform: translate(-50%, -50%);
      font-size: 10px;
      font-weight: 700;
      color: #f8fafc;
      background: rgba(15, 23, 42, 0.82);
      padding: 2.5px 7px;
      border-radius: 5px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      white-space: nowrap;
      user-select: none;
      transition: opacity 0.2s ease-in-out, transform 0.1s ease-out;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }}
    .state-label-3d.ut-label {{
      background: rgba(30, 41, 59, 0.85);
      border-color: rgba(56, 189, 248, 0.3);
      font-size: 9px;
    }}
  </style>
</head>
<body>

  <canvas id="webgl-canvas"></canvas>

  <!-- HTML Layer for Floating 3D State Labels -->
  <div id="labels-container" class="absolute inset-0 pointer-events-none overflow-hidden"></div>

  <!-- Header Info Panel -->
  <div class="absolute top-4 left-4 z-20 ui-panel rounded-xl p-5 shadow-2xl max-w-md">
    <div class="flex items-center gap-2">
      <span class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30">Phase 6 Dynamic GIS Labels</span>
      <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-400/30">Area-Weighted Centroids</span>
    </div>
    <h1 class="text-xl font-bold mt-1 text-white">3D India Interactive State Labels</h1>
    <p class="text-xs text-slate-300 mt-1">
      Labels calculated dynamically from GIS polygon geometry ring areas. Labels auto-scale with zoom and remain visible during camera rotation.
    </p>

    <div id="selection-card" class="mt-4 pt-3 border-t border-slate-700/60 text-xs space-y-1.5">
      <div class="text-slate-400 flex items-center justify-between">
        <span>Selected Region:</span>
        <span id="selected-state-id" class="font-mono text-emerald-400 font-bold">None</span>
      </div>
      <div id="selected-state-name" class="text-sm font-bold text-white">Click any 3D state to highlight & focus camera</div>
      <div id="selected-state-meta" class="text-[11px] text-slate-300"></div>
    </div>
  </div>

  <!-- Top-Right Controls Bar -->
  <div class="absolute top-4 right-4 z-20 flex flex-col gap-2">
    <button id="reset-cam-btn" class="ui-panel hover:bg-blue-600/30 text-white text-xs font-semibold px-4 py-2.5 rounded-xl shadow-lg transition-all border border-blue-500/30 flex items-center gap-2">
      Reset Camera (National View)
    </button>
    <button id="toggle-labels-btn" class="ui-panel hover:bg-slate-700/40 text-slate-200 text-xs font-medium px-4 py-2 rounded-xl transition-all border border-slate-700">
      Toggle Labels: Visible
    </button>
  </div>

  <!-- Hover Tooltip -->
  <div id="tooltip" class="tooltip-3d opacity-0 bg-slate-900/95 text-white text-xs rounded-lg p-3 shadow-2xl backdrop-blur border border-slate-700 pointer-events-none"></div>

  <script>
    const geojsonData = {geojson_str};

    const REGION_COLORS = {{
      'North': 0x2563eb,
      'South': 0x059669,
      'East': 0xd97706,
      'West': 0xdb2777,
      'Central': 0x7c3aed,
      'North-East': 0x0891b2
    }};

    // 1. Dynamic Polygon Centroid Calculation (Area-Weighted Center of Mass)
    function calculateRingAreaAndCentroid(ring) {{
      let area = 0;
      let cx = 0;
      let cy = 0;
      const n = ring.length;

      for (let i = 0; i < n - 1; i++) {{
        const x0 = ring[i][0];
        const y0 = ring[i][1];
        const x1 = ring[i+1][0];
        const y1 = ring[i+1][1];

        const cross = (x0 * y1 - x1 * y0);
        area += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
      }}

      area = area / 2.0;
      if (Math.abs(area) < 1e-9) {{
        // Fallback to bounding box center if area is near zero
        const lons = ring.map(p => p[0]);
        const lats = ring.map(p => p[1]);
        return {{
          area: 0.001,
          centroid: [(Math.min(...lats) + Math.max(...lats)) / 2.0, (Math.min(...lons) + Math.max(...lons)) / 2.0]
        }};
      }}

      cx = cx / (6.0 * area);
      cy = cy / (6.0 * area);

      return {{
        area: Math.abs(area),
        centroid: [cy, cx] // [latitude, longitude]
      }};
    }}

    function calculateGeometryCentroid(geometry) {{
      if (geometry.type === 'Polygon') {{
        return calculateRingAreaAndCentroid(geometry.coordinates[0]).centroid;
      }} else if (geometry.type === 'MultiPolygon') {{
        let maxArea = -1;
        let bestCentroid = [0, 0];

        geometry.coordinates.forEach(polyCoords => {{
          const res = calculateRingAreaAndCentroid(polyCoords[0]);
          if (res.area > maxArea) {{
            maxArea = res.area;
            bestCentroid = res.centroid;
          }}
        }});
        return bestCentroid;
      }}
      return [0, 0];
    }}

    // 2. Scene, Camera, Renderer Setup
    const canvas = document.getElementById('webgl-canvas');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070a12);
    scene.fog = new THREE.FogExp2(0x070a12, 0.007);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    const DEFAULT_CAM_POS = new THREE.Vector3(0, -22, 26);
    const DEFAULT_CAM_TARGET = new THREE.Vector3(0, 0, 0);

    camera.position.copy(DEFAULT_CAM_POS);

    const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true, alpha: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2.05;
    controls.minDistance = 6;
    controls.maxDistance = 85;
    controls.target.copy(DEFAULT_CAM_TARGET);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.75);
    dirLight1.position.set(20, 30, 40);
    dirLight1.castShadow = true;
    dirLight1.shadow.mapSize.width = 2048;
    dirLight1.shadow.mapSize.height = 2048;
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.4);
    dirLight2.position.set(-25, -25, 25);
    scene.add(dirLight2);

    const CENTER_LON = 82.5;
    const CENTER_LAT = 22.5;
    const SCALE_FACTOR = 0.8;

    function projectLon(lon) {{ return (lon - CENTER_LON) * SCALE_FACTOR; }}
    function projectLat(lat) {{ return (lat - CENTER_LAT) * SCALE_FACTOR; }}

    const stateGroup = new THREE.Group();
    scene.add(stateGroup);

    const stateMeshes = [];
    const labelElements = [];

    // 3. Build 3D Extruded State Meshes & Calculate Dynamic Centroids
    geojsonData.features.forEach(feature => {{
      const props = feature.properties;
      const stateId = feature.id || props.iso_code;
      const region = props.REGION || 'Central';
      const colorHex = REGION_COLORS[region] || 0x64748b;
      const geometryType = feature.geometry.type;
      const coordinates = feature.geometry.coordinates;

      // Dynamically calculate centroid from polygon geometry
      const computedCentroid = calculateGeometryCentroid(feature.geometry);

      const shapes = [];

      function processPolygonCoordinates(polyCoords) {{
        const outerRing = polyCoords[0];
        const shape = new THREE.Shape();

        outerRing.forEach((pt, i) => {{
          const x = projectLon(pt[0]);
          const y = projectLat(pt[1]);
          if (i === 0) shape.moveTo(x, y);
          else shape.lineTo(x, y);
        }});

        for (let h = 1; h < polyCoords.length; h++) {{
          const holeRing = polyCoords[h];
          const holePath = new THREE.Path();
          holeRing.forEach((pt, i) => {{
            const x = projectLon(pt[0]);
            const y = projectLat(pt[1]);
            if (i === 0) holePath.moveTo(x, y);
            else holePath.lineTo(x, y);
          }});
          shape.holes.push(holePath);
        }}
        shapes.push(shape);
      }}

      if (geometryType === 'Polygon') processPolygonCoordinates(coordinates);
      else if (geometryType === 'MultiPolygon') coordinates.forEach(processPolygonCoordinates);

      const extrudeSettings = {{
        depth: 0.6,
        bevelEnabled: true,
        bevelSegments: 2,
        steps: 1,
        bevelSize: 0.03,
        bevelThickness: 0.03
      }};

      const extrudeGeometry = new THREE.ExtrudeGeometry(shapes, extrudeSettings);

      const material = new THREE.MeshStandardMaterial({{
        color: colorHex,
        roughness: 0.4,
        metalness: 0.1,
        flatShading: false
      }});

      const mesh = new THREE.Mesh(extrudeGeometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = {{
        id: stateId,
        name: props.state_name || props.st_nm,
        st_nm: props.st_nm || props.state_name,
        iso_code: props.iso_code,
        type: props.TYPE,
        region: region,
        centroid: computedCentroid,  // Derived dynamically from geometry
        is_ut: props.is_union_territory || props.TYPE === 'Union Territory',
        baseColor: colorHex
      }};

      stateGroup.add(mesh);
      stateMeshes.push(mesh);

      // Unlit Boundary Line Segments
      const edgesGeom = new THREE.EdgesGeometry(extrudeGeometry, 18);
      const lineMat = new THREE.LineBasicMaterial({{
        color: 0x94a3b8,
        linewidth: 1.5,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1
      }});

      const lineSegments = new THREE.LineSegments(edgesGeom, lineMat);
      lineSegments.position.z = 0.61;
      lineSegments.renderOrder = 10;
      mesh.add(lineSegments);

      // Create Interactive HTML 3D Label
      const cx = projectLon(computedCentroid[1]);
      const cy = projectLat(computedCentroid[0]);

      const labelDiv = document.createElement('div');
      labelDiv.className = 'state-label-3d' + (mesh.userData.is_ut ? ' ut-label' : '');
      labelDiv.textContent = mesh.userData.st_nm;
      document.getElementById('labels-container').appendChild(labelDiv);

      labelElements.push({{
        element: labelDiv,
        pos3D: new THREE.Vector3(cx, cy, 0.8),
        isUT: mesh.userData.is_ut,
        stateId: stateId
      }});
    }});

    // 4. Raycasting & Selection Mechanics
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoveredMesh = null;
    let selectedMesh = null;
    const tooltip = document.getElementById('tooltip');

    function onPointerMove(event) {{
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(stateMeshes);

      if (intersects.length > 0) {{
        const hitMesh = intersects[0].object;

        if (hoveredMesh !== hitMesh) {{
          if (hoveredMesh && hoveredMesh !== selectedMesh) {{
            hoveredMesh.material.color.setHex(hoveredMesh.userData.baseColor);
            hoveredMesh.position.z = 0;
          }}
          hoveredMesh = hitMesh;

          if (hoveredMesh !== selectedMesh) {{
            hoveredMesh.material.color.setHex(0x3b82f6);
            hoveredMesh.position.z = 0.15;
          }}
        }}

        const data = hitMesh.userData;
        tooltip.style.opacity = '1';
        tooltip.style.left = (event.clientX + 15) + 'px';
        tooltip.style.top = (event.clientY - 15) + 'px';
        tooltip.innerHTML = `
          <div class="font-bold text-sm text-blue-300">${{data.name}}</div>
          <div class="text-[11px] text-slate-300 mt-1">ISO Code: <strong class="text-emerald-400">${{data.iso_code}}</strong></div>
          <div class="text-[11px] text-slate-300">Computed Centroid: <strong>${{data.centroid[0].toFixed(2)}}°N, ${{data.centroid[1].toFixed(2)}}°E</strong></div>
        `;
        canvas.style.cursor = 'pointer';
      }} else {{
        if (hoveredMesh && hoveredMesh !== selectedMesh) {{
          hoveredMesh.material.color.setHex(hoveredMesh.userData.baseColor);
          hoveredMesh.position.z = 0;
        }}
        hoveredMesh = null;
        tooltip.style.opacity = '0';
        canvas.style.cursor = 'default';
      }}
    }}

    function onPointerClick(event) {{
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(stateMeshes);

      if (intersects.length > 0) {{
        const hitMesh = intersects[0].object;
        selectStateMesh(hitMesh);
      }}
    }}

    function selectStateMesh(mesh) {{
      if (selectedMesh) {{
        selectedMesh.material.color.setHex(selectedMesh.userData.baseColor);
        selectedMesh.position.z = 0;
      }}

      selectedMesh = mesh;
      selectedMesh.material.color.setHex(0xef4444);
      selectedMesh.position.z = 0.35;

      const data = mesh.userData;
      document.getElementById('selected-state-id').textContent = data.iso_code;
      document.getElementById('selected-state-name').textContent = data.name;
      document.getElementById('selected-state-meta').innerHTML = `
        <div>Type: <strong>${{data.type}}</strong> | Region: <strong>${{data.region}}</strong></div>
        <div>Geometry Centroid: <strong>${{data.centroid[0].toFixed(2)}}°N, ${{data.centroid[1].toFixed(2)}}°E</strong></div>
      `;

      const targetX = projectLon(data.centroid[1]);
      const targetY = projectLat(data.centroid[0]);
      animateCameraTarget(targetX, targetY, 0.5);
    }}

    document.getElementById('reset-cam-btn').addEventListener('click', () => {{
      if (selectedMesh) {{
        selectedMesh.material.color.setHex(selectedMesh.userData.baseColor);
        selectedMesh.position.z = 0;
        selectedMesh = null;
      }}
      document.getElementById('selected-state-id').textContent = 'None';
      document.getElementById('selected-state-name').textContent = 'Click any 3D state to highlight & focus camera';
      document.getElementById('selected-state-meta').textContent = '';

      animateCameraTarget(0, 0, 0, DEFAULT_CAM_POS);
    }});

    let labelsVisible = true;
    document.getElementById('toggle-labels-btn').addEventListener('click', (e) => {{
      labelsVisible = !labelsVisible;
      document.getElementById('labels-container').style.display = labelsVisible ? 'block' : 'none';
      e.target.textContent = `Toggle Labels: ${{labelsVisible ? 'Visible' : 'Hidden'}}`;
    }});

    function animateCameraTarget(tx, ty, tz, camPos = null) {{
      const startTarget = controls.target.clone();
      const endTarget = new THREE.Vector3(tx, ty, tz);
      const startPos = camera.position.clone();
      const endPos = camPos || new THREE.Vector3(tx, ty - 16, 20);

      let progress = 0;
      function step() {{
        progress += 0.05;
        if (progress <= 1) {{
          controls.target.lerpVectors(startTarget, endTarget, progress);
          camera.position.lerpVectors(startPos, endPos, progress);
          controls.update();
          requestAnimationFrame(step);
        }}
      }}
      step();
    }}

    // 5. Distance-Aware Zoom Label Scaling & Occlusion Manager
    const tempV = new THREE.Vector3();
    function updateHTMLStateLabels() {{
      if (!labelsVisible) return;

      const camDistance = camera.position.distanceTo(controls.target);

      labelElements.forEach(item => {{
        tempV.copy(item.pos3D);
        tempV.project(camera);

        const x = (tempV.x * 0.5 + 0.5) * window.innerWidth;
        const y = (-(tempV.y * 0.5) + 0.5) * window.innerHeight;

        // Hide if behind camera or outside viewport
        if (tempV.z > 1.0 || x < 0 || x > window.innerWidth || y < 0 || y > window.innerHeight) {{
          item.element.style.opacity = '0';
          return;
        }}

        // Calculate zoom scale factor (fades UT labels when zoomed far out to prevent overcrowding)
        let scale = Math.min(Math.max(28.0 / camDistance, 0.7), 1.25);
        let opacity = 1.0;

        if (item.isUT && camDistance > 25.0) {{
          opacity = Math.max(0.0, 1.0 - (camDistance - 25.0) / 10.0);
        }}

        item.element.style.opacity = opacity.toFixed(2);
        item.element.style.transform = `translate(-50%, -50%) scale(${{scale.toFixed(2)}})`;
        item.element.style.left = `${{x}}px`;
        item.element.style.top = `${{y}}px`;
      }});
    }}

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('click', onPointerClick);
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      updateHTMLStateLabels();
      renderer.render(scene, camera);
    }}

    animate();
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[OK] Generated Dynamic State Label 3D Digital Twin HTML artifact: '{output_html_path}'.")


if __name__ == "__main__":
    generate_3d_map_html()
