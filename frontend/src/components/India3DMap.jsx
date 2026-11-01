import React, { useMemo, useState, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';

const REGION_COLORS = {
  'North': '#2563eb',
  'South': '#059669',
  'East': '#d97706',
  'West': '#db2777',
  'Central': '#7c3aed',
  'North-East': '#0891b2'
};

const CENTER_LON = 82.5;
const CENTER_LAT = 22.5;
const SCALE_FACTOR = 0.8;

function projectLon(lon) {
  return (lon - CENTER_LON) * SCALE_FACTOR;
}

function projectLat(lat) {
  return (lat - CENTER_LAT) * SCALE_FACTOR;
}

function State3DMesh({ feature, isSelected, onSelect, showBorders, showLabels }) {
  const props = feature.properties;
  const stateId = feature.id || props.iso_code;
  const region = props.REGION || 'Central';
  const baseColor = REGION_COLORS[region] || '#64748b';

  const [hovered, setHovered] = useState(false);

  const extrudeGeometry = useMemo(() => {
    const geometryType = feature.geometry.type;
    const coordinates = feature.geometry.coordinates;
    const shapes = [];

    function processPoly(polyCoords) {
      const outerRing = polyCoords[0];
      const shape = new THREE.Shape();
      outerRing.forEach((pt, i) => {
        const x = projectLon(pt[0]);
        const y = projectLat(pt[1]);
        if (i === 0) shape.moveTo(x, y);
        else shape.lineTo(x, y);
      });
      shapes.push(shape);
    }

    if (geometryType === 'Polygon') processPoly(coordinates);
    else if (geometryType === 'MultiPolygon') coordinates.forEach(processPoly);

    return new THREE.ExtrudeGeometry(shapes, {
      depth: 0.6,
      bevelEnabled: true,
      bevelSegments: 2,
      bevelSize: 0.03,
      bevelThickness: 0.03
    });
  }, [feature]);

  // Edges geometry with polygonOffset to prevent depth buffer z-fighting
  const edgesGeometry = useMemo(() => {
    return new THREE.EdgesGeometry(extrudeGeometry, 18);
  }, [extrudeGeometry]);

  const currentColor = isSelected ? '#ef4444' : (hovered ? '#3b82f6' : baseColor);
  const currentZ = isSelected ? 0.35 : (hovered ? 0.15 : 0.0);

  return (
    <group position={[0, 0, currentZ]}>
      <mesh
        geometry={extrudeGeometry}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(feature);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHovered(false);
        }}
        castShadow
        receiveShadow
      >
        <meshStandardMaterial
          color={currentColor}
          roughness={0.4}
          metalness={0.1}
        />
      </mesh>

      {/* Unlit State Boundary Line Segments with Anti-Z-Fighting PolygonOffset */}
      {showBorders && (
        <lineSegments geometry={edgesGeometry} position={[0, 0, 0.61]} renderOrder={10}>
          <lineBasicMaterial
            color="#94a3b8"
            linewidth={1.5}
            polygonOffset
            polygonOffsetFactor={-1}
            polygonOffsetUnits={-1}
          />
        </lineSegments>
      )}

      {/* State Centroid HTML Label */}
      {showLabels && props.centroid && !props.is_union_territory && (
        <Html
          position={[projectLon(props.centroid[1]), projectLat(props.centroid[0]), 0.8]}
          center
          distanceFactor={35}
        >
          <div className="bg-slate-900/85 text-slate-100 text-[10px] font-bold px-2 py-0.5 rounded border border-slate-700/60 shadow-md whitespace-nowrap pointer-events-none select-none">
            {props.st_nm || props.state_name}
          </div>
        </Html>
      )}

      {/* Hover Tooltip */}
      {hovered && props.centroid && (
        <Html position={[projectLon(props.centroid[1]), projectLat(props.centroid[0]), 1.4]} center>
          <div className="bg-slate-950/95 text-white text-xs px-3 py-2 rounded-lg shadow-2xl backdrop-blur border border-slate-700 font-medium pointer-events-none whitespace-nowrap z-50">
            <div className="font-bold text-blue-300">{props.state_name}</div>
            <div className="text-[10px] text-emerald-400 font-mono">ISO Code: {props.iso_code}</div>
            <div className="text-[10px] text-slate-300">{props.TYPE} ({props.REGION} Region)</div>
          </div>
        </Html>
      )}
    </group>
  );
}

export default function India3DMap({ geojsonData, selectedStateId, onSelectState, showBorders = true, showLabels = true }) {
  const controlsRef = useRef();

  return (
    <Canvas
      camera={{ position: [0, -22, 26], fov: 45 }}
      shadows
      className="w-full h-full bg-[#070a12]"
    >
      <ambientLight intensity={0.7} />
      <directionalLight position={[20, 30, 40]} intensity={0.75} castShadow />
      <directionalLight position={[-25, -25, 25]} intensity={0.4} color="#38bdf8" />

      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.05}
        maxPolarAngle={Math.PI / 2.05}
        minDistance={6}
        maxDistance={85}
      />

      <group>
        {geojsonData?.features?.map((feature) => (
          <State3DMesh
            key={feature.id || feature.properties.iso_code}
            feature={feature}
            isSelected={(feature.id || feature.properties.iso_code) === selectedStateId}
            onSelect={onSelectState}
            showBorders={showBorders}
            showLabels={showLabels}
          />
        ))}
      </group>
    </Canvas>
  );
}
