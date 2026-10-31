
import { Viewer, CameraFlyTo } from "resium";
import { Cartesian3 } from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

export default function App() {
  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      <Viewer
        full
        baseLayer={false}
        animation={false}
        timeline={false}
        geocoder={false}
        homeButton={false}
        sceneModePicker={false}
        navigationHelpButton={false}
      >
        <CameraFlyTo
          destination={Cartesian3.fromDegrees(
            78.9629,
            22.5937,
            6500000
          )}
          duration={0}
        />
      </Viewer>
    </div>
  );
}
