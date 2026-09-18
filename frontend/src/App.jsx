import { useEffect, useState } from "react";
import "./App.css";

const detections = [
  { label: "Car", confidence: 94, count: 5 },
  { label: "Human", confidence: 91, count: 2 },
  { label: "Truck", confidence: 88, count: 1 },
  { label: "Motorbike", confidence: 86, count: 1 },
];

function App() {
  const [altitude, setAltitude] = useState(42.3);
  const [velocity, setVelocity] = useState(5.2);
  const [heading, setHeading] = useState(127);
  const [frame, setFrame] = useState(4827);

  useEffect(() => {
    const timer = setInterval(() => {
      setAltitude((value) => +(value + (Math.random() - 0.5) * 0.4).toFixed(1));
      setVelocity((value) => +(value + (Math.random() - 0.5) * 0.2).toFixed(1));
      setHeading((value) => (value + (Math.random() - 0.5) * 2 + 360) % 360);
      setFrame((value) => value + 1);
    }, 2000);

    return () => clearInterval(timer);
  }, []);

  const awareness =
    velocity > 5.8 ? "CAUTION" : altitude < 35 ? "CAUTION" : "NORMAL";

  return (
    <div className="app">

      <header className="topbar">
        <div>
          <h1>AEROSENTINEL</h1>
          <p>FLIGHT-AWARE UAV INTELLIGENCE</p>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          SYSTEM ONLINE
        </div>
      </header>

      <main className="dashboard">

        {/* LEFT */}

        <section className="main-panel">

          <div className="panel-header">
            <div>
              <h2>AERIAL PERCEPTION</h2>
              <span>AU-AIR RECORDED FRAME</span>
            </div>

            <div className="frame-id">
              FRAME {frame}
            </div>
          </div>

          <div className="image-container">

            <img
              src="/image/sample_1.jpg"
              alt="Aerial UAV scene"
            />

            <div className="detection-box box-one">
              <span>CAR 94%</span>
            </div>

            <div className="detection-box box-two">
              <span>HUMAN 91%</span>
            </div>

            <div className="detection-box box-three">
              <span>CAR 89%</span>
            </div>

            <div className="image-overlay">
              AI VISION ACTIVE
            </div>

          </div>

          <div className="detection-summary">

            <h3>DETECTION SUMMARY</h3>

            <div className="detection-grid">

              {detections.map((item) => (
                <div className="detection-card" key={item.label}>

                  <div>
                    <strong>{item.count}</strong>
                    <span>{item.label}</span>
                  </div>

                  <small>{item.confidence}% CONF.</small>

                </div>
              ))}

            </div>

          </div>

        </section>

        {/* RIGHT */}

        <aside className="side-panel">

          <section className="status-panel">

            <div className="panel-title">
              <h2>UAV STATUS</h2>
              <span>TELEMETRY STREAM</span>
            </div>

            <div className="telemetry-grid">

              <div className="telemetry-item">
                <span>ALTITUDE</span>
                <strong>
                  {altitude} <small>m</small>
                </strong>
              </div>

              <div className="telemetry-item">
                <span>VELOCITY</span>
                <strong>
                  {velocity} <small>m/s</small>
                </strong>
              </div>

              <div className="telemetry-item">
                <span>HEADING</span>
                <strong>
                  {heading.toFixed(0)}°
                </strong>
              </div>

              <div className="telemetry-item">
                <span>OBJECTS</span>
                <strong>9</strong>
              </div>

            </div>

            <div className="gps">
              <span>GPS POSITION</span>

              <strong>
                13.0827° N, 80.2707° E
              </strong>
            </div>

          </section>


          <section className="awareness-panel">

            <div className="panel-title">
              <h2>FLIGHT AWARENESS</h2>
              <span>CONTEXT ENGINE</span>
            </div>

            <div className="awareness-state">

              <div className="awareness-icon">
                !
              </div>

              <div>
                <span>CURRENT STATE</span>

                <h3>
                  {awareness}
                </h3>
              </div>

            </div>

            <p className="awareness-text">
              Visual detections are combined with UAV
              telemetry to provide contextual flight awareness.
            </p>

            <div className="awareness-metrics">

              <div>
                <span>VISUAL OBJECTS</span>
                <strong>9</strong>
              </div>

              <div>
                <span>HIGH CONFIDENCE</span>
                <strong>7</strong>
              </div>

            </div>

          </section>


          <section className="pipeline-panel">

            <div className="panel-title">
              <h2>AI PIPELINE</h2>
            </div>

            <div className="pipeline">

              <div className="pipeline-step active">
                <span>01</span>
                IMAGE
              </div>

              <div className="arrow">→</div>

              <div className="pipeline-step active">
                <span>02</span>
                YOLO
              </div>

              <div className="arrow">→</div>

              <div className="pipeline-step active">
                <span>03</span>
                TELEMETRY
              </div>

              <div className="arrow">→</div>

              <div className="pipeline-step active">
                <span>04</span>
                AWARENESS
              </div>

            </div>

          </section>

        </aside>

      </main>

      <footer>
        <span>AEROSENTINEL</span>
        <span>AU-AIR MULTIMODAL UAV DATASET</span>
        <span>PERCEPTION + TELEMETRY</span>
      </footer>

    </div>
  );
}

export default App;