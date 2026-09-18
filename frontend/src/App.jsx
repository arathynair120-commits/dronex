import "./App.css";

function App() {
  return (
    <div className="app">

      {/* HEADER */}
      <header className="header">
        <div>
          <h1>AEROSENTINEL</h1>
          <p>Flight-Aware UAV Intelligence</p>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          SYSTEM ONLINE
        </div>
      </header>

      {/* MAIN DASHBOARD */}
      <main className="dashboard">

        {/* CAMERA / DETECTION AREA */}
        <section className="camera-panel">
          <div className="panel-title">
            <h2>AERIAL VIEW</h2>
            <span>RECORDED DATA</span>
          </div>

          <div className="camera-view">
            <div className="camera-placeholder">
              <div className="crosshair">+</div>
              <p>UAV CAMERA FEED</p>
              <small>Detection visualization</small>
            </div>

            {/* Example detection boxes */}
            <div className="detection-box car">
              <span>CAR 91%</span>
            </div>

            <div className="detection-box human">
              <span>HUMAN 87%</span>
            </div>
          </div>
        </section>

        {/* UAV STATUS */}
        <section className="status-panel">
          <div className="panel-title">
            <h2>UAV STATUS</h2>
          </div>

          <div className="stat">
            <span>ALTITUDE</span>
            <strong>42.3 m</strong>
          </div>

          <div className="stat">
            <span>VELOCITY</span>
            <strong>5.2 m/s</strong>
          </div>

          <div className="stat">
            <span>GPS</span>
            <strong>12.9698, 79.1559</strong>
          </div>

          <div className="stat">
            <span>OBJECTS DETECTED</span>
            <strong>7</strong>
          </div>
        </section>

        {/* DETECTIONS */}
        <section className="detections-panel">
          <div className="panel-title">
            <h2>DETECTIONS</h2>
          </div>

          <div className="detection-stats">
            <div>
              <strong>4</strong>
              <span>CAR</span>
            </div>

            <div>
              <strong>2</strong>
              <span>HUMAN</span>
            </div>

            <div>
              <strong>1</strong>
              <span>TRUCK</span>
            </div>

            <div>
              <strong>0</strong>
              <span>BUS</span>
            </div>
          </div>
        </section>

        {/* AWARENESS */}
        <section className="awareness-panel">
          <span>FLIGHT AWARENESS</span>

          <div className="awareness-status">
            ⚠ CAUTION
          </div>

          <p>
            Objects detected within the current flight context.
          </p>
        </section>

      </main>

      <footer>
        AeroSentinel • UAV Situational Awareness Prototype
      </footer>

    </div>
  );
}

export default App;