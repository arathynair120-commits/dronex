import { useEffect, useState } from "react";
import "./App.css";


function App() {

  // ==========================================================
  // Backend state
  // ==========================================================

  const [health, setHealth] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [risk, setRisk] = useState([]);
  const [results, setResults] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [frameVersion, setFrameVersion] = useState(
    Date.now()
  );


  // ==========================================================
  // Load data from FastAPI backend
  // ==========================================================

  async function loadBackendData() {

    try {

      setError(null);

      const [
        healthResponse,
        telemetryResponse,
        riskResponse,
        resultsResponse,
      ] = await Promise.all([

        fetch("/api/health"),

        fetch("/api/telemetry"),

        fetch("/api/risk"),

        fetch("/api/results"),

      ]);


      // ------------------------------------------------------
      // Health
      // ------------------------------------------------------

      if (healthResponse.ok) {

        const healthData =
          await healthResponse.json();

        setHealth(healthData);

      }


      // ------------------------------------------------------
      // Telemetry
      // ------------------------------------------------------

      if (telemetryResponse.ok) {

        const telemetryData =
          await telemetryResponse.json();

        setTelemetry(telemetryData);

      }


      // ------------------------------------------------------
      // Risk
      // ------------------------------------------------------

      if (riskResponse.ok) {

        const riskData =
          await riskResponse.json();

        setRisk(
          riskData.objects || []
        );

      }


      // ------------------------------------------------------
      // Pipeline results
      // ------------------------------------------------------

      if (resultsResponse.ok) {

        const resultData =
          await resultsResponse.json();

        setResults(resultData);

      }

    } catch (err) {

      console.error(
        "Backend connection error:",
        err
      );

      setError(
        "Unable to connect to AeroSentinel backend."
      );

    } finally {

      setLoading(false);

    }
  }


  // ==========================================================
  // Load on startup + refresh every 5 seconds
  // ==========================================================

  useEffect(() => {

    loadBackendData();

    const timer = setInterval(
      loadBackendData,
      5000
    );

    return () => {
      clearInterval(timer);
    };

  }, []);


  // ==========================================================
  // Derived risk values
  // ==========================================================

  const highCount =
    risk.filter(
      item =>
        item.risk_level === "HIGH"
    ).length;


  const cautionCount =
    risk.filter(
      item =>
        item.risk_level === "CAUTION"
    ).length;


  const lowCount =
    risk.filter(
      item =>
        item.risk_level === "LOW"
    ).length;


  const objectCount =
    risk.length;


  // ==========================================================
  // Overall awareness
  // ==========================================================

  const awareness =
    highCount > 0
      ? "HIGH RISK"
      : cautionCount > 0
        ? "CAUTION"
        : "NORMAL";


  // ==========================================================
  // Loading screen
  // ==========================================================

  if (loading) {

    return (

      <div className="app">

        <div
          style={{
            padding: "50px",
            fontFamily: "Arial",
          }}
        >

          CONNECTING TO AEROSENTINEL BACKEND...

        </div>

      </div>

    );
  }


  return (

    <div className="app">

      {/* =====================================================
          HEADER
      ====================================================== */}

      <header className="topbar">

        <div>

          <h1>
            AEROSENTINEL
          </h1>

          <p>
            FLIGHT-AWARE UAV INTELLIGENCE
          </p>

        </div>


        <div className="system-status">

          <span className="status-dot" />

          {health?.status === "online"
            ? "SYSTEM ONLINE"
            : "BACKEND OFFLINE"}

        </div>

      </header>


      {/* =====================================================
          ERROR
      ====================================================== */}

      {error && (

        <div
          style={{
            padding: "12px",
            marginBottom: "15px",
            background: "#241314",
            border: "1px solid #713f3f",
            color: "#ff9b9b",
          }}
        >

          {error}

        </div>

      )}


      <main className="dashboard">


        {/* ===================================================
            LEFT PANEL
        ==================================================== */}

        <section className="main-panel">


          <div className="panel-header">

            <div>

              <h2>
                AERIAL PERCEPTION
              </h2>

              <span>
                AU-AIR RECORDED FRAME
              </span>

            </div>


            <div className="frame-id">

              {results
                ? `FRAMES ${results.frames_processed}`
                : "NO DATA"}

            </div>

          </div>


          {/* =================================================
              REAL BACKEND FRAME
          ================================================== */}

          <div className="image-container">

            <img
              src={`/api/frame?v=${frameVersion}`}
              alt="AeroSentinel aerial analysis"
            />

            <div className="image-overlay">

              AI VISION + FLIGHT AWARENESS ACTIVE

            </div>

          </div>


          {/* =================================================
              DETECTION SUMMARY
          ================================================== */}

          <div className="detection-summary">

            <h3>
              DETECTION / RISK SUMMARY
            </h3>


            <div className="detection-grid">


              <div className="detection-card">

                <div>

                  <strong>
                    {objectCount}
                  </strong>

                  <span>
                    OBJECTS
                  </span>

                </div>

                <small>
                  TRACKED
                </small>

              </div>


              <div className="detection-card">

                <div>

                  <strong>
                    {highCount}
                  </strong>

                  <span>
                    HIGH
                  </span>

                </div>

                <small>
                  RISK
                </small>

              </div>


              <div className="detection-card">

                <div>

                  <strong>
                    {cautionCount}
                  </strong>

                  <span>
                    CAUTION
                  </span>

                </div>

                <small>
                  RISK
                </small>

              </div>


              <div className="detection-card">

                <div>

                  <strong>
                    {lowCount}
                  </strong>

                  <span>
                    LOW
                  </span>

                </div>

                <small>
                  RISK
                </small>

              </div>


            </div>

          </div>

        </section>


        {/* ===================================================
            RIGHT PANEL
        ==================================================== */}

        <aside className="side-panel">


          {/* =================================================
              UAV STATUS
          ================================================== */}

          <section className="status-panel">

            <div className="panel-title">

              <h2>
                UAV STATUS
              </h2>

              <span>
                TELEMETRY STREAM
              </span>

            </div>


            <div className="telemetry-grid">


              <div className="telemetry-item">

                <span>
                  ALTITUDE
                </span>

                <strong>

                  {telemetry
                    ? telemetry.altitude.toFixed(2)
                    : "--"}

                  <small>
                    units
                  </small>

                </strong>

              </div>


              <div className="telemetry-item">

                <span>
                  VELOCITY
                </span>

                <strong>

                  {telemetry
                    ? telemetry.total_speed.toFixed(3)
                    : "--"}

                  <small>
                    units/s
                  </small>

                </strong>

              </div>


              <div className="telemetry-item">

                <span>
                  HEADING
                </span>

                <strong>

                  {telemetry
                    ? telemetry.heading.toFixed(1)
                    : "--"}°

                </strong>

              </div>


              <div className="telemetry-item">

                <span>
                  OBJECTS
                </span>

                <strong>
                  {objectCount}
                </strong>

              </div>

            </div>


            <div className="gps">

              <span>
                GPS POSITION
              </span>

              <strong>

                {telemetry

                  ? `${telemetry.latitude.toFixed(5)}° N, ${telemetry.longitude.toFixed(5)}° E`

                  : "NO TELEMETRY"}

              </strong>

            </div>

          </section>


          {/* =================================================
              FLIGHT AWARENESS
          ================================================== */}

          <section className="awareness-panel">

            <div className="panel-title">

              <h2>
                FLIGHT AWARENESS
              </h2>

              <span>
                CONTEXT ENGINE
              </span>

            </div>


            <div className="awareness-state">

              <div className="awareness-icon">

                !

              </div>


              <div>

                <span>
                  CURRENT STATE
                </span>

                <h3>
                  {awareness}
                </h3>

              </div>

            </div>


            <p className="awareness-text">

              Visual detections are combined with
              object motion, UAV telemetry and the
              projected flight corridor to estimate
              short-term flight-path hazard relevance.

            </p>


            <div className="awareness-metrics">


              <div>

                <span>
                  VISUAL OBJECTS
                </span>

                <strong>
                  {objectCount}
                </strong>

              </div>


              <div>

                <span>
                  TRACKS
                </span>

                <strong>

                  {results?.tracked_objects || 0}

                </strong>

              </div>

            </div>

          </section>


          {/* =================================================
              OBJECT RISK
          ================================================== */}

          <section className="pipeline-panel">

            <div className="panel-title">

              <h2>
                OBJECT RISK
              </h2>

              <span>
                FLIGHT-AWARE
              </span>

            </div>


            <div
              style={{
                marginTop: "15px",
              }}
            >

              {risk.length === 0 && (

                <div
                  style={{
                    padding: "15px",
                    color: "#758390",
                  }}
                >

                  No tracked objects available.

                </div>

              )}


              {risk.map(
                item => (

                  <div
                    key={item.track_id}
                    style={{
                      padding: "11px",
                      marginBottom: "7px",
                      background: "#111820",
                      border:
                        "1px solid #202b36",
                    }}
                  >

                    <div
                      style={{
                        display: "flex",
                        justifyContent:
                          "space-between",
                        alignItems:
                          "center",
                      }}
                    >

                      <strong>

                        {item.class_name}
                        {" #"}
                        {item.track_id}

                      </strong>


                      <strong>

                        {(
                          item.risk_score * 100
                        ).toFixed(1)}

                      </strong>

                    </div>


                    <div
                      style={{
                        marginTop: "5px",
                        fontSize: "9px",
                        color: "#758390",
                      }}
                    >

                      {item.risk_level}
                      {" — "}
                      {item.reason}

                    </div>


                    <div
                      style={{
                        marginTop: "7px",
                        fontSize: "8px",
                        color: "#53616d",
                      }}
                    >

                      Corridor:
                      {" "}
                      {(
                        item.corridor_proximity * 100
                      ).toFixed(0)}%
                      {" | "}
                      Alignment:
                      {" "}
                      {(
                        item.motion_alignment * 100
                      ).toFixed(0)}%
                      {" | "}
                      Persistence:
                      {" "}
                      {(
                        item.persistence * 100
                      ).toFixed(0)}%

                    </div>

                  </div>

                )
              )}

            </div>

          </section>


          {/* =================================================
              AI PIPELINE
          ================================================== */}

          <section className="pipeline-panel">

            <div className="panel-title">

              <h2>
                AI PIPELINE
              </h2>

            </div>


            <div className="pipeline">


              <div className="pipeline-step active">

                <span>
                  01
                </span>

                IMAGE

              </div>


              <div className="arrow">
                →
              </div>


              <div className="pipeline-step active">

                <span>
                  02
                </span>

                YOLO

              </div>


              <div className="arrow">
                →
              </div>


              <div className="pipeline-step active">

                <span>
                  03
                </span>

                TRACK

              </div>


              <div className="arrow">
                →
              </div>


              <div className="pipeline-step active">

                <span>
                  04
                </span>

                TELEMETRY

              </div>


              <div className="arrow">
                →
              </div>


              <div className="pipeline-step active">

                <span>
                  05
                </span>

                RISK

              </div>

            </div>

          </section>


        </aside>

      </main>


      <footer>

        <span>
          AEROSENTINEL
        </span>

        <span>
          AU-AIR MULTIMODAL UAV DATASET
        </span>

        <span>
          YOLO + BYTETRACK + TELEMETRY + RISK
        </span>

      </footer>

    </div>

  );
}


export default App;