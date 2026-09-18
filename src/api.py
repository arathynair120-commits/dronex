from pathlib import Path
import subprocess
import sys

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


# ============================================================
# AeroSentinel API
# ============================================================

app = FastAPI(
    title="AeroSentinel API",
    description="Flight-aware UAV hazard intelligence backend",
    version="1.0.0",
)


# ------------------------------------------------------------
# Allow React/Vite frontend
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PIPELINE_DIR = BASE_DIR / "outputs" / "pipeline"

RESULT_FILE = (
    PIPELINE_DIR /
    "aerosentinel_result.json"
)

RISK_FILE = (
    PIPELINE_DIR /
    "object_risk.csv"
)

TRACKING_FILE = (
    PIPELINE_DIR /
    "tracking_results.csv"
)

DEMO_IMAGE = (
    PIPELINE_DIR /
    "aerosentinel_demo.jpg"
)

TELEMETRY_FILE = (
    BASE_DIR /
    "outputs" /
    "telemetry" /
    "uav_motion.csv"
)


# ============================================================
# Health
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "online",
        "system": "AeroSentinel",
        "backend": "Python",
        "pipeline_available": RESULT_FILE.exists(),
    }


# ============================================================
# Current Results
# ============================================================

@app.get("/api/results")
def results():

    if not RESULT_FILE.exists():

        return JSONResponse(
            status_code=404,
            content={
                "error": "Pipeline results not found.",
                "message": "Run the AeroSentinel pipeline first."
            }
        )

    import json

    with open(
        RESULT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data


# ============================================================
# Risk Table
# ============================================================

@app.get("/api/risk")
def risk():

    if not RISK_FILE.exists():

        return JSONResponse(
            status_code=404,
            content={
                "error": "Risk results not found."
            }
        )

    df = pd.read_csv(
        RISK_FILE
    )

    return {
        "count": len(df),
        "objects": df.to_dict(
            orient="records"
        )
    }


# ============================================================
# Telemetry
# ============================================================

@app.get("/api/telemetry")
def telemetry():

    if not TELEMETRY_FILE.exists():

        return JSONResponse(
            status_code=404,
            content={
                "error": "Telemetry file not found."
            }
        )

    df = pd.read_csv(
        TELEMETRY_FILE
    )

    if df.empty:

        return {
            "available": False
        }

    row = df.iloc[0]

    horizontal_speed = (
        float(row["horizontal_speed"])
        if "horizontal_speed" in df.columns
        else (
            float(row["linear_x"]) ** 2
            + float(row["linear_y"]) ** 2
        ) ** 0.5
    )

    return {

        "available": True,

        "image_name":
            row["image_name"],

        "platform":
            row["platform"],

        "latitude":
            float(row["latitude"]),

        "longitude":
            float(row["longitude"]),

        "altitude":
            float(row["altitude"]),

        "linear_x":
            float(row["linear_x"]),

        "linear_y":
            float(row["linear_y"]),

        "linear_z":
            float(row["linear_z"]),

        "horizontal_speed":
            horizontal_speed,

        "total_speed":
            float(row["total_speed"])
            if "total_speed" in df.columns
            else (
                float(row["linear_x"]) ** 2
                + float(row["linear_y"]) ** 2
                + float(row["linear_z"]) ** 2
            ) ** 0.5,

        "heading":
            float(row["motion_heading_deg"]),

        "pitch":
            float(row["angle_theta"]),

        "roll":
            float(row["angle_phi"]),

        "yaw":
            float(row["angle_psi"]),
    }


# ============================================================
# Tracking
# ============================================================

@app.get("/api/tracking")
def tracking():

    if not TRACKING_FILE.exists():

        return JSONResponse(
            status_code=404,
            content={
                "error": "Tracking data not found."
            }
        )

    df = pd.read_csv(
        TRACKING_FILE
    )

    return {
        "records": len(df),
        "tracks": int(
            df["track_id"]
            .nunique()
        ),
        "detections": df.to_dict(
            orient="records"
        )
    }


# ============================================================
# Demo Frame
# ============================================================

@app.get("/api/frame")
def frame():

    if not DEMO_IMAGE.exists():

        return JSONResponse(
            status_code=404,
            content={
                "error": "Demo frame not found.",
                "message": "Run the pipeline first."
            }
        )

    return FileResponse(
        DEMO_IMAGE,
        media_type="image/jpeg"
    )


# ============================================================
# Run Pipeline
# ============================================================

@app.post("/api/run")
def run_pipeline():

    pipeline_script = (
        BASE_DIR /
        "src" /
        "pipeline.py"
    )

    if not pipeline_script.exists():

        return JSONResponse(
            status_code=500,
            content={
                "error": "pipeline.py not found."
            }
        )

    try:

        result = subprocess.run(
            [
                sys.executable,
                str(pipeline_script)
            ],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=900,
        )

        if result.returncode != 0:

            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout,
                }
            )

        return {
            "success": True,
            "message": "AeroSentinel pipeline completed.",
            "stdout": result.stdout,
        }

    except subprocess.TimeoutExpired:

        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "Pipeline timed out."
            }
        )


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "system": "AeroSentinel",
        "status": "online",
        "message": "Flight-aware UAV intelligence API"
    }