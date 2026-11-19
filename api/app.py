"""FastAPI Backend Application for PathSnap GPS Map Matching.

Integrates:
  - Person 1: Candidate generation & spatial UTM indexing
  - Person 2: Feature engineering & HistGradientBoosting classifier scoring
  - Person 3: Hidden Markov Model with Viterbi trellis decoding
  - Person 4: REST API endpoints and static UI serving
"""
import io
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from hmm.decoder import FullPipelineDecoder

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT / "ui"
DATA_DIR = PROJECT_ROOT / "data"

# Initialize FastAPI app
app = FastAPI(
    title="PathSnap Map Matching API",
    description="High-performance GPS-to-road map matching API using Machine Learning and Hidden Markov Models.",
    version="1.0.0",
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Decoder
decoder: Optional[FullPipelineDecoder] = None
try:
    model_file = PROJECT_ROOT / "artifacts" / "final_model.joblib"
    if model_file.exists():
        decoder = FullPipelineDecoder(model_path=model_file)
        print("PathSnap Map Matching Pipeline initialized successfully.")
except Exception as e:
    print(f"Warning: Failed to load decoder on startup: {e}")

# Multi-client activity broadcast state
server_state: Dict[str, Any] = {
    "latest_activity": None,
    "connected_clients": {},
    "request_history": [],
}


def register_client(request: Optional[Request], action: str = "ping") -> str:
    """Track connected clients across local network (ignoring server localhost)."""
    if request and request.client:
        ip = request.client.host
        # Exclude server loopback addresses so only external laptops count
        if ip in ("127.0.0.1", "localhost", "::1"):
            return ip
        now = time.time()
        if ip not in server_state["connected_clients"]:
            server_state["connected_clients"][ip] = {
                "ip": ip,
                "first_seen": now,
                "requests_count": 0,
            }
        server_state["connected_clients"][ip]["last_seen"] = now
        server_state["connected_clients"][ip]["requests_count"] += 1
        server_state["connected_clients"][ip]["last_action"] = action
        return ip
    return "127.0.0.1"


# Pydantic Schemas
class GPSCoordinate(BaseModel):
    latitude: float = Field(..., description="Latitude in degrees [-90, 90]")
    longitude: float = Field(..., description="Longitude in degrees [-180, 180]")
    timestamp: Optional[float] = Field(None, description="Epoch timestamp in seconds")

    @field_validator("latitude")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"Latitude {v} out of range [-90, 90]")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError(f"Longitude {v} out of range [-180, 180]")
        return v


class MatchRequest(BaseModel):
    trip_id: Optional[str] = Field("00000001", description="Reference trip ID for road network")
    partition: Optional[str] = Field("student_test", description="'student_test' or 'train'")
    coordinates: List[GPSCoordinate] = Field(..., min_length=2, description="List of GPS trajectory coordinates")


class MatchResponse(BaseModel):
    trip_id: str
    point_count: int
    total_distance_m: float
    execution_time_sec: float
    mean_confidence: float
    ground_truth_accuracy: Optional[float] = None
    matched_arcs: List[int]
    unique_route_arcs: List[int]
    matched_coordinates: List[List[float]]  # [[lat, lon], ...]
    raw_coordinates: List[List[float]]      # [[lat, lon], ...]
    confidences: List[float]


# API Endpoints
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Health status and pipeline readiness check."""
    return {
        "status": "online",
        "service": "PathSnap GPS Map Matching API",
        "version": "1.0.0",
        "model_loaded": decoder is not None and decoder.model is not None,
        "available_evaluator_test_trips": len(list((DATA_DIR / "evaluator_test").glob("*"))) if (DATA_DIR / "evaluator_test").exists() else 0,
        "available_student_test_trips": len(list((DATA_DIR / "student_test").glob("*"))) if (DATA_DIR / "student_test").exists() else 0,
        "available_train_trips": len(list((DATA_DIR / "train").glob("*"))) if (DATA_DIR / "train").exists() else 0,
    }


@app.get("/api/trips")
def list_sample_trips() -> Dict[str, Any]:
    """List available sample trips prioritizing student_test and train from Student folder."""
    trips = []
    for partition in ["student_test", "train", "evaluator_test"]:
        p_dir = DATA_DIR / partition
        if p_dir.exists():
            for t_path in sorted(p_dir.iterdir()):
                if t_path.is_dir():
                    trk_file = t_path / f"{t_path.name}.track"
                    pts_count = 0
                    if trk_file.exists():
                        with open(trk_file) as f:
                            pts_count = sum(1 for _ in f)
                    trips.append({
                        "trip_id": t_path.name,
                        "partition": partition,
                        "point_count": pts_count,
                        "has_route": (t_path / f"{t_path.name}.route").exists(),
                    })
    return {"trips": trips, "total_trips": len(trips)}


@app.get("/api/trip/{partition}/{trip_id}")
def get_sample_trip(partition: str, trip_id: str) -> Dict[str, Any]:
    """Fetch raw GPS coordinates and metadata for a sample trip."""
    trip_path = DATA_DIR / partition / trip_id
    if not trip_path.exists():
        found = False
        for alt_part in ["evaluator_test", "student_test", "train"]:
            if (DATA_DIR / alt_part / trip_id).exists():
                trip_path = DATA_DIR / alt_part / trip_id
                partition = alt_part
                found = True
                break
        if not found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trip {trip_id} not found in evaluator_test, student_test, or train",
            )

    track_file = trip_path / f"{trip_id}.track"
    if not track_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track file for trip {trip_id} not found",
        )

    df = pd.read_csv(track_file, sep="\t", header=None)
    coords = [
        {"latitude": float(row[1]), "longitude": float(row[0]), "timestamp": float(row[2])}
        for _, row in df.iterrows()
    ]

    has_route = (trip_path / f"{trip_id}.route").exists()

    return {
        "trip_id": trip_id,
        "partition": partition,
        "point_count": len(coords),
        "has_route": has_route,
        "coordinates": coords,
    }


@app.post("/api/match", response_model=MatchResponse)
def match_trajectory(request: MatchRequest, http_req: Request):
    """Match GPS trajectory points using the complete Person 1-2-3 pipeline."""
    if decoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Map matching pipeline is not initialized.",
        )

    if len(request.coordinates) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trajectory must contain at least 2 GPS observations.",
        )

    client_ip = http_req.client.host if (http_req and http_req.client) else "127.0.0.1"
    print(f"\n" + "=" * 68, flush=True)
    print(f"  [CLIENT ACTIVITY] Inbound match from: {client_ip} at {time.strftime('%H:%M:%S')}", flush=True)
    print(f"  Trip: {request.trip_id} ({request.partition}) | GPS Points: {len(request.coordinates)}", flush=True)
    print("  Processing: Spatial Candidates -> Feature Extraction -> HistGBM -> Viterbi...", flush=True)

    t0 = time.time()
    points = [
        {"lat": c.latitude, "lon": c.longitude, "ts": c.timestamp if c.timestamp is not None else idx}
        for idx, c in enumerate(request.coordinates)
    ]

    try:
        # Check if this matches a full pre-existing trip folder
        trip_dir = DATA_DIR / request.partition / request.trip_id
        if trip_dir.exists() and len(request.coordinates) > 50:
            # Full trip match
            res = decoder.match_trip(trip_dir)
        else:
            # In-memory raw points match against road network
            res = decoder.match_raw_coordinates(
                points=points,
                reference_trip_id=request.trip_id,
                partition=request.partition,
                data_dir=DATA_DIR,
            )

        if "error" in res:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=res["error"],
            )

        exec_time = round(time.time() - t0, 3)

        print(f"  >>> RESULT: {res['point_count']} points snapped to {len(res['unique_route_arcs'])} unique arcs", flush=True)
        if res.get("ground_truth_accuracy") is not None:
            acc = res["ground_truth_accuracy"]
            acc_str = f"{acc:.2f}%" if acc > 1.0 else f"{acc:.2%}"
            print(f"  >>> Sequence Accuracy: {acc_str}", flush=True)
        print(f"  >>> Duration: {exec_time}s | Mean Confidence: {res['mean_confidence']:.1%}", flush=True)
        print("=" * 68 + "\n", flush=True)

        resp = MatchResponse(
            trip_id=res["trip_id"],
            point_count=res["point_count"],
            total_distance_m=res["total_distance_m"],
            execution_time_sec=exec_time,
            mean_confidence=res["mean_confidence"],
            ground_truth_accuracy=res.get("ground_truth_accuracy"),
            matched_arcs=res["matched_arcs"],
            unique_route_arcs=res["unique_route_arcs"],
            matched_coordinates=[[c[0], c[1]] for c in res["matched_coordinates"]],
            raw_coordinates=[[c[0], c[1]] for c in res["raw_coordinates"]],
            confidences=res["confidences"],
        )

        # Record into request history
        server_state["request_history"].append({
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S"),
            "client_ip": client_ip,
            "trip_id": res["trip_id"],
            "points": res["point_count"],
            "unique_arcs": len(res["unique_route_arcs"]),
            "accuracy": res.get("ground_truth_accuracy"),
            "latency_sec": exec_time,
        })

        # Broadcast for multi-laptop sync
        server_state["latest_activity"] = {
            "timestamp": time.time(),
            "client_ip": client_ip,
            "trip_id": res["trip_id"],
            "partition": request.partition,
            "point_count": res["point_count"],
            "total_distance_m": res["total_distance_m"],
            "execution_time_sec": exec_time,
            "mean_confidence": res["mean_confidence"],
            "ground_truth_accuracy": res.get("ground_truth_accuracy"),
            "matched_arcs": res["matched_arcs"],
            "unique_route_arcs": res["unique_route_arcs"],
            "matched_coordinates": [[c[0], c[1]] for c in res["matched_coordinates"]],
            "raw_coordinates": [[c[0], c[1]] for c in res["raw_coordinates"]],
            "confidences": res["confidences"],
        }

        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution error: {str(e)}",
        )


@app.get("/api/latest-activity")
def get_latest_activity(http_req: Request):
    """Fetch the most recent match activity performed by any client."""
    register_client(http_req, "poll-activity")
    return server_state["latest_activity"] or {"status": "idle"}


@app.get("/api/server-stats")
def get_server_stats(http_req: Request):
    """Telemetry endpoint for Server Monitor view."""
    register_client(http_req, "check-stats")
    now = time.time()
    clients = []
    for ip, info in list(server_state["connected_clients"].items()):
        # Consider client active if heartbeat/request seen within last 10 seconds
        is_active = (now - info["last_seen"]) < 10
        if not is_active:
            # Drop dead clients from active tracking
            info["is_active"] = False
        else:
            info["is_active"] = True

        clients.append({
            "ip": ip,
            "is_active": info["is_active"],
            "requests_count": info["requests_count"],
            "last_seen_sec_ago": round(now - info["last_seen"], 1),
            "last_action": info.get("last_action", "active"),
        })

    active_count = sum(1 for c in clients if c["is_active"])

    return {
        "server_ip": "10.20.202.73",
        "port": 8000,
        "status": "online",
        "model_loaded": decoder is not None and decoder.model is not None,
        "clients": clients,
        "active_client_count": active_count,
        "request_history": server_state["request_history"][-20:],
        "latest_activity": server_state["latest_activity"],
    }


@app.get("/api/client-heartbeat")
def client_heartbeat(http_req: Request):
    """Periodic heartbeat from active client tabs."""
    ip = register_client(http_req, "heartbeat")
    return {"status": "ok", "client_ip": ip, "timestamp": time.time()}


@app.post("/api/client-disconnect")
@app.get("/api/client-disconnect")
def client_disconnect(http_req: Request):
    """Triggered on client unload or tab closure."""
    if http_req and http_req.client:
        ip = http_req.client.host
        if ip in server_state["connected_clients"]:
            del server_state["connected_clients"][ip]
    server_state["latest_activity"] = None
    return {"status": "disconnected"}


@app.post("/api/server-reset")
def reset_server_state():
    """Reset live match activity cache and counter for demo restarts."""
    server_state["latest_activity"] = None
    server_state["request_history"].clear()
    return {"status": "ok", "message": "Server state reset successfully."}


@app.post("/api/upload-and-match")
async def upload_and_match(
    http_req: Request,
    file: UploadFile = File(...),
    trip_id: str = Form("00000001"),
    partition: str = Form("student_test"),
):
    """Upload a .track or .csv file and match it against the road network."""
    if decoder is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    content = await file.read()
    try:
        # Detect whether tab-separated or comma-separated
        text = content.decode("utf-8")
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if not lines:
            raise ValueError("Uploaded file is empty.")

        sep = "\t" if "\t" in lines[0] else ","
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None)

        # Detect columns: lon, lat, ts
        if df.shape[1] < 2:
            raise ValueError("File must contain at least 2 columns: longitude and latitude.")

        # Coordinate detection: lon usually in column 0 or 1
        col0_vals = pd.to_numeric(df[0], errors="coerce")
        col1_vals = pd.to_numeric(df[1], errors="coerce")

        # In PathSnap .track and GIS datasets, Col 0 is Longitude, Col 1 is Latitude
        if col1_vals.abs().max() > 90.0:
            lats = col0_vals.to_numpy()
            lons = col1_vals.to_numpy()
        else:
            lons = col0_vals.to_numpy()
            lats = col1_vals.to_numpy()

        ts = df[2].to_numpy() if df.shape[1] >= 3 else np.arange(len(df))

        points = [
            GPSCoordinate(latitude=float(lat), longitude=float(lon), timestamp=float(t))
            for lat, lon, t in zip(lats, lons, ts)
            if not np.isnan(lat) and not np.isnan(lon)
        ]

        req = MatchRequest(trip_id=trip_id, partition=partition, coordinates=points)
        return match_trajectory(req, http_req=http_req)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")


# Mount UI static directory if it exists
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    @app.get("/")
    def serve_root(http_req: Request):
        client_ip = http_req.client.host if (http_req and http_req.client) else "127.0.0.1"
        is_local = client_ip in ("127.0.0.1", "localhost", "::1")
        target_file = (UI_DIR / "server.html") if is_local else (UI_DIR / "client.html")
        if target_file.exists():
            return FileResponse(target_file)
        index_file = UI_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "UI not found"}

    @app.get("/server")
    def serve_server():
        server_file = UI_DIR / "server.html"
        if server_file.exists():
            return FileResponse(server_file)
        return FileResponse(UI_DIR / "index.html")

    @app.get("/client")
    def serve_client():
        client_file = UI_DIR / "client.html"
        if client_file.exists():
            return FileResponse(client_file)
        return FileResponse(UI_DIR / "index.html")
