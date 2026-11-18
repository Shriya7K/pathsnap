
import csv
import io
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching Application",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_FILE = PROJECT_DIR / "ui" / "index.html"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def extract_gps_points(content: bytes):
    """Validate and extract GPS coordinates from a CSV file."""

    try:
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))

        if not reader.fieldnames:
            raise ValueError("CSV file has no header row")

        reader.fieldnames = [
            name.strip().lower() if name else ""
            for name in reader.fieldnames
        ]

        required = {"latitude", "longitude", "timestamp"}

        if not required.issubset(set(reader.fieldnames)):
            missing = required - set(reader.fieldnames)
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing))}"
            )

        points = []

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Malformed CSV row: {row_number}")

            timestamp = row.get("timestamp")

            if not timestamp or not timestamp.strip():
                raise ValueError(
                    f"Missing timestamp at row {row_number}"
                )

            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid GPS coordinates at row {row_number}"
                )

            if (
                not math.isfinite(latitude)
                or not -90 <= latitude <= 90
            ):
                raise ValueError(
                    f"Invalid latitude at row {row_number}"
                )

            if (
                not math.isfinite(longitude)
                or not -180 <= longitude <= 180
            ):
                raise ValueError(
                    f"Invalid longitude at row {row_number}"
                )

            points.append({
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": timestamp.strip(),
            })

        if not points:
            raise ValueError("CSV contains no GPS data rows")

        return points

    except UnicodeDecodeError:
        raise ValueError("CSV must use UTF-8 encoding")

    except csv.Error:
        raise ValueError("Unable to read malformed CSV file")


def parse_network_file(content: bytes, filename: str):
    """Read a whitespace-separated network file."""

    try:
        text = content.decode("utf-8-sig")

        data = np.loadtxt(
            io.StringIO(text),
            dtype=float,
        )

        data = np.atleast_2d(data)

    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise ValueError(
            f"Unable to read {filename}: {error}"
        )

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"{filename} must contain exactly two columns"
        )

    if len(data) == 0 or not np.isfinite(data).all():
        raise ValueError(
            f"{filename} contains no valid data or invalid numbers"
        )

    return data


@app.get("/")
async def home():
    if not FRONTEND_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend file not found.",
        )

    return FileResponse(FRONTEND_FILE)


@app.get("/health")
async def health():
    return {
        "status": "running",
        "application": "PathSnap",
    }


@app.post("/upload")
async def upload_trip(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file",
        )

    safe_name = Path(file.filename).name
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    try:
        points = extract_gps_points(content)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    destination = UPLOAD_DIR / unique_name
    destination.write_bytes(content)

    return {
        "status": "success",
        "message": "GPS CSV validated and uploaded successfully",
        "filename": safe_name,
        "stored_filename": unique_name,
        "size_bytes": len(content),
        "gps_points": len(points),
        "gps_coordinates": points,
        "note": "Map matching is available through the /match endpoint.",
    }


@app.post("/match")
async def match_trip(
    gps_file: UploadFile = File(...),
    nodes_file: UploadFile = File(...),
    arcs_file: UploadFile = File(...),
):
    """Run candidate generation, RF scoring and HMM map matching."""

    if not gps_file.filename or not gps_file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="GPS input must be a CSV file",
        )

    if not nodes_file.filename or not nodes_file.filename.lower().endswith(".nodes"):
        raise HTTPException(
            status_code=400,
            detail="Road network nodes file must have a .nodes extension",
        )

    if not arcs_file.filename or not arcs_file.filename.lower().endswith(".arcs"):
        raise HTTPException(
            status_code=400,
            detail="Road network arcs file must have a .arcs extension",
        )

    gps_content = await gps_file.read()
    nodes_content = await nodes_file.read()
    arcs_content = await arcs_file.read()

    if not gps_content or not nodes_content or not arcs_content:
        raise HTTPException(
            status_code=400,
            detail="One or more uploaded files are empty",
        )

    try:
        gps_points = extract_gps_points(gps_content)

        nodes = parse_network_file(
            nodes_content,
            "Nodes file",
        )

        arcs_data = parse_network_file(
            arcs_content,
            "Arcs file",
        )

        if not np.equal(arcs_data, np.floor(arcs_data)).all():
            raise ValueError("Arcs file must contain integer node indexes")

        arcs = arcs_data.astype(np.int64)

        if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
            raise ValueError(
                "Arcs file contains node indexes outside the nodes file"
            )

        timestamps = []

        for index, point in enumerate(gps_points, start=2):
            try:
                timestamp = float(point["timestamp"])
            except ValueError:
                raise ValueError(
                    f"Timestamp at row {index} must be numeric seconds"
                )

            if not math.isfinite(timestamp):
                raise ValueError(
                    f"Invalid timestamp at row {index}"
                )

            timestamps.append(timestamp)

        gps_df = pd.DataFrame({
            "lon": [point["longitude"] for point in gps_points],
            "lat": [point["latitude"] for point in gps_points],
            "t": timestamps,
        })

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    try:
        from hmm.infer import route_for_inputs

        point_arcs, route = route_for_inputs(
            gps_df,
            nodes,
            arcs,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Map matching failed: {error}",
        )

    point_arcs = [
        int(arc) if arc is not None else None
        for arc in point_arcs
    ]

    route = [int(arc) for arc in route]

    matched_points = sum(
        arc is not None for arc in point_arcs
    )

    route_geometry = []

    for arc_id in route:
        if arc_id < 0 or arc_id >= len(arcs):
            raise HTTPException(
                status_code=500,
                detail="HMM returned an invalid route arc ID",
            )

        start_node, end_node = arcs[arc_id]

        route_geometry.append({
            "arc_id": arc_id,
            "coordinates": [
                [
                    float(nodes[start_node, 0]),
                    float(nodes[start_node, 1]),
                ],
                [
                    float(nodes[end_node, 0]),
                    float(nodes[end_node, 1]),
                ],
            ],
        })

    return {
        "status": "success",
        "message": "GPS map matching completed successfully",
        "gps_points": len(point_arcs),
        "matched_points": matched_points,
        "unmatched_points": len(point_arcs) - matched_points,
        "match_percentage": round(
            matched_points / len(point_arcs) * 100,
            2,
        ),
        "road_nodes": len(nodes),
        "road_arcs": len(arcs),
        "route_arc_count": len(route),
        "point_arcs": point_arcs,
        "route": route,
        "route_geometry": route_geometry,
        "model": "Random Forest + HMM/Viterbi",
    }
