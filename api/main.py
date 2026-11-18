
import csv
import io
import math
import uuid
from pathlib import Path

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


@app.get("/")
async def home():

    if not FRONTEND_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend file not found.",
        )

    return FileResponse(FRONTEND_FILE)


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
        "note": "ML and HMM integration is pending.",
    }


@app.get("/health")
async def health():

    return {
        "status": "running",
        "application": "PathSnap",
    }
