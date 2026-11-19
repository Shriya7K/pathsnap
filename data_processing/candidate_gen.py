"""Label-free candidate generation.

Needs only <trip>.track, <trip>.nodes and <trip>.arcs (no .route), so it works
on evaluator trips. Used by the API and by the feature pipeline.
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from pyproj import Transformer

K_DEFAULT, R_DEFAULT = 30, 100.0


@dataclass
class TripData:
    trip_id: str
    trip: str             # alias for trip_id
    epsg: int
    xy: np.ndarray        # (N, 2) GPS points in projected metres
    ts: np.ndarray        # (N,) raw timestamps
    nx: np.ndarray        # node x, projected metres (whole network)
    ny: np.ndarray        # node y
    a_from: np.ndarray    # arc start node index (arc_id = row index)
    a_to: np.ndarray      # arc end node index
    cands: pd.DataFrame   # candidate rows


def _read_tsv(path):
    return pd.read_csv(path, sep="\t", header=None)


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def generate(trip_dir, k=K_DEFAULT, r=R_DEFAULT):
    trip_dir = Path(trip_dir)
    track_file = next(trip_dir.glob("*.track"))
    stem = track_file.stem

    trk = _read_tsv(track_file)
    nodes = _read_tsv(trip_dir / (stem + ".nodes"))
    arcs = _read_tsv(trip_dir / (stem + ".arcs"))
    a_from, a_to = arcs[0].to_numpy(), arcs[1].to_numpy()

    # Project to local metric CRS (UTM zone chosen per trip)
    mean_lon, mean_lat = float(trk[0].mean()), float(trk[1].mean())
    epsg = utm_epsg(mean_lon, mean_lat)
    tr = Transformer.from_crs(4326, epsg, always_xy=True)

    nx, ny = tr.transform(nodes[0].to_numpy(), nodes[1].to_numpy())
    nx, ny = np.asarray(nx), np.asarray(ny)
    px, py = tr.transform(trk[0].to_numpy(), trk[1].to_numpy())
    px, py = np.asarray(px), np.asarray(py)
    xy = np.column_stack([px, py])

    # Pre-filter candidate bounding box
    xmin, ymin = xy.min(0) - r
    xmax, ymax = xy.max(0) + r
    x1, y1, x2, y2 = nx[a_from], ny[a_from], nx[a_to], ny[a_to]
    keep = np.where(
        (np.maximum(x1, x2) >= xmin) & (np.minimum(x1, x2) <= xmax) &
        (np.maximum(y1, y2) >= ymin) & (np.minimum(y1, y2) <= ymax)
    )[0]

    cols = [
        "trip_id", "point_idx", "arc_id", "perpendicular_distance",
        "distance_rank", "dist", "rank", "x", "y", "ts", "trip"
    ]
    if len(keep) == 0:
        c = pd.DataFrame(columns=cols)
    else:
        coords = np.stack([
            np.column_stack([x1[keep], y1[keep]]),
            np.column_stack([x2[keep], y2[keep]])
        ], axis=1)
        lines = shapely.linestrings(coords)
        tree = STRtree(lines)
        pts = shapely.points(xy)
        pi, li = tree.query(pts, predicate="dwithin", distance=r)
        dist = shapely.distance(pts[pi], lines[li])

        # Stable sort breaking ties deterministically
        c = pd.DataFrame({"point_idx": pi, "arc_id": keep[li], "dist": dist})
        c = c.sort_values(["point_idx", "dist", "arc_id"], kind="stable").reset_index(drop=True)
        c["rank"] = c.groupby("point_idx").cumcount()
        c = c[c["rank"] < k].reset_index(drop=True)

        c["trip_id"] = stem
        c["trip"] = stem
        c["perpendicular_distance"] = c["dist"].round(4)
        c["distance_rank"] = c["rank"]
        c["x"] = px[c["point_idx"].to_numpy()]
        c["y"] = py[c["point_idx"].to_numpy()]
        c["ts"] = trk[2].iloc[c["point_idx"].to_numpy()].to_numpy()

    return TripData(
        trip_id=stem, trip=stem, epsg=epsg, xy=xy, ts=trk[2].to_numpy(),
        nx=nx, ny=ny, a_from=a_from, a_to=a_to, cands=c
    )


if __name__ == "__main__":
    t0 = time.time()
    trip_path = sys.argv[1] if len(sys.argv) > 1 else "data/train/00000001"
    td = generate(trip_path)
    n = len(td.xy)
    with_cand = td.cands.point_idx.nunique() if len(td.cands) else 0
    print(f"Trip {td.trip_id}: {n} points, EPSG {td.epsg}, {len(td.cands)} candidate rows, "
          f"{n - with_cand} points with no candidate, {time.time() - t0:.2f}s")
