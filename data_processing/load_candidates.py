from pathlib import Path
import pandas as pd

CAND_DIR = Path("results/candidates")
K_DEFAULT, R_DEFAULT = 30, 100.0


def list_trips():
    """Return [(partition, trip), ...] for every generated trip."""
    out = []
    for f in sorted(CAND_DIR.glob("*_cands.csv.gz")):
        partition, trip = f.name.replace("_cands.csv.gz", "").split("_", 1)
        out.append((partition, trip))
    return out


def load_trip(partition, trip, k=K_DEFAULT, r=R_DEFAULT):
    """One row per (GPS point, candidate arc).

    Columns: point_idx, arc_id, dist, rank, label, label_valid, trip,
             x, y, ts   (x, y = projected metres, ts = timestamp)
    """
    c = pd.read_csv(CAND_DIR / f"{partition}_{trip}_cands.csv.gz", dtype={"trip": str})
    p = pd.read_csv(CAND_DIR / f"{partition}_{trip}_points.csv.gz")
    c = c[(c["rank"] < k) & (c["dist"] <= r)]
    # route_pos, true_arc and d_true are label information, so they are left out on purpose
    return c.merge(p[["point_idx", "x", "y", "ts"]], on="point_idx", how="left")


if __name__ == "__main__":
    total = pos = valid = 0
    for part, trip in list_trips():
        d = load_trip(part, trip)
        total += len(d)
        pos += int(d.label.sum())
        valid += int((d.label_valid == 1).sum())
    print(f"trips: {len(list_trips())} | rows: {total} | positives: {pos} | rows with valid labels: {valid}")