import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from pyproj import Transformer

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
R_MAX, K_MAX = 150.0, 50

# Configurations to evaluate (K, radius_meters)
GRID = [
    (5, 50),
    (10, 100),
    (20, 100),
    (30, 100),
    (50, 100),
    (50, 150),
]

OUT = Path("results/candidates")
OUT.mkdir(parents=True, exist_ok=True)
Path("results/eda").mkdir(parents=True, exist_ok=True)


def read_tsv(d, s, ext):
    return pd.read_csv(d / (s + ext), sep="\t", header=None)


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def pt_seg_dist(p, A, B):
    """Vectorized perpendicular distance from point p (or array of points) to segments [A, B]."""
    AB = B - A
    L2 = (AB ** 2).sum(1)
    # Avoid division by zero for zero-length segments
    L2_safe = np.where(L2 == 0, 1.0, L2)
    t = ((p - A) * AB).sum(1) / L2_safe
    t = np.clip(t, 0.0, 1.0)
    proj = A + t[:, None] * AB
    return np.hypot(*(p - proj).T)


def align(xy, A, B):
    """Monotonic dynamic programming alignment of GPS points to route arcs.

    Enforces that observations progress forward along the reference route:
    route_index(t) >= route_index(t-1).
    Minimizes cumulative perpendicular distance to the assigned route arcs.
    """
    N, M = len(xy), len(A)
    idx = np.arange(M)
    back = np.empty((N, M), dtype=np.int32)
    back[0] = idx
    dp = pt_seg_dist(xy[0], A, B)
    for i in range(1, N):
        pm = np.minimum.accumulate(dp)
        back[i] = np.maximum.accumulate(np.where(dp == pm, idx, 0))
        dp = pt_seg_dist(xy[i], A, B) + pm
    j = np.empty(N, dtype=np.int32)
    j[-1] = int(np.argmin(dp))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i][j[i]]
    return j


def process_trip(tf, root_dir):
    d, s = tf.parent, tf.stem
    if ONLY and s != ONLY:
        return None
    part = tf.relative_to(root_dir).parts[0]
    t0 = time.time()

    trk = read_tsv(d, s, ".track")
    nodes = read_tsv(d, s, ".nodes")
    arcs = read_tsv(d, s, ".arcs")
    route = read_tsv(d, s, ".route")

    a_from, a_to = arcs[0].to_numpy(), arcs[1].to_numpy()
    rt = route[0].to_numpy()

    # 1. Project GPS track and road network to local metric CRS (UTM)
    mean_lon, mean_lat = float(trk[0].mean()), float(trk[1].mean())
    epsg = utm_epsg(mean_lon, mean_lat)
    tr = Transformer.from_crs(4326, epsg, always_xy=True)

    nx, ny = tr.transform(nodes[0].to_numpy(), nodes[1].to_numpy())
    nx, ny = np.asarray(nx), np.asarray(ny)
    px, py = tr.transform(trk[0].to_numpy(), trk[1].to_numpy())
    px, py = np.asarray(px), np.asarray(py)
    xy = np.column_stack([px, py])
    N = len(xy)

    # 2. Ground-truth label alignment: map GPS points to route arcs
    A = np.column_stack([nx[a_from[rt]], ny[a_from[rt]]])
    B = np.column_stack([nx[a_to[rt]], ny[a_to[rt]]])
    j = align(xy, A, B)
    true_arc = rt[j]

    # Calculate distance to aligned reference arc
    d_true = pt_seg_dist(xy, A[j], B[j])
    # Observations >50m from their aligned route arc are flagged as unalignable/off-route
    valid = d_true <= 50.0
    n_valid = int(valid.sum())
    n_unaligned = int((~valid).sum())

    # 3. Spatial indexing & candidate generation using STRtree
    xmin, ymin = xy.min(0) - R_MAX
    xmax, ymax = xy.max(0) + R_MAX
    x1, y1, x2, y2 = nx[a_from], ny[a_from], nx[a_to], ny[a_to]
    keep = np.where(
        (np.maximum(x1, x2) >= xmin) & (np.minimum(x1, x2) <= xmax) &
        (np.maximum(y1, y2) >= ymin) & (np.minimum(y1, y2) <= ymax)
    )[0]

    if len(keep) == 0:
        c = pd.DataFrame(columns=[
            "trip_id", "point_idx", "arc_id", "label", "perpendicular_distance",
            "distance_rank", "dist", "rank", "label_valid", "x", "y", "ts"
        ])
        pi, li, dist = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    else:
        coords = np.stack([
            np.column_stack([x1[keep], y1[keep]]),
            np.column_stack([x2[keep], y2[keep]])
        ], axis=1)
        lines = shapely.linestrings(coords)
        tree = STRtree(lines)
        pts = shapely.points(xy)
        pi, li = tree.query(pts, predicate="dwithin", distance=R_MAX)
        dist = shapely.distance(pts[pi], lines[li])

        # Stable sort by point_idx, dist, and arc_id to break distance ties deterministically
        c = pd.DataFrame({"point_idx": pi, "arc_id": keep[li], "dist": dist})
        c = c.sort_values(["point_idx", "dist", "arc_id"], kind="stable").reset_index(drop=True)
        c["rank"] = c.groupby("point_idx").cumcount()
        c = c[c["rank"] < K_MAX].copy()

        c["trip_id"] = s
        c["perpendicular_distance"] = c["dist"].round(4)
        c["distance_rank"] = c["rank"]
        c["label"] = (c["arc_id"].to_numpy() == true_arc[c["point_idx"].to_numpy()]).astype(np.int8)
        c["label_valid"] = valid[c["point_idx"].to_numpy()].astype(np.int8)

        # Merge spatial/temporal coordinates
        c["x"] = px[c["point_idx"].to_numpy()]
        c["y"] = py[c["point_idx"].to_numpy()]
        c["ts"] = trk[2].iloc[c["point_idx"].to_numpy()].to_numpy()
        c["trip"] = s  # backwards compatibility

    # 4. Miss rate & evaluation across grid
    hit = c[c["label"] == 1]
    t_rank = np.full(N, 999, dtype=int)
    t_dist = np.full(N, 999.0, dtype=float)
    if len(hit) > 0:
        t_rank[hit["point_idx"].to_numpy()] = hit["distance_rank"].to_numpy()
        t_dist[hit["point_idx"].to_numpy()] = hit["perpendicular_distance"].to_numpy()

    elapsed = time.time() - t0

    # Points with candidates within 100m
    points_with_cands_100m = c[c["dist"] <= 100.0]["point_idx"].nunique() if len(c) else 0
    points_no_cands_100m = N - points_with_cands_100m

    row = {
        "partition": part,
        "trip_id": s,
        "trip": s,
        "epsg": epsg,
        "n_points": N,
        "n_valid": n_valid,
        "n_unaligned": n_unaligned,
        "n_no_cands_100m": points_no_cands_100m,
        "label_dist_gt50": n_unaligned,
        "avg_cands": round(len(c) / max(N, 1), 2),
        "runtime_sec": round(elapsed, 2),
    }

    for K, R in GRID:
        misses = int((((t_rank >= K) | (t_dist > R)) & valid).sum())
        row[f"miss_K{K}_R{R}"] = misses
        miss_rate = misses / max(n_valid, 1)
        row[f"miss_rate_K{K}_R{R}"] = round(miss_rate * 100, 3)
        row[f"recall_K{K}_R{R}"] = round((1.0 - miss_rate) * 100, 3)

    # 5. Save per-trip datasets
    # Person 2 clean schema
    cols_order = [
        "trip_id", "point_idx", "arc_id", "label", "perpendicular_distance",
        "distance_rank", "dist", "rank", "label_valid", "x", "y", "ts"
    ]
    c[cols_order].to_csv(OUT / f"{part}_{s}_cands.csv.gz", index=False, compression="gzip")

    # Points metadata
    points_df = pd.DataFrame({
        "trip_id": s,
        "point_idx": np.arange(N),
        "x": px,
        "y": py,
        "ts": trk[2],
        "route_pos": j,
        "true_arc": true_arc,
        "d_true": d_true.round(4),
        "label_valid": valid.astype(np.int8),
    })
    points_df.to_csv(OUT / f"{part}_{s}_points.csv.gz", index=False, compression="gzip")

    print(f"[{part}/{s}] {N} pts ({n_valid} valid, {n_unaligned} unaligned) | EPSG {epsg} | {len(c)} cands | {elapsed:.2f}s")
    return row


def main():
    files = sorted(ROOT.rglob("*.track"))
    if ONLY:
        files = [f for f in files if f.stem == ONLY]
        if not files:
            print(f"Trip '{ONLY}' not found under {ROOT}")
            return

    summary = []
    print(f"Starting candidate generation for {len(files)} trips (R_MAX={R_MAX}m, K_MAX={K_MAX})...")
    t_start = time.time()

    for tf in files:
        r = process_trip(tf, ROOT)
        summary.append(r)

    total_time = time.time() - t_start
    sm = pd.DataFrame(summary)

    # If running full dataset, write full evaluation tables
    if not ONLY or not Path("results/candidate_summary.csv").exists():
        sm.to_csv("results/candidate_summary.csv", index=False)
        sm.to_csv("results/eda/candidate_miss_rates.csv", index=False)
    else:
        # Merge single trip update safely into existing candidate_summary.csv
        old_sm = pd.read_csv("results/candidate_summary.csv", dtype={"trip_id": str, "trip": str})
        old_sm = old_sm[old_sm["trip_id"] != ONLY]
        combined = pd.concat([old_sm, sm], ignore_index=True).sort_values(["partition", "trip_id"])
        combined.to_csv("results/candidate_summary.csv", index=False)
        combined.to_csv("results/eda/candidate_miss_rates.csv", index=False)
        sm = combined

    # Aggregated miss-rate report
    n_total = sm["n_points"].sum()
    n_valid_total = sm["n_valid"].sum()
    n_unaligned_total = sm["n_unaligned"].sum()

    print("\n" + "=" * 60)
    print("           CANDIDATE GENERATION & EVALUATION REPORT")
    print("=" * 60)
    print(f"Total trips processed: {len(sm)}")
    print(f"Total GPS observations: {n_total:,}")
    print(f"Valid labelled observations (d_true <= 50m): {n_valid_total:,} ({n_valid_total / n_total * 100:.2f}%)")
    print(f"Unalignable observations (d_true > 50m): {n_unaligned_total:,} ({n_unaligned_total / n_total * 100:.2f}%)")
    print(f"Total execution time: {total_time:.1f}s")
    print("-" * 60)
    print("Miss Rate & Recall across Configurations (on valid labels):")
    print(f"{'Configuration':<20} {'Misses':<10} {'Miss Rate':<12} {'Recall':<12}")
    print("-" * 60)

    comparison_rows = []
    for K, R in GRID:
        misses = sm[f"miss_K{K}_R{R}"].sum()
        m_rate = (misses / n_valid_total) * 100
        recall = 100.0 - m_rate
        print(f"K={K:<3} R={R:<4} m:      {misses:<10} {m_rate:>6.3f} %     {recall:>6.3f} %")
        comparison_rows.append({
            "K": K,
            "Radius_m": R,
            "Total_Evaluated": n_valid_total,
            "Misses": misses,
            "Miss_Rate_pct": round(m_rate, 4),
            "Recall_pct": round(recall, 4),
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv("results/eda/candidate_configuration_comparison.csv", index=False)
    print("-" * 60)
    print("Worst 5 trips at K=10, R=100m:")
    sm["miss_pct_K10"] = (sm["miss_K10_R100"] / sm["n_valid"].clip(lower=1)) * 100
    print(sm.nlargest(5, "miss_pct_K10")[["partition", "trip_id", "n_points", "n_valid", "miss_K10_R100", "miss_pct_K10"]])
    print("=" * 60)
    print("Saved:")
    print("  - results/candidate_summary.csv")
    print("  - results/eda/candidate_miss_rates.csv")
    print("  - results/eda/candidate_configuration_comparison.csv")
    print(f"  - results/candidates/ ({len(files)} trips saved as compressed CSVs)")


if __name__ == "__main__":
    main()
