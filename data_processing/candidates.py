import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from pyproj import Transformer

ROOT = Path(sys.argv[1])
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
R_MAX, K_MAX = 150.0, 20
GRID = [(5, 30), (5, 50), (10, 50), (10, 100), (20, 100), (20, 150)]
OUT = Path("results/candidates")
OUT.mkdir(parents=True, exist_ok=True)


def read(d, s, ext):
    return pd.read_csv(d / (s + ext), sep="\t", header=None)


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def pt_seg_dist(p, A, B):
    AB = B - A
    L2 = (AB ** 2).sum(1)
    t = ((p - A) * AB).sum(1) / np.where(L2 == 0, 1, L2)
    t = np.clip(t, 0, 1)
    proj = A + t[:, None] * AB
    return np.hypot(*(p - proj).T)


def align(xy, A, B):
    """Monotonic alignment of GPS points to route positions (label creation)."""
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


summary = []
files = sorted(ROOT.rglob("*.track"))
for tf in files:
    d, s = tf.parent, tf.stem
    if ONLY and s != ONLY:
        continue
    part = tf.relative_to(ROOT).parts[0]
    t0 = time.time()

    trk = read(d, s, ".track")
    nodes, arcs, route = read(d, s, ".nodes"), read(d, s, ".arcs"), read(d, s, ".route")
    a_from, a_to = arcs[0].to_numpy(), arcs[1].to_numpy()
    rt = route[0].to_numpy()

    # 1. project to a local metric CRS
    epsg = utm_epsg(trk[0].mean(), trk[1].mean())
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    nx, ny = tr.transform(nodes[0].to_numpy(), nodes[1].to_numpy())
    nx, ny = np.asarray(nx), np.asarray(ny)
    px, py = tr.transform(trk[0].to_numpy(), trk[1].to_numpy())
    xy = np.column_stack([px, py])

    # 2. labels: align each point to a route position
    A = np.column_stack([nx[a_from[rt]], ny[a_from[rt]]])
    B = np.column_stack([nx[a_to[rt]], ny[a_to[rt]]])
    j = align(xy, A, B)
    true_arc = rt[j]
    d_true = pt_seg_dist(xy, A[j], B[j])

    # 3. spatial index on arcs near the track, then candidates
    xmin, ymin = xy.min(0) - R_MAX
    xmax, ymax = xy.max(0) + R_MAX
    x1, y1, x2, y2 = nx[a_from], ny[a_from], nx[a_to], ny[a_to]
    keep = np.where((np.maximum(x1, x2) >= xmin) & (np.minimum(x1, x2) <= xmax) &
                    (np.maximum(y1, y2) >= ymin) & (np.minimum(y1, y2) <= ymax))[0]
    coords = np.stack([np.column_stack([x1[keep], y1[keep]]),
                       np.column_stack([x2[keep], y2[keep]])], axis=1)
    lines = shapely.linestrings(coords)
    tree = STRtree(lines)
    pts = shapely.points(xy)
    pi, li = tree.query(pts, predicate="dwithin", distance=R_MAX)
    dist = shapely.distance(pts[pi], lines[li])
    c = pd.DataFrame({"point_idx": pi, "arc_id": keep[li], "dist": dist})
    c = c.sort_values(["point_idx", "dist"], kind="stable")
    c["rank"] = c.groupby("point_idx").cumcount()
    c = c[c["rank"] < K_MAX].copy()
    c["label"] = (c["arc_id"].to_numpy() == true_arc[c["point_idx"].to_numpy()]).astype(int)

    # 4. miss rates
    hit = c[c.label == 1]
    t_rank = np.full(len(xy), 999)
    t_dist = np.full(len(xy), 999.0)
    t_rank[hit.point_idx.to_numpy()] = hit["rank"].to_numpy()
    t_dist[hit.point_idx.to_numpy()] = hit["dist"].to_numpy()
    row = {"partition": part, "trip": s, "epsg": epsg, "n_points": len(xy),
           "label_dist_gt50": int((d_true > 50).sum()),
           "avg_cands": round(len(c) / len(xy), 2)}
    for K, R in GRID:
        row[f"miss_K{K}_R{R}"] = int(((t_rank >= K) | (t_dist > R)).sum())
    summary.append(row)

    # 5. save
    c.assign(trip=s).to_csv(OUT / f"{part}_{s}_cands.csv.gz", index=False)
    pd.DataFrame({"point_idx": np.arange(len(xy)), "x": px, "y": py, "ts": trk[2],
                  "route_pos": j, "true_arc": true_arc, "d_true": d_true}
                 ).to_csv(OUT / f"{part}_{s}_points.csv.gz", index=False)
    print(f"{part}/{s}: {len(xy)} pts, EPSG {epsg}, {time.time() - t0:.1f}s")

sm = pd.DataFrame(summary)
sm.to_csv("results/candidate_summary.csv", index=False)
n = sm.n_points.sum()
print("\nTotal points:", n, "| label points >50 m from their arc:",
      round(sm.label_dist_gt50.sum() / n * 100, 2), "%")
print("\nCandidate miss rate (% of points whose true arc is not a candidate):")
for K, R in GRID:
    print(f"  K={K:>2}, R={R:>3} m: {sm[f'miss_K{K}_R{R}'].sum() / n * 100:.2f} %")
print("\nWorst trips at K=10, R=100:")
sm["miss_pct"] = sm["miss_K10_R100"] / sm.n_points * 100
print(sm.nlargest(5, "miss_pct")[["partition", "trip", "n_points", "miss_pct"]])