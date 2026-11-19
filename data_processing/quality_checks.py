import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")


def read_tsv(path):
    return pd.read_csv(path, sep="\t", header=None)


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


rows = []
for track_file in sorted(root.rglob("*.track")):
    d = track_file.parent
    s = track_file.stem
    partition = track_file.relative_to(root).parts[0]

    trk = read_tsv(track_file)
    nodes = read_tsv(d / (s + ".nodes"))
    arcs = read_tsv(d / (s + ".arcs"))
    route = read_tsv(d / (s + ".route"))

    r = {"partition": partition, "trip_id": s}

    # ---- 1. GPS Quality & Coordinates Plausibility ----
    lon_raw = pd.to_numeric(trk[0], errors="coerce")
    lat_raw = pd.to_numeric(trk[1], errors="coerce")
    n_points = len(trk)
    r["n_points"] = n_points
    r["nan_coords"] = int(lon_raw.isna().sum() + lat_raw.isna().sum())
    r["lon_out_of_range"] = int((lon_raw.abs() > 180).sum())
    r["lat_out_of_range"] = int((lat_raw.abs() > 90).sum())
    r["dup_points"] = int(trk.duplicated(subset=[0, 1]).sum())
    r["lon_min"], r["lon_max"] = float(lon_raw.min()), float(lon_raw.max())
    r["lat_min"], r["lat_max"] = float(lat_raw.min()), float(lat_raw.max())

    # Check lon/lat ordering plausibility (standard GPS format: col 0 = lon, col 1 = lat)
    # Valid ranges: lon between -180 and 180, lat between -90 and 90
    r["coord_order_valid"] = (r["lon_out_of_range"] == 0) and (r["lat_out_of_range"] == 0)

    # ---- 2. Geographic Analysis (Local Metric Projection - UTM) ----
    mean_lon = float(lon_raw.mean())
    mean_lat = float(lat_raw.mean())
    epsg = utm_epsg(mean_lon, mean_lat)
    r["epsg"] = epsg
    r["utm_zone"] = int((mean_lon + 180) // 6) + 1
    r["utm_hemisphere"] = "North" if mean_lat >= 0 else "South"

    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    px, py = tr.transform(lon_raw.to_numpy(), lat_raw.to_numpy())
    px, py = np.asarray(px), np.asarray(py)

    x_extent_m = float(px.max() - px.min())
    y_extent_m = float(py.max() - py.min())
    r["extent_km_ew"] = round(x_extent_m / 1000.0, 3)
    r["extent_km_ns"] = round(y_extent_m / 1000.0, 3)
    r["extent_bbox_area_km2"] = round((x_extent_m * y_extent_m) / 1e6, 3)

    # ---- 3. Timestamp Analysis ----
    ts_raw = pd.to_numeric(trk[2], errors="coerce")
    if ts_raw.isna().all():
        ts = pd.to_datetime(trk[2], errors="coerce").astype("int64") / 1e9
    else:
        ts = ts_raw
    dt = ts.diff().dropna()
    r["ts_nan"] = int(ts.isna().sum())
    r["ts_decreasing"] = int((dt < 0).sum())
    r["ts_zero_gap"] = int((dt == 0).sum())
    r["gap_min"] = float(dt.min()) if len(dt) else np.nan
    r["gap_median"] = float(dt.median()) if len(dt) else np.nan
    r["gap_mean"] = round(float(dt.mean()), 2) if len(dt) else np.nan
    r["gap_max"] = float(dt.max()) if len(dt) else np.nan
    r["gap_std"] = round(float(dt.std()), 2) if len(dt) else np.nan
    r["pct_gap_1s"] = round(float((dt == 1).mean() * 100), 2) if len(dt) else np.nan
    r["n_gap_gt_10s"] = int((dt > 10).sum())
    r["n_gap_gt_60s"] = int((dt > 60).sum())

    # Point-to-point speed plausibility (m/s)
    dist_consec = np.hypot(np.diff(px), np.diff(py))
    dt_arr = dt.to_numpy()
    valid_dt = dt_arr > 0
    speeds = np.zeros_like(dist_consec)
    speeds[valid_dt] = dist_consec[valid_dt] / dt_arr[valid_dt]
    r["max_speed_mps"] = round(float(speeds.max()), 1) if len(speeds) else 0.0
    r["median_speed_mps"] = round(float(np.median(speeds)), 1) if len(speeds) else 0.0

    # ---- 4. Road-Network Analysis ----
    n_nodes = len(nodes)
    n_arcs = len(arcs)
    a_from = arcs[0].to_numpy()
    a_to = arcs[1].to_numpy()
    r["n_nodes"] = n_nodes
    r["n_arcs"] = n_arcs
    r["arc_node_min"] = int(min(a_from.min(), a_to.min()))
    r["arc_node_max"] = int(max(a_from.max(), a_to.max()))
    base = 1 if r["arc_node_min"] >= 1 and r["arc_node_max"] == n_nodes else 0
    r["node_id_base_guess"] = base

    used_nodes = np.unique(np.concatenate([a_from, a_to]))
    r["isolated_nodes"] = int(n_nodes - len(used_nodes))
    f = a_from - base
    t = a_to - base
    ok_nodes = (f >= 0) & (f < n_nodes) & (t >= 0) & (t < n_nodes)
    r["arcs_bad_node_ref"] = int((~ok_nodes).sum())
    r["self_loops"] = int((a_from == a_to).sum())

    # Connected components
    g = coo_matrix((np.ones(ok_nodes.sum()), (f[ok_nodes], t[ok_nodes])), shape=(n_nodes, n_nodes))
    ncomp, labels = connected_components(g, directed=False)
    r["n_components"] = int(ncomp)
    r["largest_component_pct"] = round(float(np.bincount(labels).max() / n_nodes * 100), 2)

    # ---- 5. Route Analysis ----
    rt = route[0].to_numpy()
    r["n_route"] = len(rt)
    r["route_min_id"] = int(rt.min())
    r["route_max_id"] = int(rt.max())
    r["route_invalid_arcs"] = int(((rt < 0) | (rt >= n_arcs)).sum())
    r["route_dups"] = int(len(rt) - len(np.unique(rt)))

    # Consecutive route connectivity
    best_base, best_conn = 0, 0.0
    for rbase in (0, 1):
        idx = rt - rbase
        if idx.min() < 0 or idx.max() >= n_arcs:
            continue
        conn = (a_to[idx[:-1]] == a_from[idx[1:]]).mean() * 100
        if conn > best_conn:
            best_base = rbase
            best_conn = conn
    r["arc_id_base"] = best_base
    r["route_connected_pct"] = round(float(best_conn), 2) if len(rt) > 1 else 100.0

    rows.append(r)

df = pd.DataFrame(rows)
df["trip"] = df["trip_id"]  # keep backward compatibility

Path("results/eda").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(parents=True, exist_ok=True)

df.to_csv("results/eda/eda_statistics.csv", index=False)
df.to_csv("results/data_quality.csv", index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("=== 1. GPS Validity (Total across trips) ===")
print(df[["nan_coords", "lon_out_of_range", "lat_out_of_range", "dup_points"]].sum())
print("\n=== 2. Timestamp Quality ===")
print("Trips with decreasing timestamps:", int((df.ts_decreasing > 0).sum()))
print("Trips with zero gaps (duplicate timestamps):", int((df.ts_zero_gap > 0).sum()))
print("Trips with gaps > 60s:", int((df.n_gap_gt_60s > 0).sum()))
print("\nTop 5 trips with the largest time gap:")
print(df.nlargest(5, "gap_max")[["partition", "trip_id", "n_points", "gap_max", "n_gap_gt_60s"]])

print("\n=== 3. Geographic Extent (Metric UTM km) ===")
print(df[["extent_km_ew", "extent_km_ns", "extent_bbox_area_km2"]].describe().round(2))

print("\n=== 4. Road Network Structure ===")
print(df[["isolated_nodes", "self_loops", "n_components", "largest_component_pct", "arcs_bad_node_ref"]].describe().round(2))

print("\n=== 5. Route Structure ===")
print(df[["n_route", "route_invalid_arcs", "route_dups", "route_connected_pct"]].describe().round(2))

print("\nSaved results/eda/eda_statistics.csv and results/data_quality.csv")
