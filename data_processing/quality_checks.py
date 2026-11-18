import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

root = Path(sys.argv[1])


def read(path):
    return pd.read_csv(path, sep="\t", header=None)


rows = []
for track_file in sorted(root.rglob("*.track")):
    d = track_file.parent
    s = track_file.stem
    partition = track_file.relative_to(root).parts[0]

    trk = read(track_file)
    nodes = read(d / (s + ".nodes"))
    arcs = read(d / (s + ".arcs"))
    route = read(d / (s + ".route"))

    r = {"partition": partition, "trip": s}

    # ---- GPS validity ----
    lon = pd.to_numeric(trk[0], errors="coerce")
    lat = pd.to_numeric(trk[1], errors="coerce")
    r["n_points"] = len(trk)
    r["nan_coords"] = int(lon.isna().sum() + lat.isna().sum())
    r["lon_out_of_range"] = int((lon.abs() > 180).sum())
    r["lat_out_of_range"] = int((lat.abs() > 90).sum())
    r["dup_points"] = int(trk.duplicated(subset=[0, 1]).sum())
    r["lon_min"], r["lon_max"] = lon.min(), lon.max()
    r["lat_min"], r["lat_max"] = lat.min(), lat.max()
    # rough extent in km
    r["extent_km_ns"] = (r["lat_max"] - r["lat_min"]) * 111.0
    r["extent_km_ew"] = (r["lon_max"] - r["lon_min"]) * 111.0 * np.cos(np.radians(lat.mean()))

    # ---- timestamps ----
    ts = pd.to_numeric(trk[2], errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(trk[2], errors="coerce").astype("int64") / 1e9
    dt = ts.diff().dropna()
    r["ts_nan"] = int(ts.isna().sum())
    r["ts_decreasing"] = int((dt < 0).sum())
    r["ts_zero_gap"] = int((dt == 0).sum())
    r["pct_gap_1s"] = round(float((dt == 1).mean() * 100), 1) if len(dt) else np.nan
    r["gap_median"] = float(dt.median()) if len(dt) else np.nan
    r["gap_max"] = float(dt.max()) if len(dt) else np.nan
    r["n_gap_gt_10s"] = int((dt > 10).sum())
    r["n_gap_gt_60s"] = int((dt > 60).sum())

    # ---- network ----
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

    used = np.unique(np.concatenate([a_from, a_to]))
    r["isolated_nodes"] = int(n_nodes - len(used))
    f = a_from - base
    t = a_to - base
    ok = (f >= 0) & (f < n_nodes) & (t >= 0) & (t < n_nodes)
    r["arcs_bad_node_ref"] = int((~ok).sum())
    g = coo_matrix((np.ones(ok.sum()), (f[ok], t[ok])), shape=(n_nodes, n_nodes))
    ncomp, labels = connected_components(g, directed=False)
    r["n_components"] = int(ncomp)
    r["largest_component_pct"] = round(float(np.bincount(labels).max() / n_nodes * 100), 1)

    # ---- route structure ----
    rt = route[0].to_numpy()
    r["n_route"] = len(rt)
    r["route_min"], r["route_max"] = int(rt.min()), int(rt.max())
    r["route_dups"] = int(len(rt) - len(np.unique(rt)))
    best = None
    for rbase in (0, 1):
        idx = rt - rbase
        if idx.min() < 0 or idx.max() >= n_arcs:
            continue
        # arc i ends where arc i+1 starts
        conn = (a_to[idx[:-1]] == a_from[idx[1:]]).mean() * 100
        if best is None or conn > best[1]:
            best = (rbase, conn)
    r["arc_id_base_guess"] = best[0] if best else -1
    r["route_connected_pct"] = round(float(best[1]), 1) if best else np.nan

    rows.append(r)

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/data_quality.csv", index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("=== GPS validity (total across trips) ===")
print(df[["nan_coords", "lon_out_of_range", "lat_out_of_range", "dup_points"]].sum())
print("\n=== Timestamp problems ===")
print("trips with decreasing timestamps:", int((df.ts_decreasing > 0).sum()))
print("trips with zero gaps:", int((df.ts_zero_gap > 0).sum()))
print("\n5 trips with the largest gap:")
print(df.nlargest(5, "gap_max")[["partition", "trip", "n_points", "gap_max", "n_gap_gt_60s"]])
print("\n5 trips with the lowest % of 1-second steps:")
print(df.nsmallest(5, "pct_gap_1s")[["partition", "trip", "pct_gap_1s", "gap_median"]])
print("\n=== Extent (km) ===")
print(df[["extent_km_ns", "extent_km_ew"]].describe().round(1))
print("lon range:", df.lon_min.min(), "to", df.lon_max.max())
print("\n=== Network ===")
print(df[["isolated_nodes", "n_components", "largest_component_pct", "arcs_bad_node_ref"]].describe().round(1))
print("node id base guess counts:\n", df.node_id_base_guess.value_counts())
print("\n=== Route ===")
print("arc id base guess counts:\n", df.arc_id_base_guess.value_counts())
print(df[["route_dups", "route_connected_pct"]].describe().round(1))
print("\nSaved results/data_quality.csv")