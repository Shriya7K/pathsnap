from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

out = Path("results/eda")
out.mkdir(parents=True, exist_ok=True)

dq = pd.read_csv("results/eda/eda_statistics.csv", dtype={"trip_id": str})
cs = pd.read_csv("results/candidate_summary.csv", dtype={"trip": str})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print("saved", out / name)


# 1. points per trip
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.logspace(np.log10(dq.n_points.min()), np.log10(dq.n_points.max()), 20)
for part, g in dq.groupby("partition"):
    ax.hist(g.n_points, bins=bins, alpha=0.6, label=f"{part} (n={len(g)} trips)")
ax.set_xscale("log")
ax.set_xlabel("GPS points per trip (log scale)")
ax.set_ylabel("Number of trips")
ax.set_title("Trip Size Distribution by Partition")
ax.legend()
save(fig, "points_per_trip.png")

# 2. largest time gap per trip (worst 10)
top = dq.nlargest(10, "gap_max")
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(top.partition + "/" + top.trip_id, top.gap_max, color="#d95f02")
ax.set_yscale("log")
ax.set_ylabel("Max timestamp gap (s, log scale)")
ax.set_title("Top 10 Trips with Largest Temporal Gaps")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
save(fig, "largest_gaps.png")

# 3. timestamp gap distribution across all trips (sampling granularity)
fig, ax = plt.subplots(figsize=(7, 4))
# Distribution of % 1-second gaps
ax.hist(dq["pct_gap_1s"], bins=20, color="#1b9e77", edgecolor="black", alpha=0.8)
ax.set_xlabel("Percentage of 1-second gaps per trip (%)")
ax.set_ylabel("Number of trips")
ax.set_title("GPS Sampling Frequency Regularity (Dominance of 1 Hz)")
save(fig, "timestamp_gap_distribution.png")

# 4. geographic extent per trip (metric UTM bounding boxes)
ext = np.maximum(dq.extent_km_ns, dq.extent_km_ew)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ext, bins=25, color="#7570b3", edgecolor="black", alpha=0.7)
ax.set_xlabel("Longest dimension of metric track bounding box (km)")
ax.set_ylabel("Number of trips")
ax.set_title("Geographic Metric Extent per Trip")
save(fig, "extent_km.png")

# 5. network connectivity components
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(dq.n_components, bins=25, color="#386cb0", edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Connected components per road network")
axes[0].set_ylabel("Number of trips")
axes[0].set_title("Component Count")

axes[1].hist(dq.largest_component_pct, bins=25, color="#f0027f", edgecolor="black", alpha=0.7)
axes[1].set_xlabel("Nodes in the largest component (%)")
axes[1].set_ylabel("Number of trips")
axes[1].set_title("Giant Component Share")
fig.suptitle("Road Network Partitioning & Connectivity", y=1.02)
save(fig, "network_components.png")

# 6. comprehensive network statistics (nodes vs arcs)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(dq["n_nodes"], dq["n_arcs"], c=dq["n_components"], cmap="viridis", alpha=0.8, edgecolors="none")
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label("Number of Connected Components")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of Nodes (log scale)")
ax.set_ylabel("Number of Arcs (log scale)")
ax.set_title("Road Network Scale & Graph Complexity")
save(fig, "network_statistics.png")

# 7. sample GPS trajectories and road network projections
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sample_trips = [("train", "00000001"), ("train", "00000033"), ("student_test", "00000000")]

for ax, (part, tr_id) in zip(axes, sample_trips):
    t_file = Path("data") / part / tr_id / f"{tr_id}.track"
    n_file = Path("data") / part / tr_id / f"{tr_id}.nodes"
    a_file = Path("data") / part / tr_id / f"{tr_id}.arcs"
    r_file = Path("data") / part / tr_id / f"{tr_id}.route"

    if t_file.exists() and n_file.exists() and a_file.exists():
        trk = pd.read_csv(t_file, sep="\t", header=None)
        nodes = pd.read_csv(n_file, sep="\t", header=None)
        arcs = pd.read_csv(a_file, sep="\t", header=None)
        route = pd.read_csv(r_file, sep="\t", header=None)[0].to_numpy()

        zone = int((trk[0].mean() + 180) // 6) + 1
        epsg = (32600 if trk[1].mean() >= 0 else 32700) + zone
        tr = Transformer.from_crs(4326, epsg, always_xy=True)

        nx, ny = tr.transform(nodes[0].to_numpy(), nodes[1].to_numpy())
        px, py = tr.transform(trk[0].to_numpy(), trk[1].to_numpy())
        a_from, a_to = arcs[0].to_numpy(), arcs[1].to_numpy()

        # Plot local network surrounding track
        pad = 200.0  # 200m buffer
        xmin, xmax = px.min() - pad, px.max() + pad
        ymin, ymax = py.min() - pad, py.max() + pad

        x1, y1, x2, y2 = nx[a_from], ny[a_from], nx[a_to], ny[a_to]
        mask = (np.maximum(x1, x2) >= xmin) & (np.minimum(x1, x2) <= xmax) & \
               (np.maximum(y1, y2) >= ymin) & (np.minimum(y1, y2) <= ymax)

        idx = np.where(mask)[0][:3000]
        for i in idx:
            ax.plot([x1[i], x2[i]], [y1[i], y2[i]], color="#cccccc", lw=0.6, zorder=1)

        # Plot route
        for i in route:
            ax.plot([nx[a_from[i]], nx[a_to[i]]], [ny[a_from[i]], ny[a_to[i]]], color="#2ca02c", lw=2.0, zorder=2)

        # Plot GPS points
        ax.scatter(px[::3], py[::3], color="#d62728", s=6, alpha=0.7, zorder=3, label="GPS (subsampled)")
        ax.set_title(f"Trip {tr_id} ({part})")
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")
        ax.set_aspect("equal", "datalim")

axes[0].legend(loc="upper left")
fig.suptitle("Sample GPS Trajectories, Road Networks & Reference Routes (Metric UTM)", y=1.03)
save(fig, "gps_trajectories_sample.png")

# 8. share of points without a valid label (investigation of >50m stationarity)
cs["invalid_pct"] = (1 - cs.n_valid / cs.n_points) * 100
w = cs.nlargest(15, "invalid_pct")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(w.partition + "/" + w.trip, w.invalid_pct, color="#e41a1c")
ax.set_ylabel("Points with no valid label (%)")
ax.set_title("Top 15 Trips with Unaligned/Off-Route GPS Observations (>50m)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
save(fig, "invalid_labels.png")

print("\n=== Summary Stats for EDA Report ===")
print("Trips per partition:\n", dq.partition.value_counts().to_string())
print("Trips with a gap > 60 s:", int((dq.n_gap_gt_60s > 0).sum()))
print("Median 1-second steps (%):", dq.pct_gap_1s.median())
print("Median components per network:", dq.n_components.median())
print("Median largest component (%):", dq.largest_component_pct.median())
print("Median route connectivity (%):", dq.route_connected_pct.median())
print("Overall points with no valid label (%):",
      round((1 - cs.n_valid.sum() / cs.n_points.sum()) * 100, 2))
