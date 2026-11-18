from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("results/eda")
out.mkdir(parents=True, exist_ok=True)

dq = pd.read_csv("results/data_quality.csv", dtype={"trip": str})
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
ax.set_title("Trip size by partition")
ax.legend()
save(fig, "points_per_trip.png")

# 2. largest time gap per trip (worst 10)
top = dq.nlargest(10, "gap_max")
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(top.partition + "/" + top.trip, top.gap_max)
ax.set_yscale("log")
ax.set_ylabel("Largest timestamp gap (s, log scale)")
ax.set_title("10 trips with the largest time gap")
plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
save(fig, "largest_gaps.png")

# 3. geographic extent per trip
ext = np.maximum(dq.extent_km_ns, dq.extent_km_ew)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ext, bins=30)
ax.set_xlabel("Longest side of the track's bounding box (km)")
ax.set_ylabel("Number of trips")
ax.set_title("Geographic extent per trip")
save(fig, "extent_km.png")

# 4. network connectivity
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(dq.n_components, bins=30)
axes[0].set_xlabel("Connected components per trip network")
axes[0].set_ylabel("Number of trips")
axes[1].hist(dq.largest_component_pct, bins=30)
axes[1].set_xlabel("Nodes in the largest component (%)")
fig.suptitle("Road network connectivity")
save(fig, "network_components.png")

# 5. share of points without a valid label
cs["invalid_pct"] = (1 - cs.n_valid / cs.n_points) * 100
w = cs.nlargest(15, "invalid_pct")
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(w.partition + "/" + w.trip, w.invalid_pct)
ax.set_ylabel("Points with no valid label (%)")
ax.set_title("15 trips with the most off-route points")
plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
save(fig, "invalid_labels.png")

# summary numbers for the report text
print("\nTrips per partition:\n", dq.partition.value_counts().to_string())
print("\nTrips with a gap > 60 s:", int((dq.n_gap_gt_60s > 0).sum()))
print("Median 1-second steps (%):", dq.pct_gap_1s.median())
print("Median components per network:", dq.n_components.median())
print("Median largest component (%):", dq.largest_component_pct.median())
print("Median route connectivity (%):", dq.route_connected_pct.median())
print("Points with no valid label (%):",
      round((1 - cs.n_valid.sum() / cs.n_points.sum()) * 100, 2))