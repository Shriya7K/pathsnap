import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
EXTS = [".track", ".nodes", ".arcs", ".route"]

rows = []
for track_file in sorted(root.rglob("*.track")):
    trip_dir = track_file.parent
    stem = track_file.stem
    rel = track_file.relative_to(root).parts
    partition = rel[0] if len(rel) > 1 else "root"

    row = {
        "partition": partition,
        "trip_id": stem,
        "folder": str(trip_dir),
    }
    total_size = 0
    all_present = True
    for ext in EXTS:
        f = trip_dir / (stem + ext)
        exists = f.exists()
        size = f.stat().st_size if exists else 0
        total_size += size
        row["has" + ext] = exists
        row["size" + ext + "_bytes"] = size
        if not exists:
            all_present = False
            row["n" + ext] = 0
        else:
            try:
                row["n" + ext] = len(pd.read_csv(f, sep="\t", header=None))
            except Exception:
                row["n" + ext] = -1

    row["total_size_bytes"] = total_size
    row["all_files_present"] = all_present
    rows.append(row)

df = pd.DataFrame(rows)

# Rename columns for clarity while preserving original column aliases
df["n_points"] = df["n.track"]
df["n_nodes"] = df["n.nodes"]
df["n_arcs"] = df["n.arcs"]
df["n_route"] = df["n.route"]

print("=== Dataset Inventory Summary ===")
print("Trips per partition:\n", df.groupby("partition").size())
print("\nRow count statistics per partition:")
print(df.groupby("partition")[["n_points", "n_nodes", "n_arcs", "n_route"]].describe().T)
print("\nTotal dataset files:", len(df) * 4)
print("Total dataset size: {:.2f} MB".format(df["total_size_bytes"].sum() / (1024 * 1024)))
print("Trips missing any file:", int((~df["all_files_present"]).sum()))

Path("results/eda").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(parents=True, exist_ok=True)

# Save both in results/eda/ and results/ for full compatibility
df.to_csv("results/eda/dataset_inventory.csv", index=False)
df.to_csv("results/inventory.csv", index=False)
print("\nSaved results/eda/dataset_inventory.csv and results/inventory.csv")
