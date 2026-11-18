import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
EXTS = [".track", ".nodes", ".arcs", ".route"]

rows = []
for track_file in sorted(root.rglob("*.track")):
    trip_dir = track_file.parent
    stem = track_file.stem
    rel = track_file.relative_to(root).parts
    partition = rel[0] if len(rel) > 1 else "root"

    row = {"partition": partition, "trip": stem, "folder": str(trip_dir)}
    for ext in EXTS:
        f = trip_dir / (stem + ext)
        row["has" + ext] = f.exists()
        row["size" + ext] = f.stat().st_size if f.exists() else 0
        if f.exists():
            try:
                row["n" + ext] = len(pd.read_csv(f, sep="\t", header=None))
            except Exception:
                row["n" + ext] = -1
        else:
            row["n" + ext] = 0
    rows.append(row)

df = pd.DataFrame(rows)
print(df.groupby("partition").size().rename("trips"))
print()
print(df.groupby("partition")[["n.track", "n.nodes", "n.arcs", "n.route"]].describe().T)
print()
print("Trips missing any file:", int((~df[["has.track", "has.nodes", "has.arcs", "has.route"]].all(axis=1)).sum()))

Path("results").mkdir(exist_ok=True)
df.to_csv("results/inventory.csv", index=False)
print("\nSaved results/inventory.csv")
