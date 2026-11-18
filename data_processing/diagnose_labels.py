import sys
import numpy as np
import pandas as pd

part = "train"
for trip in sys.argv[1:]:
    p = pd.read_csv(f"results/candidates/{part}_{trip}_points.csv.gz")
    bad = (p.d_true > 50).to_numpy()
    print(f"\n=== {trip}: {len(p)} points ===")
    print(f"label dist >50 m: {bad.mean()*100:.1f}% | >100 m: {(p.d_true > 100).mean()*100:.1f}%")
    print(f"median d_true: {p.d_true.median():.1f} m | max: {p.d_true.max():.0f} m")
    print(f"route_pos range: {p.route_pos.min()}..{p.route_pos.max()}, "
          f"distinct positions used: {p.route_pos.nunique()}")

    # contiguous runs of bad labels
    runs, start = [], None
    for i, b in enumerate(bad):
        if b and start is None:
            start = i
        if not b and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(bad) - 1))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    print("longest bad runs (start_idx, end_idx, length, route_pos there):")
    for a, b in runs[:5]:
        print(f"  {a}..{b}  len={b-a+1}  route_pos={p.route_pos[a]}..{p.route_pos[b]}")
    print("bad share in first 10% / middle 80% / last 10% of the track:",
          f"{bad[:len(bad)//10].mean()*100:.0f}% /",
          f"{bad[len(bad)//10:-len(bad)//10].mean()*100:.0f}% /",
          f"{bad[-len(bad)//10:].mean()*100:.0f}%")