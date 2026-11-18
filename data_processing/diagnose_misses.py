import sys
import pandas as pd

for trip in sys.argv[1:]:
    c = pd.read_csv(f"results/candidates/train_{trip}_cands.csv.gz")
    p = pd.read_csv(f"results/candidates/train_{trip}_points.csv.gz").set_index("point_idx")
    v = p[p.label_valid]
    hit = c[c.label == 1].set_index("point_idx")["rank"]
    rank = hit.reindex(v.index)
    miss = v[rank.isna()]
    n_c = c.groupby("point_idx").size().reindex(miss.index)
    print(f"\n=== {trip}: {len(v)} valid points ===")
    print(f"true arc not in top 20: {rank.isna().mean()*100:.1f}%")
    print("rank of the true arc when found (quantiles 50/90/99):",
          rank.dropna().quantile([.5, .9, .99]).tolist())
    if len(miss):
        print(f"missed points: median d_true {miss.d_true.median():.1f} m, "
              f"median candidates present {n_c.median():.0f}")