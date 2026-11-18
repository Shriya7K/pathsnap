"""DISTANCE-ONLY BASELINE scores (a real, non-ML score source; NOT synthetic).

Stand-in for P2's classifier so the whole real-data path can run now. Per GPS point:

    prob_k = softmax_k( -dist_k / sigma )        k = the point's candidate arcs

dist = metres from the GPS point to the candidate arc (from P1's candidates). The nearest
arc gets the highest probability; sigma (metres) sets how sharply. No ML, no labels.

Input columns used: point_idx, arc_id, dist - NOTHING ELSE (label, label_valid, route_pos,
true_arc, d_true are never read for scoring).

Swap: when P2's model exists, give hmm.evaluate a probability file with the same columns
(trip_id, point_idx, arc_id, prob) - nothing else changes.

  python -m hmm.baseline_scores --partitions train student_test --sigma 20
  -> results/hmm_baseline/baseline_probs.csv.gz
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import DEV_PARTITIONS, REPO
from .evaluate import list_trips

BASELINE_DIR = REPO / "results" / "hmm_baseline"
LABEL = "distance-only baseline"
DEFAULT_SIGMA_M = 20.0
SCORE_COLUMNS = ["point_idx", "arc_id", "dist"]          # the only columns scoring may read


def distance_softmax(cands, sigma=DEFAULT_SIGMA_M):
    """cands: DataFrame with point_idx, arc_id, dist. Returns DataFrame(point_idx, arc_id, prob)
    where prob sums to 1 over each point's candidates."""
    c = cands[SCORE_COLUMNS].copy()                      # <- select the 3 allowed columns, ignore the rest
    c["point_idx"] = c["point_idx"].astype(np.int64)
    c["arc_id"] = c["arc_id"].astype(np.int64)
    # numerically stable softmax: subtract each point's smallest distance before exp()
    dmin = c.groupby("point_idx")["dist"].transform("min")
    w = np.exp(-(c["dist"].to_numpy(dtype=float) - dmin.to_numpy(dtype=float)) / sigma)
    c["prob"] = w / pd.Series(w, index=c.index).groupby(c["point_idx"]).transform("sum").to_numpy()
    return c[["point_idx", "arc_id", "prob"]].reset_index(drop=True)


def load_candidates_label_free(partition, trip):
    """P1's candidates for one trip via data_processing/load_candidates.py, reduced at once to
    the three scoring columns (their loader also returns label columns; we drop them here)."""
    import sys
    sys.path.insert(0, str(REPO / "data_processing"))
    import load_candidates as lc                          # P1's loader (not edited)
    lc.CAND_DIR = REPO / "results" / "candidates"         # make the path independent of the working dir
    return lc.load_trip(partition, trip)[SCORE_COLUMNS]


def main():
    ap = argparse.ArgumentParser(description=f"{LABEL}: softmax(-dist/sigma) per GPS point")
    ap.add_argument("--partitions", nargs="+", default=list(DEV_PARTITIONS))
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA_M, help="metres (default 20)")
    ap.add_argument("--out", default=str(BASELINE_DIR / "baseline_probs.csv.gz"))
    a = ap.parse_args()
    if any(p not in DEV_PARTITIONS for p in a.partitions):
        raise SystemExit(f"Only {DEV_PARTITIONS} are allowed (evaluator_test is never used in development).")

    t0, parts = time.time(), []
    for part, trip in list_trips(a.partitions):
        p = distance_softmax(load_candidates_label_free(part, trip), a.sigma)
        p.insert(0, "trip_id", trip)
        parts.append(p)
        print(f"  {part}/{trip}: {p['point_idx'].nunique()} points, {len(p)} (point, candidate) rows")
    out = pd.concat(parts, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\n[{LABEL}] sigma={a.sigma} m | {out.trip_id.nunique()} trips | {len(out)} rows | "
          f"{time.time() - t0:.1f}s\nSaved {a.out}")


if __name__ == "__main__":
    main()
