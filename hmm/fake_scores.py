"""SYNTHETIC classifier scores for debugging the decoder before P2's model exists.

!!! SYNTHETIC DEBUGGING DATA ONLY. NEVER REPORT ANYTHING PRODUCED FROM THESE AS A RESULT. !!!
The probabilities are built FROM THE TRUE ROUTE (P1's point->arc labels) plus noise, so any
metric computed on them says nothing about a real classifier.

How the fake classifier behaves (per point, over P1's candidate arcs):
  logit = -distance_to_arc / DIST_SCALE + gaussian noise     (nearer arcs look likelier)
  true arc gets a bonus                                       (a "decent" classifier)
  ERROR BURSTS: runs of 3-15 consecutive points in which the nearest WRONG candidate (the
  parallel street) outscores the true arc - the typical failure the HMM should repair.
  probs = softmax(logits) over the point's candidates.

Needs P1's candidate files (python data_processing/candidates.py <data_root>), default dir
results/candidates. Output: results/hmm_synthetic/hmm_synthetic_probs.csv.gz
  python -m hmm.fake_scores --partitions train --limit 5 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import CANDS_DIR, SYNTH_BANNER, SYNTH_DIR
from .graph import DEV_PARTITIONS

DIST_SCALE = 25.0       # metres
NOISE_SD = 0.7
TRUE_BONUS = 1.5
BURST_START_PROB = 0.02  # chance a point starts an error burst
BURST_LEN = (3, 15)


def fake_probs_for_trip(cands, seed):
    """cands: P1 candidate DataFrame (point_idx, arc_id, dist, label, label_valid). Returns
    DataFrame(point_idx, arc_id, prob)."""
    rng = np.random.default_rng(seed)
    c = cands.sort_values(["point_idx", "dist"], kind="stable").reset_index(drop=True)
    logit = -c["dist"].to_numpy() / DIST_SCALE + rng.normal(0, NOISE_SD, len(c))
    is_true = (c["label"].to_numpy() == 1) & (c["label_valid"].to_numpy() == 1)
    logit = logit + TRUE_BONUS * is_true

    pts = c["point_idx"].to_numpy()
    n_pts = pts.max() + 1
    starts = np.searchsorted(pts, np.arange(n_pts + 1))

    # choose burst points
    in_burst = np.zeros(n_pts, bool)
    i = 0
    while i < n_pts:
        if rng.random() < BURST_START_PROB:
            ln = rng.integers(BURST_LEN[0], BURST_LEN[1] + 1)
            in_burst[i:i + ln] = True
            i += ln
        i += 1
    for i in np.nonzero(in_burst)[0]:
        s, e = starts[i], starts[i + 1]
        if e - s < 2:
            continue
        true_idx = np.nonzero(is_true[s:e])[0]
        if len(true_idx) == 0:
            continue
        wrong = [k for k in range(e - s) if not is_true[s + k]]
        k_wrong = wrong[0]                                  # nearest wrong candidate (rows sorted by dist)
        logit[s + k_wrong] = logit[s + true_idx[0]] + 2.0    # parallel street wins

    # softmax within each point
    p = np.empty(len(c))
    for i in range(n_pts):
        s, e = starts[i], starts[i + 1]
        if e > s:
            z = np.exp(logit[s:e] - logit[s:e].max())
            p[s:e] = z / z.sum()
    return pd.DataFrame({"point_idx": pts, "arc_id": c["arc_id"].to_numpy(), "prob": p})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands-dir", default=str(CANDS_DIR))
    ap.add_argument("--partitions", nargs="+", default=["train"])
    ap.add_argument("--limit", type=int, default=5, help="number of trips")
    ap.add_argument("--trips", nargs="*", help="only these trip ids (overrides --limit order)")
    ap.add_argument("--max-points", type=int, default=None, help="skip trips longer than this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(SYNTH_DIR / "hmm_synthetic_probs.csv.gz"))
    a = ap.parse_args()
    if "evaluator_test" in a.partitions or any(p not in DEV_PARTITIONS for p in a.partitions):
        raise SystemExit(f"Only {DEV_PARTITIONS} allowed.")
    print(SYNTH_BANNER)

    parts = []
    for part in a.partitions:
        files = sorted(Path(a.cands_dir).glob(f"{part}_*_cands.csv.gz"))
        done = 0
        for f in files:
            if done >= a.limit:
                break
            if a.trips and int(f.name.split("_")[-2]) not in {int(x) for x in a.trips}:
                continue
            trip = f.name.split("_")[-2]   # file: <partition>_<trip>_cands.csv.gz
            c = pd.read_csv(f)
            if a.max_points and c["point_idx"].max() + 1 > a.max_points:
                continue
            fp = fake_probs_for_trip(c, seed=a.seed * 100003 + int(trip))
            fp.insert(0, "trip_id", str(trip).zfill(8))
            parts.append(fp)
            done += 1
            print(f"  {part}/{trip}: {c['point_idx'].max() + 1} points, {len(fp)} (point,candidate) rows")
    out = pd.concat(parts, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Saved {a.out}  ({len(out)} rows, {out.trip_id.nunique()} trips)  seed={a.seed}")
    print(SYNTH_BANNER)


if __name__ == "__main__":
    main()
