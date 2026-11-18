"""Merge P2's per-trip prediction files into ONE probability file for the HMM pipeline.

Input layout (P2, data_processing/predict_test_probabilities.py):
    <in-dir>/<partition>_<trip>.csv[.gz]   e.g. student_test_00000000.csv.gz
    columns: (trip), point_idx, arc_id, <features...>, probability     (no trip_id column needed)
The same layout is accepted for out-of-fold train files: train_<trip>.csv.gz.

Output: one file with columns  partition, trip_id, point_idx, arc_id, prob, prob_raw
    prob_raw = P2's probability as is (balanced class weights => per-point sums are NOT 1)
    prob     = prob_raw renormalised per GPS point so it sums to 1 (all-zero point -> uniform)

Standalone on purpose: pandas / numpy only, no other hmm module is imported.

    python -m hmm.import_p2 --in-dir results/predictions --partition student_test \
        --out results/p2/student_test_probs.csv.gz --expect-trips 10
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED = ("train", "student_test")                  # evaluator_test is never accepted
NAME_RE = re.compile(r"^(train|student_test)_(\d+)\.csv(\.gz)?$")
NEEDED = ["point_idx", "arc_id", "probability"]


class ImportError_(Exception):
    """Raised for any problem that must stop the import (message is shown to the user)."""


def parse_name(path):
    """('student_test', '00000007') from a file name; refuses evaluator_test and anything else."""
    name = Path(path).name
    if "evaluator" in name.lower():
        raise ImportError_(f"{name}: evaluator_test files are never imported (held-out partition).")
    m = NAME_RE.match(name)
    if not m:
        raise ImportError_(f"{name}: file name must look like student_test_<digits>.csv.gz or train_<digits>.csv.gz")
    return m.group(1), m.group(2).zfill(8)


def read_one(path):
    """Read only point_idx, arc_id, probability of one file and validate it. Returns a DataFrame."""
    name = Path(path).name
    try:
        df = pd.read_csv(path, usecols=lambda c: c in NEEDED)
    except Exception as e:                                          # unreadable / corrupt file
        raise ImportError_(f"{name}: cannot read ({e})")
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise ImportError_(f"{name}: missing column(s) {missing}")
    for c in ("point_idx", "arc_id"):
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any() or (v % 1 != 0).any() or (v < 0).any():
            raise ImportError_(f"{name}: column {c} must hold non-negative integers")
        df[c] = v.astype("int64")
    p = pd.to_numeric(df["probability"], errors="coerce").to_numpy(float)
    if np.isnan(p).any():
        raise ImportError_(f"{name}: {int(np.isnan(p).sum())} NaN / non-numeric probability value(s)")
    if not np.isfinite(p).all():
        raise ImportError_(f"{name}: infinite probability value(s)")
    if (p < 0).any() or (p > 1).any():
        raise ImportError_(f"{name}: probability outside [0,1] (min {p.min():.4g}, max {p.max():.4g})")
    df["probability"] = p
    dup = int(df.duplicated(["point_idx", "arc_id"]).sum())
    if dup:
        raise ImportError_(f"{name}: {dup} duplicate (point_idx, arc_id) row(s)")
    return df


def normalise_points(df):
    """Add prob = prob_raw / per-point sum; all-zero points get uniform. Returns (df, n_zero_points, sums)."""
    raw = df["prob_raw"]
    key = [df["trip_id"], df["point_idx"]]
    s = raw.groupby(key).transform("sum")
    n = raw.groupby(key).transform("size")
    zero = (s <= 0)
    prob = np.where(zero, 1.0 / n, raw / s.where(~zero, 1.0))
    out = df.copy()
    out["prob"] = prob
    point_sums = raw.groupby(key).sum()
    return out, int(zero.groupby(key).first().sum()), point_sums


def import_dir(in_dir, partition, expect_trips=None, verbose=True):
    """Merge every <partition>_<trip> file of in_dir. Returns the merged DataFrame."""
    if partition not in ALLOWED:
        raise ImportError_(f"partition '{partition}' refused; allowed: {', '.join(ALLOWED)} "
                           "(evaluator_test is never imported).")
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise ImportError_(f"input directory not found: {in_dir}")
    files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.name.lower().endswith((".csv", ".csv.gz")))
    if any("evaluator" in p.name.lower() for p in files):
        bad = [p.name for p in files if "evaluator" in p.name.lower()]
        raise ImportError_(f"evaluator_test file(s) found in the input directory: {bad}. Refusing.")
    chosen = []
    for p in files:
        part, trip = parse_name(p)                                  # also validates every name
        if part == partition:
            chosen.append((p, trip))
    if not chosen:
        raise ImportError_(f"no {partition}_<trip>.csv(.gz) files in {in_dir}")
    trips = [t for _, t in chosen]
    if len(set(trips)) != len(trips):
        raise ImportError_("two files map to the same trip id (e.g. .csv and .csv.gz): " + str(sorted(trips)))
    frames = []
    for p, trip in chosen:
        d = read_one(p)
        d.insert(0, "trip_id", trip)
        d.insert(0, "partition", partition)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).rename(columns={"probability": "prob_raw"})
    df, n_zero, point_sums = normalise_points(df)
    df = df[["partition", "trip_id", "point_idx", "arc_id", "prob", "prob_raw"]]
    df = df.sort_values(["trip_id", "point_idx", "arc_id"], kind="stable").reset_index(drop=True)

    msg = [f"partition            : {partition}",
           f"trips               : {len(trips)}  ({', '.join(sorted(trips))})",
           f"rows                : {len(df):,}",
           f"GPS points          : {len(point_sums):,}",
           f"raw per-point sum   : mean {point_sums.mean():.4f}  min {point_sums.min():.4f}  max {point_sums.max():.4f}"
           "   (not 1 if P2 used class weights)",
           f"all-zero points     : {n_zero:,} (given a uniform distribution)"]
    if expect_trips is not None and len(trips) != expect_trips:
        msg.append(f"WARNING: --expect-trips {expect_trips} but found {len(trips)} trips")
    if verbose:
        print("\n".join(msg))
    df.attrs["warnings"] = [m for m in msg if m.startswith("WARNING")]
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="results/predictions")
    ap.add_argument("--partition", required=True, help="student_test or train (out-of-fold files)")
    ap.add_argument("--out", required=True, help="output .csv / .csv.gz / .parquet")
    ap.add_argument("--expect-trips", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        df = import_dir(a.in_dir, a.partition, a.expect_trips)
    except ImportError_ as e:
        print(f"IMPORT ERROR: {e}")
        return 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)                                 # compression inferred from .gz
    print(f"written             : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
