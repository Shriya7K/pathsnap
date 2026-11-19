"""Default decoder config vs tuned decoder config on trips the tuner NEVER saw.

  * student_test trips (10)        - never used in tuning
  * the train OOF trips that are NOT in the tuning subset (hmm_rf_tuning_trips.csv)
Same probabilities for both configs; only the decoder parameters differ. Reference routes are used for scoring only.

    python -m hmm.compare_configs --probs results/p2/probs_all.csv.gz --default-config results/hmm_default_config.json \
        --tuned-config hmm/config.json --tuning-trips results/hmm_rf_tuning_trips.csv --out results/hmm_rf_tuned_vs_default.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

from .evaluate import evaluate_many
from .graph import DATA_ROOT
from .probs_io import CANDS_DIR, load_probs

METRICS = ["point_acc", "precision", "recall", "f1", "lcs_sim", "continuity"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--default-config", required=True)
    ap.add_argument("--tuned-config", required=True)
    ap.add_argument("--tuning-trips", required=True, help="CSV with the trips the tuner used (excluded here)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"], choices=["train", "student_test"])
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--cands-dir", default=str(CANDS_DIR))
    a = ap.parse_args(argv)

    used = set(pd.read_csv(a.tuning_trips, dtype={"trip_id": str}).trip_id)
    lp = load_probs(a.probs, a.partitions, a.data_root, a.cands_dir)
    trips = [t for t in lp.trips if t[1] not in used]
    print(f"{len(used)} tuning trips excluded; evaluating {len(trips)} unseen trips "
          f"({sum(p == 'student_test' for p, _ in trips)} student_test + {sum(p == 'train' for p, _ in trips)} train OOF)")
    cfgs = {"default": json.load(open(a.default_config)), "tuned": json.load(open(a.tuned_config))}
    frames = {}
    for name, cfg in cfgs.items():
        print(f"--- {name} config: " + ", ".join(f"{k}={cfg[k]}" for k in
              ["beta", "switch_penalty", "unreach_penalty", "max_path_m", "temperature", "gap_scale_mps"]))
        frames[name] = evaluate_many(lp.by_trip, trips, cfg, a.data_root, a.cands_dir, verbose=False)
    d, t = frames["default"], frames["tuned"]
    out = d[["partition", "trip_id", "n_points"]].copy()
    for m in METRICS:
        out[f"default_{m}"], out[f"tuned_{m}"] = d[f"hmm_{m}"], t[f"hmm_{m}"]
        out[f"delta_{m}"] = out[f"tuned_{m}"] - out[f"default_{m}"]
    out["default_breaks"], out["tuned_breaks"] = d["hmm_breaks"], t["hmm_breaks"]
    out["classifier_only_f1"] = d["cls_f1"]
    out.to_csv(a.out, index=False)

    def table(df, title):
        print(f"\n=== {title} ({len(df)} trips): HMM decoder, default vs tuned config ===")
        print(f"{'metric':<12}{'default mean':>13}{'tuned mean':>12}{'d mean':>9} |{'def med':>9}{'tuned med':>10}{'d med':>9}"
              f" | improved/tied/worse")
        for m in METRICS:
            x, y = df[f"default_{m}"], df[f"tuned_{m}"]
            dd = y - x
            print(f"{m:<12}{x.mean():>13.4f}{y.mean():>12.4f}{y.mean() - x.mean():>+9.4f} |{x.median():>9.4f}"
                  f"{y.median():>10.4f}{y.median() - x.median():>+9.4f} | {int((dd > 1e-9).sum())}/"
                  f"{int((dd.abs() <= 1e-9).sum())}/{int((dd < -1e-9).sum())}")
        print(f"breaks total: default {int(df.default_breaks.sum())}  tuned {int(df.tuned_breaks.sum())}")

    if (out.partition == "student_test").any():
        table(out[out.partition == "student_test"], "student_test (never seen by tuning)")
    if (out.partition == "train").any():
        table(out[out.partition == "train"], "remaining train OOF trips (not in the tuning subset)")
    if out.partition.nunique() > 1:
        table(out, "all unseen trips")
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
