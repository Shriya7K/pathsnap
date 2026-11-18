"""Side-by-side table of the systems, built ONLY from saved per-trip CSVs (no new decoding).

Systems (all with the same default hmm decoder config unless the CSVs say otherwise):
  distance-only alone        = cls_* columns of the baseline comparison CSV
  distance-only + HMM        = hmm_* columns of the baseline comparison CSV
  random forest alone        = cls_* columns of the random-forest comparison CSV
  random forest + HMM        = hmm_* columns of the random-forest comparison CSV
Subsets: all trips present in both files, and the student_test trips only.

    python -m hmm.compare_models --baseline results/hmm_baseline_comparison.csv \
        --rf results/hmm_rf_default_comparison.csv --out results/hmm_model_comparison.csv
"""
import argparse

import pandas as pd

METRICS = ["point_acc", "precision", "recall", "f1", "lcs_sim", "continuity"]


def build(base, rf):
    key = ["partition", "trip_id"]
    for d in (base, rf):
        d["trip_id"] = d["trip_id"].astype(str).str.zfill(8)
    common = base[key].merge(rf[key], on=key)
    base = base.merge(common, on=key).sort_values(key).reset_index(drop=True)
    rf = rf.merge(common, on=key).sort_values(key).reset_index(drop=True)
    systems = [("distance-only alone", base, "cls_"), ("distance-only + HMM", base, "hmm_"),
               ("random forest alone", rf, "cls_"), ("random forest + HMM", rf, "hmm_")]
    rows = []
    for subset, sel in (("all trips", lambda d: d.index == d.index), ("student_test", lambda d: d.partition == "student_test")):
        for name, d, pre in systems:
            m = sel(d)
            row = {"subset": subset, "system": name, "n_trips": int(m.sum())}
            for k in METRICS:
                row[f"{k}_mean"] = d.loc[m, pre + k].mean()
                row[f"{k}_median"] = d.loc[m, pre + k].median()
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--rf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    df = build(pd.read_csv(a.baseline, dtype={"trip_id": str}), pd.read_csv(a.rf, dtype={"trip_id": str})).round(4)
    df.to_csv(a.out, index=False)
    for subset, g in df.groupby("subset", sort=False):
        print(f"\n=== {subset} ({g.n_trips.iloc[0]} trips)  mean / median ===")
        t = pd.DataFrame({"system": g.system})
        for k in METRICS:
            t[k] = [f"{m:.4f}/{d:.4f}" for m, d in zip(g[f"{k}_mean"], g[f"{k}_median"])]
        print(t.to_string(index=False))
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
