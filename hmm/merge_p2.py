"""Merge the imported student_test file and the imported train OOF file into one probability file.

Adds a `fold` column: the GroupKFold fold (1..5, from results/p2_oof_meta/folds.csv) for train rows, -1 for
student_test rows (the final model never saw them). hmm.probs_io reads `fold >= 0` on every train row as
"declares out-of-fold".

    python -m hmm.merge_p2 --student results/p2/student_test_probs.csv.gz --train-oof results/p2/train_oof_probs.csv.gz \
        --folds results/p2_oof_meta/folds.csv --out results/p2/probs_all.csv.gz
"""
import argparse

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True)
    ap.add_argument("--train-oof", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    st = pd.read_csv(a.student, dtype={"trip_id": str})
    tr = pd.read_csv(a.train_oof, dtype={"trip_id": str})
    fo = pd.read_csv(a.folds, dtype={"trip": str}).rename(columns={"trip": "trip_id"})
    assert set(st.partition) == {"student_test"} and set(tr.partition) == {"train"}
    tr = tr.merge(fo, on="trip_id", how="left")
    assert tr["fold"].notna().all(), "a train trip has no fold"
    st["fold"] = -1
    df = pd.concat([tr, st], ignore_index=True)
    df["fold"] = df["fold"].astype(int)
    df.to_csv(a.out, index=False)
    print(f"merged: {df.trip_id.nunique()} trips ({tr.trip_id.nunique()} train OOF + {st.trip_id.nunique()} student_test), "
          f"{len(df):,} rows -> {a.out}")


if __name__ == "__main__":
    main()
