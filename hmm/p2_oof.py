"""Out-of-fold (OOF) probabilities of P2's random forest for the 80 TRAIN trips (P3 utility).

Why: the HMM can only be tuned / evaluated fairly on train trips if the probabilities of a train trip
come from a model that never saw that trip. P2's final model was trained on all 80 train trips, so its
predictions on them would be in-sample.

What it does (P2's code is imported, not copied or edited):
  1. 5-fold GroupKFold over the 80 train trips, same splitter call as P2's train_models.py
     (GroupKFold has no randomness when it is not shuffled, so the folds are the same every run and are
      the same folds P2 used for model_comparison.csv).
  2. per fold: a RandomForest with P2's BEST_PARAMS (class_weight="balanced_subsample", random_state=42),
     trained on the sample P2's load_training_sample() draws from THAT FOLD'S TRAINING TRIPS ONLY.
  3. predict ALL candidate rows of the held-out trips. results/training_features/ only holds rows with
     label_valid==1, so the full label-free feature table of each train trip is rebuilt with P2's
     generate() + build_features() (the same call build_test_features.py uses) and cached in
     results/p2_oof_features/. It is checked against results/training_features/ on the shared rows.
  4. write results/p2_oof/train_<trip>.csv.gz   columns: trip, point_idx, arc_id, probability
     (+ results/p2_oof_meta/folds.csv: trip -> fold) and print per-fold Top-1 (on label_valid points only,
     labels used for scoring only).

Run from the repository root:  python -m hmm.p2_oof
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_processing"))        # P2/P1 modules import each other by bare name

import train_final_model as p2_final                      # noqa: E402  BEST_PARAMS (main() is guarded)
from candidate_gen import generate                        # noqa: E402
from features import FEATURE_COLUMNS, build_features      # noqa: E402
from train_models import (RANDOM_STATE, TRAIN_ROW_LIMITS, N_SPLITS,   # noqa: E402
                          get_trip_files, load_training_sample, trip_id_from_file)

OUT = REPO / "results" / "p2_oof"
FEAT = REPO / "results" / "p2_oof_features"
META = REPO / "results" / "p2_oof_meta"      # folds.csv, oof_fold_top1.csv (kept out of the prediction folder)


def full_features(trip):
    """All candidate rows of a train trip (label-free), cached on disk."""
    f = FEAT / f"train_{trip}.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    td = generate(REPO / "data" / "train" / trip)
    X = build_features(td)[["point_idx", "arc_id"] + FEATURE_COLUMNS]
    FEAT.mkdir(parents=True, exist_ok=True)
    X.to_csv(f, index=False, compression="gzip")
    return X


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-trips", type=int, default=None, help="debug: only the first N trips (folds are then fewer)")
    a = ap.parse_args(argv)

    files = get_trip_files()
    if a.max_trips:
        files = files[: a.max_trips]
    trips = [trip_id_from_file(f) for f in files]
    assert len(set(trips)) == len(trips) and all("train_" in f.name for f in files)   # train trips only
    print(f"train trips: {len(trips)}   features: {len(FEATURE_COLUMNS)}   params: {p2_final.BEST_PARAMS}")

    OUT.mkdir(parents=True, exist_ok=True)
    folds = list(GroupKFold(n_splits=N_SPLITS).split(trips, groups=trips))     # same call as P2

    # --- integrity: disjoint train / held-out in every fold, every trip held out exactly once
    seen = []
    for k, (tr, va) in enumerate(folds, 1):
        tr_t, va_t = {trips[i] for i in tr}, {trips[i] for i in va}
        assert not (tr_t & va_t), f"fold {k}: a trip is in both the training and held-out set"
        assert len(tr_t) + len(va_t) == len(trips)
        seen += sorted(va_t)
    assert sorted(seen) == sorted(trips) and len(seen) == len(set(seen)), "a trip is not held out exactly once"
    print(f"integrity OK: {len(folds)} folds, train/held-out disjoint in each, each of the {len(trips)} trips held out once")

    fold_rows, trip_fold, t0 = [], {}, time.time()
    for k, (tr, va) in enumerate(folds, 1):
        train_files = [files[i] for i in tr]
        X, y = load_training_sample(train_files, TRAIN_ROW_LIMITS["Random Forest"], seed=RANDOM_STATE)
        model = RandomForestClassifier(**p2_final.BEST_PARAMS, class_weight="balanced_subsample",
                                       n_jobs=2, random_state=RANDOM_STATE)
        model.fit(X, y)
        pos = list(model.classes_).index(1)
        correct = total = 0
        for i in va:
            trip = trips[i]
            trip_fold[trip] = k
            F = full_features(trip)
            Xh = F[FEATURE_COLUMNS].astype(np.float32)
            assert np.isfinite(Xh.to_numpy()).all()
            p = np.concatenate([model.predict_proba(Xh.iloc[s:s + 5000])[:, pos] for s in range(0, len(Xh), 5000)])
            out = pd.DataFrame({"trip": trip, "point_idx": F["point_idx"].to_numpy(),
                                "arc_id": F["arc_id"].to_numpy(), "probability": p})
            out.to_csv(OUT / f"train_{trip}.csv.gz", index=False, compression="gzip")
            # Top-1 on label_valid points (labels from P2's training_features file, scoring only)
            lab = pd.read_csv(files[i], usecols=["point_idx", "arc_id", "label"])
            same = lab.merge(F, on=["point_idx", "arc_id"], how="left", suffixes=("", "_full"))
            assert len(same) == len(lab) and same["dist"].notna().all(), f"{trip}: rows missing from full features"
            m = lab.merge(out, on=["point_idx", "arc_id"]).merge(F[["point_idx", "arc_id", "dist"]], on=["point_idx", "arc_id"])
            best = m.sort_values(["point_idx", "probability", "dist"], ascending=[True, False, True],
                                 kind="stable").drop_duplicates("point_idx")
            correct += int(best["label"].sum())
            total += len(best)
        fold_rows.append({"fold": k, "train_trips": len(tr), "heldout_trips": len(va),
                          "train_rows": len(X), "heldout_points": total, "top1": correct / max(total, 1)})
        print(f"fold {k}: train trips {len(tr)}, held-out {len(va)}, train rows {len(X):,}, "
              f"held-out labelled points {total:,}, Top-1 {correct / max(total, 1):.4f}   [{time.time() - t0:.0f}s]")

    fr = pd.DataFrame(fold_rows)
    META.mkdir(parents=True, exist_ok=True)
    fr.to_csv(META / "oof_fold_top1.csv", index=False)
    pd.DataFrame({"trip": list(trip_fold), "fold": list(trip_fold.values())}).to_csv(META / "folds.csv", index=False)
    print(f"\nOOF Top-1 per fold: {', '.join(f'{v:.4f}' for v in fr.top1)}   mean {fr.top1.mean():.4f}")
    p2 = REPO / "results" / "model_training" / "fold_results.csv"
    if p2.exists():
        r = pd.read_csv(p2, header=0)
        r = r[r.iloc[:, 1] == "Random Forest"] if r.shape[1] > 2 else r
        print(f"\nP2's own random-forest fold rows ({p2.name}, a different RF config: 100 trees, depth 18):")
        print(r.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
