"""Render maps for 5 student_test trips chosen by a FIXED RULE from the per-trip table (not by looks).

Input: results/hmm_rf_tuned_vs_default.csv (columns tuned_f1, classifier_only_f1, partition, trip_id).
gain = tuned_f1 - classifier_only_f1 per student_test trip. Rule, applied in this order, no repeats:
  1 helps a lot        : largest gain
  2 does not help/fails: smallest gain
  3 clean case         : highest tuned_f1 among the rest
  4 weakest result     : lowest tuned_f1 among the rest
  5 typical            : gain closest to the median gain among the rest
The choice is saved to results/maps/selection_rf_tuned.csv. Reference route is drawn (dev partition only); the HMM
uses hmm/config.json. evaluator_test is never touched.

    python -m viz.render_selected --probs results/p2/probs_all.csv.gz --table results/hmm_rf_tuned_vs_default.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hmm.graph import REPO
from hmm.probs_io import load_probs
from hmm.viterbi import load_config
from viz.folium_map import render_maps


def choose(t):
    t = t[t.partition == "student_test"].copy()
    t["gain"] = t.tuned_f1 - t.classifier_only_f1
    picks, rest = [], t.copy()

    def take(why, idx):
        picks.append((why, rest.loc[idx]))
        return rest.drop(idx)

    rest = take("helps a lot (largest tuned_f1 - classifier_only_f1)", rest.gain.idxmax())
    rest = take("does not help / fails (smallest gain)", rest.gain.idxmin())
    rest = take("clean case (highest tuned_f1)", rest.tuned_f1.idxmax())
    rest = take("weakest result (lowest tuned_f1)", rest.tuned_f1.idxmin())
    rest = take("typical (gain closest to the median gain)", (rest.gain - rest.gain.median()).abs().idxmin())
    return picks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "maps"))
    ap.add_argument("--label", default="random forest (P2) tuned")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args(argv)
    t = pd.read_csv(a.table, dtype={"trip_id": str})
    picks = choose(t)
    sel = pd.DataFrame([{"reason": w, "trip_id": r.trip_id, "classifier_only_f1": r.classifier_only_f1,
                         "tuned_f1": r.tuned_f1, "gain": r.gain, "n_points": r.n_points} for w, r in picks])
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    sel.to_csv(Path(a.out_dir) / "selection_rf_tuned.csv", index=False)
    print(sel.round(4).to_string(index=False))
    lp = load_probs(a.probs, ["student_test"])
    trips = [("student_test", r.trip_id) for _, r in picks]
    cfg = load_config()
    done = render_maps(lp.by_trip, trips, a.out_dir, cfg, a.label, False, not a.no_png, n=5)
    print(f"{len(done)} maps, {sum(1 for d in done if d[1])} PNGs in {a.out_dir}")


if __name__ == "__main__":
    main()
