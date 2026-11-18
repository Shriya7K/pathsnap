"""End-to-end inference latency per trip (what P4's API pays), on DEV trips only.

Per trip: load GPS + nodes + arcs (NOT timed, the API receives them), then time
  candidates (P1 generate) -> features (P2 build_features) -> model (predict_proba) -> decode (HMM/Viterbi).
The .route file is never read. evaluator_test is refused.

    python -m hmm.latency [--partition student_test] [--scorer rf|baseline] [--out results/hmm_latency.csv]
"""
import argparse
import sys
import time

import pandas as pd

from . import infer
from .graph import DEV_PARTITIONS, load_trip, REPO
from .viterbi import decode, load_config


def time_trip(partition, trip, scorer, cfg):
    gps, nodes, arcs, _ = load_trip(partition, trip, with_route=False)
    t0 = time.perf_counter()
    cands = infer.generate_candidates(gps, nodes, arcs)
    t1 = time.perf_counter()
    tm = {}
    if scorer == "rf":
        probs = infer.score_rf(cands.attrs["trip_data"], cands, timing=tm)
    else:
        probs = infer.score_candidates(cands, scorer="baseline")
        tm = {"features_s": 0.0, "model_s": time.perf_counter() - t1}
    t2 = time.perf_counter()
    decode(gps, nodes, arcs, probs, config=cfg)
    t3 = time.perf_counter()
    return {"partition": partition, "trip_id": trip, "n_points": len(gps), "n_candidates": len(cands),
            "scorer": scorer, "candidates_s": t1 - t0, "features_s": tm["features_s"], "model_s": tm["model_s"],
            "decode_s": t3 - t2, "total_s": t3 - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--scorer", default="rf", choices=infer.SCORERS)
    ap.add_argument("--trips", nargs="*", default=None)
    ap.add_argument("--config", default="hmm/config.json")
    ap.add_argument("--out", default="results/hmm_latency.csv")
    a = ap.parse_args(argv)
    if a.partition not in DEV_PARTITIONS:
        print(f"REFUSED: partition {a.partition!r}; allowed {DEV_PARTITIONS}")
        return 2
    cfg = load_config(a.config)
    trips = a.trips or sorted(p.name for p in (REPO / "data" / a.partition).iterdir() if p.is_dir())
    if a.scorer == "rf":
        infer.load_rf()                                              # model load is a one-off at API start-up
    rows = []
    for t in trips:
        r = time_trip(a.partition, t, a.scorer, cfg)
        rows.append(r)
        print(f"{t}: {r['n_points']:>6} pts  cand {r['candidates_s']:.2f}s  feat {r['features_s']:.2f}s  "
              f"model {r['model_s']:.2f}s  decode {r['decode_s']:.2f}s  total {r['total_s']:.2f}s")
    df = pd.DataFrame(rows).round(4)
    df.to_csv(a.out, index=False)
    w = df.loc[df.total_s.idxmax()]
    print(f"\nslowest trip: {w.trip_id}  {int(w.n_points)} points  total {w.total_s:.2f}s")
    print(f"mean total {df.total_s.mean():.2f}s  median {df.total_s.median():.2f}s  written {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
