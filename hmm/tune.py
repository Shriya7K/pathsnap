"""Tune the decoder parameters (random search) on train + student_test ONLY.

  python -m hmm.tune --probs <probs> --n-trials 40 --seed 0 --probs-are-oof [--write-config]

* Only dev partitions (train, student_test) are ever loaded; evaluator_test is refused.
* TRAIN TRIPS: tuning on them is refused unless --probs-are-oof is given (the classifier's probabilities
  on its own training trips are over-confident). With student_test only (--partitions student_test) no flag is needed.
* Objective (maximise), --objective f1_lcs (default) = mean over trips of 0.5*(route F1 + LCS similarity), or
  --objective f1 = mean route F1 of the final (fill_path) route. Every trial also logs pointwise accuracy,
  precision, recall, and the F1 of the route WITHOUT fill_path, so fill_path cannot silently inflate the objective.
  The reference is used only to SCORE a trial, never inside the decoder.
* --max-trips N picks a seeded random subset; --exclude-trips drops trips (e.g. unreliable references);
  --trips-out saves the chosen trip list.
* Every trial (params, seed, metrics, time) is appended to the log CSV. Trial 0 = the current config.
* hmm/config.json is changed ONLY by --write-config. Synthetic probabilities are refused for it.
"""
import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import SYNTH_BANNER, SYNTH_DIR, is_synthetic, pointwise_accuracy, route_metrics
from .graph import DATA_ROOT, DEV_PARTITIONS, REPO, build_network, load_trip
from .probs_io import CANDS_DIR, add_args, load_from_args, require_oof_for_train
from .viterbi import CONFIG_PATH, decode_detailed, load_config

# search space: (low, high, log-uniform?)
SPACE = {
    "beta": (2.0, 60.0, True),              # m   - tolerance of |route_dist - gps_dist|
    "switch_penalty": (0.0, 4.0, False),    # log - extra cost for changing arc
    "unreach_penalty": (4.0, 25.0, False),  # log - cost of an unreachable transition
    "max_path_m": (100.0, 600.0, False),    # m   - base bounded-path distance
    "temperature": (0.5, 3.0, True),        # emission softening
    "gap_scale_mps": (5.0, 50.0, True),     # m/s - widening per second of time gap
}


def sample(rng):
    out = {}
    for k, (lo, hi, log) in SPACE.items():
        v = np.exp(rng.uniform(np.log(lo), np.log(hi))) if log else rng.uniform(lo, hi)
        out[k] = round(float(v), 4)
    return out


def run_trial(params, base_cfg, data, objective="f1_lcs"):
    cfg = {**base_cfg, **params}
    rows = []
    for ref, net, gps, nodes, arcs, probs, lab in data:
        res = decode_detailed(gps, nodes, arcs, probs, cfg, network=net)
        m = route_metrics(res.route, ref, net)                         # final route (fill_path as configured)
        m["f1_raw"] = route_metrics(res.raw_route, ref, net)["f1"]     # same decode WITHOUT fill_path insertions
        m["point_acc"] = pointwise_accuracy(res.point_arcs, lab) if lab is not None else np.nan
        rows.append(m)
    df = pd.DataFrame(rows)
    obj = df.f1 if objective == "f1" else 0.5 * (df.f1 + df.lcs_sim)
    return {"objective": float(obj.mean()), "mean_f1": df.f1.mean(), "mean_f1_raw": df.f1_raw.mean(),
            "mean_precision": df.precision.mean(), "mean_recall": df.recall.mean(),
            "mean_point_acc": df.point_acc.mean(), "mean_lcs_sim": df.lcs_sim.mean(),
            "mean_continuity": df.continuity.mean(), "total_breaks": int(df.breaks.sum())}


def tune(by_trip, trips, base_cfg, n_trials, seed, log_path, label=None, max_trips=None,
         data_root=DATA_ROOT, partitions=DEV_PARTITIONS, probs_oof=False, synthetic=False, verbose=True,
         objective="f1_lcs", exclude=(), cands_dir=CANDS_DIR, trips_out=None):
    """Random search. Returns (best_row, log_rows). Does NOT write hmm/config.json.
    trips: [(partition, trip_id)]; by_trip: {trip_id: probs DataFrame}."""
    if any(p not in DEV_PARTITIONS for p, _ in trips):
        raise SystemExit("evaluator_test (or any non-dev partition) must not be used for tuning.")
    rng = np.random.default_rng(seed)
    trips = [t for t in trips if t[1] in by_trip and len(by_trip[t[1]]) > 0]   # trips with no candidates carry no signal
    trips = [t for t in trips if t[1] not in set(exclude)]
    if max_trips and len(trips) > max_trips:
        pick = np.random.default_rng(seed).choice(len(trips), max_trips, replace=False)   # subset: seeded, separate rng
        trips = [trips[i] for i in sorted(pick)]
    if trips_out:
        Path(trips_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trips, columns=["partition", "trip_id"]).assign(seed=seed).to_csv(trips_out, index=False)
    if verbose:
        print(f"Tuning on {len(trips)} trips from {list(partitions)}; {n_trials} trials; seed {seed}")
    data = []                                   # load every trip once; reuse across trials
    for part, trip in trips:
        gps, nodes, arcs, ref = load_trip(part, trip, data_root)
        net = build_network(gps, nodes, arcs, cache_key=f"{part}_{trip}")
        lab_f = Path(cands_dir) / f"{part}_{trip}_points.csv.gz"           # P1 labels: scoring only (point accuracy)
        lab = pd.read_csv(lab_f) if lab_f.exists() else None
        data.append((list(ref), net, gps, nodes, arcs, by_trip[trip], lab))

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_rows, best = [], None
    for k in range(n_trials):
        params = {kk: base_cfg[kk] for kk in SPACE} if k == 0 else sample(rng)    # trial 0 = baseline config
        t0 = time.time()
        r = run_trial(params, base_cfg, data, objective)
        row = {"trial": k, "seed": seed, "n_trips": len(trips), "partitions": "+".join(partitions),
               "is_baseline": k == 0, **params, **r, "seconds": round(time.time() - t0, 1),
               "synthetic": synthetic, "score_source": label, "train_probs_oof": probs_oof, "objective_name": objective}
        log_rows.append(row)
        if best is None or r["objective"] > best["objective"]:
            best = row
        if verbose:
            print(f"trial {k:>3}  obj={r['objective']:.4f}  f1={r['mean_f1']:.3f} f1_raw={r['mean_f1_raw']:.3f} "
                  f"P={r['mean_precision']:.3f} R={r['mean_recall']:.3f} acc={r['mean_point_acc']:.3f} "
                  f"lcs={r['mean_lcs_sim']:.3f} breaks={r['total_breaks']}  {params}  ({row['seconds']}s)")
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)  # append each trial
    return best, log_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--partitions", nargs="+", default=list(DEV_PARTITIONS))
    ap.add_argument("--max-trips", type=int, default=None, help="random subset of this many trips (seeded)")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--objective", choices=["f1_lcs", "f1"], default="f1_lcs")
    ap.add_argument("--exclude-trips", nargs="*", default=[], help="trip ids to leave out of tuning")
    ap.add_argument("--trips-out", default=None, help="write the chosen tuning trips to this CSV")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--cands-dir", default=str(CANDS_DIR))
    ap.add_argument("--log", default=None)
    ap.add_argument("--label", default=None, help='score source, e.g. "distance-only baseline"; stored in the log')
    ap.add_argument("--write-config", action="store_true", help="write the best params to hmm/config.json")
    add_args(ap)
    a = ap.parse_args(argv)

    synth = is_synthetic(a.probs)
    if synth and a.write_config:
        raise SystemExit("Refusing --write-config: probabilities are synthetic (debug data, not results).")
    log_path = Path(a.log) if a.log else (SYNTH_DIR / "hmm_tuning_log_SYNTHETIC.csv" if synth
                                          else REPO / "results" / "hmm_tuning_log.csv")
    if synth:
        print(SYNTH_BANNER)
    if a.label:
        print(f"Score source: {a.label}")

    lp = load_from_args(a, a.partitions, a.data_root, a.cands_dir)
    msg = require_oof_for_train(a.partitions, a.probs_are_oof, lp)
    if msg:
        print("\n" + "!" * 78 + f"\nREFUSING TO TUNE: {msg}\n" + "!" * 78)
        raise SystemExit(2)
    if "train" in a.partitions and a.probs_are_oof and not lp.declares_oof:
        print("WARNING: relying on --probs-are-oof; the file itself has no fold/split column declaring out-of-fold.")

    base = load_config()
    best, rows = tune(lp.by_trip, lp.trips, base, a.n_trials, a.seed, log_path, a.label, a.max_trips,
                      a.data_root, a.partitions, a.probs_are_oof, synth, objective=a.objective,
                      exclude=a.exclude_trips, cands_dir=a.cands_dir, trips_out=a.trips_out)
    print(f"\nBest trial {best['trial']}  objective={best['objective']:.4f} (baseline config {rows[0]['objective']:.4f})")
    print({k: best[k] for k in SPACE})
    if a.write_config:
        cfg = json.load(open(CONFIG_PATH))
        cfg.update({k: best[k] for k in SPACE})
        cfg["_meta"] = {"status": f"TUNED {date.today().isoformat()}: random search, seed {a.seed}, {a.n_trials} trials, "
                                  f"objective {a.objective} = {best['objective']:.4f} (default config {rows[0]['objective']:.4f}) "
                                  f"on {best['n_trips']} dev trips ({', '.join(a.partitions)}, out-of-fold probs: "
                                  f"{a.probs_are_oof}); score source: {a.label}; see {log_path.name}",
                        "units": "metres, seconds, natural-log scores"}
        json.dump(cfg, open(CONFIG_PATH, "w"), indent=2)
        print(f"Wrote best params to {CONFIG_PATH}")
    print(f"Log: {log_path}")
    if synth:
        print(SYNTH_BANNER)


if __name__ == "__main__":
    main()
