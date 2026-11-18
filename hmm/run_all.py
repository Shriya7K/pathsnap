"""ONE command for the whole HMM side, whatever probability file P2 delivers.

  python -m hmm.run_all --probs <file> --label "<model name>" [--oof] [--tune-trials 40] [--tune-max-trips N]

Steps, in order (the run stops at the first blocking error):
  1. check_probs   validate / normalise the file (hmm/probs_io.py)
  2. evaluate      classifier-only vs classifier + HMM on all train + student_test trips
  3. diagnose      error types, fill_path effect, extra route arcs (same decode as step 2)
  4. tune          only if --tune-trials > 0 (needs --oof for train trips); then re-evaluate with the tuned params
  5. maps          Folium maps (+ PNG) for 5 student_test trips
  6. latency       per-stage timing on a few trips
Everything is written to results/hmm_<label_slug>/ . hmm/config.json is changed ONLY with --write-config
(together with --tune-trials > 0). evaluator_test is never read.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import infer
from .diagnose import diagnose_trip, summarize_diagnostics
from .evaluate import SYNTH_BANNER, add_prob_kind, evaluate_many, is_synthetic, summarize
from .graph import DATA_ROOT, DEV_PARTITIONS, REPO, build_network, load_trip
from .probs_io import CANDS_DIR, ProbsError, add_args, load_probs, require_oof_for_train
from .tune import SPACE, tune
from .viterbi import CONFIG_PATH, decode_detailed, load_config


def slugify(label):
    s = "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


class Log:
    """print() and remember, so the whole run is saved as run_summary.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(str(text))


def pick_latency_trips(df, k=5):
    """k trips spread over the size range (smallest ... largest), preferring student_test."""
    d = df[df.partition == "student_test"] if (df.partition == "student_test").sum() >= 2 else df
    d = d.sort_values("n_points")
    idx = sorted(set(np.linspace(0, len(d) - 1, min(k, len(d))).round().astype(int)))
    return [(r.partition, r.trip_id) for r in d.iloc[idx].itertuples()]


def latency_rows(picks, probs_by_trip, cfg_path, data_root):
    """Stage timings (seconds). 'decode_from_probs' uses the supplied probabilities (network build + Viterbi).
    The API path (candidates -> score_candidates -> decode) uses whatever scorer is registered in hmm.infer
    (the distance-only baseline unless a model was set), so it excludes P2's model time unless set_model() was used."""
    rows = []
    cfg = load_config(cfg_path)
    for part, trip in picks:
        gps, nodes, arcs, _ = load_trip(part, trip, data_root, with_route=False)      # .route is not read
        t0 = time.perf_counter()
        net = build_network(gps, nodes, arcs)
        t1 = time.perf_counter()
        t_net = t1 - t0
        probs = probs_by_trip.get(trip)
        t_dec = np.nan
        if probs is not None and len(probs):
            t2 = time.perf_counter()
            decode_detailed(gps, nodes, arcs, probs, cfg, network=net)
            t_dec = time.perf_counter() - t2
        a0 = time.perf_counter()
        cands = infer.generate_candidates(gps, nodes, arcs)
        a1 = time.perf_counter()
        pr = infer.score_candidates(cands)
        a2 = time.perf_counter()
        decode_detailed(gps, nodes, arcs, pr, cfg)
        a3 = time.perf_counter()
        rows.append({"partition": part, "trip_id": trip, "n_points": len(gps),
                     "network_build_s": round(t_net, 3), "decode_from_probs_s": round(t_dec, 3),
                     "api_candidates_s": round(a1 - a0, 3), "api_score_s": round(a2 - a1, 3),
                     "api_decode_s": round(a3 - a2, 3), "api_total_s": round(a3 - a0, 3),
                     "ms_per_point_api": round(1000 * (a3 - a0) / max(len(gps), 1), 3)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--label", required=True, help='model name, e.g. "xgboost v2"; names the output folder')
    ap.add_argument("--oof", dest="probs_are_oof", action="store_true", help="same as --probs-are-oof")
    ap.add_argument("--tune-trials", type=int, default=0)
    ap.add_argument("--tune-max-trips", type=int, default=None)
    ap.add_argument("--tune-seed", type=int, default=0)
    ap.add_argument("--tune-partitions", nargs="+", default=list(DEV_PARTITIONS))
    ap.add_argument("--write-config", action="store_true", help="after tuning, ALSO write the best params to hmm/config.json")
    ap.add_argument("--partitions", nargs="+", default=list(DEV_PARTITIONS))
    ap.add_argument("--config", default=None, help="base decoder config (default hmm/config.json)")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--cands-dir", default=str(CANDS_DIR))
    ap.add_argument("--out-root", default=str(REPO / "results"))
    ap.add_argument("--maps-n", type=int, default=5)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="allow writing into an existing non-empty output folder")
    add_args(ap)
    a = ap.parse_args(argv)
    log = Log()
    t_all = time.time()

    slug = slugify(a.label)
    out_dir = Path(a.out_root) / f"hmm_{slug}"
    synth = is_synthetic(a.probs) or "synthetic" in slug
    if not slug or slug == "baseline":
        log(f"BLOCKING: label '{a.label}' maps to the reserved/empty folder name hmm_{slug}; choose another label "
            f"(results/hmm_baseline* belongs to the distance-only baseline run).")
        return 1
    if out_dir.exists() and any(out_dir.iterdir()) and not a.overwrite:
        log(f"BLOCKING: {out_dir} already exists and is not empty (results are never overwritten silently). "
            f"Use another --label or pass --overwrite.")
        return 1
    if a.write_config and a.tune_trials <= 0:
        log("BLOCKING: --write-config needs --tune-trials > 0.")
        return 1
    if a.write_config and synth:
        log("BLOCKING: --write-config refused for synthetic data.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"=== hmm.run_all | score source: {a.label} | output: {out_dir} ===")
    if synth:
        log(SYNTH_BANNER)

    # ---- 1. check_probs ------------------------------------------------------------------------------
    log("\n[1/6] check_probs")
    try:
        lp = load_probs(a.probs, a.partitions, a.data_root, a.cands_dir, accept_renames=a.accept_renames,
                        renormalise=a.renormalise, fill_missing=a.fill_missing, drop_duplicates=a.drop_duplicates,
                        allow_partial=a.allow_partial, candidates="require")
    except ProbsError as e:
        log(f"BLOCKING PROBLEM(S) in {a.probs}:\n{e}\nStopped at step 1.")
        (out_dir / "run_summary.txt").write_text("\n".join(log.lines))
        return 1
    log(lp.text)
    (out_dir / "check_probs_report.txt").write_text(lp.text)
    if a.tune_trials > 0:
        msg = require_oof_for_train(a.tune_partitions, a.probs_are_oof, lp)
        if msg:
            log("\n" + "!" * 78 + f"\nBLOCKING: cannot tune: {msg}\n" + "!" * 78 + "\nStopped before evaluation.")
            (out_dir / "run_summary.txt").write_text("\n".join(log.lines))
            return 2
    if "train" in a.partitions and not a.probs_are_oof:
        log("NOTE: train trips included without --oof: if the model was fitted on them, their metrics are in-sample "
            "(column prob_kind in the CSV records this).")

    base_cfg = load_config(a.config)
    trips = lp.trips

    # ---- 2+3. evaluate + diagnose (one decode per trip) ---------------------------------------------
    log("\n[2/6] evaluate  +  [3/6] diagnose")
    diag_rows = []

    def on_trip(row, keep):
        diag_rows.append(diagnose_trip(row["partition"], row["trip_id"], keep, a.cands_dir))

    t0 = time.time()
    df = evaluate_many(lp.by_trip, trips, base_cfg, a.data_root, a.cands_dir, verbose=False, on_trip=on_trip)
    wall = time.time() - t0
    add_prob_kind(df, a.probs_are_oof)
    df.insert(0, "score_source", a.label)
    df.to_csv(out_dir / "comparison.csv", index=False)
    dg = pd.DataFrame(diag_rows)
    dg.to_csv(out_dir / "diagnose_by_trip.csv", index=False)
    s_eval = summarize(df, a.label, wall, table=False)
    s_diag = summarize_diagnostics(dg)
    (out_dir / "summary.txt").write_text(s_eval)
    (out_dir / "diagnose_summary.txt").write_text(s_diag)
    log(s_eval)
    log("\n" + s_diag)

    # ---- 4. tune (optional) --------------------------------------------------------------------------
    final_cfg, final_df, final_cfg_path, tuned = base_cfg, df, Path(a.config) if a.config else CONFIG_PATH, False
    if a.tune_trials > 0:
        log(f"\n[4/6] tune: {a.tune_trials} trials, partitions {a.tune_partitions}, seed {a.tune_seed}")
        tune_trips = [t for t in trips if t[0] in a.tune_partitions]
        best, rows = tune(lp.by_trip, tune_trips, base_cfg, a.tune_trials, a.tune_seed, out_dir / "tuning_log.csv",
                          a.label, a.tune_max_trips, a.data_root, a.tune_partitions, a.probs_are_oof, synth,
                          cands_dir=a.cands_dir)
        log(f"best trial {best['trial']}: objective {best['objective']:.4f} (starting config {rows[0]['objective']:.4f})")
        params = {k: best[k] for k in SPACE}
        final_cfg = {**base_cfg, **params}
        meta = {"status": f"TUNED for '{a.label}': random search seed {a.tune_seed}, {a.tune_trials} trials on "
                          f"{best['n_trips']} trips ({'+'.join(a.tune_partitions)}); see tuning_log.csv",
                "units": "metres, seconds, natural-log scores"}
        final_cfg_path = out_dir / "tuned_config.json"
        final_cfg_path.write_text(json.dumps({"_meta": meta, **final_cfg}, indent=2))
        log(f"tuned parameters written to {final_cfg_path}")
        if a.write_config:
            CONFIG_PATH.write_text(json.dumps({"_meta": meta, **final_cfg}, indent=2))
            log(f"--write-config: ALSO wrote {CONFIG_PATH}")
        else:
            log(f"{CONFIG_PATH} was NOT modified (no --write-config).")
        t0 = time.time()
        final_df = evaluate_many(lp.by_trip, trips, final_cfg, a.data_root, a.cands_dir, verbose=False)
        add_prob_kind(final_df, a.probs_are_oof)
        final_df.insert(0, "score_source", a.label)
        final_df.to_csv(out_dir / "comparison_tuned.csv", index=False)
        s_t = summarize(final_df, f"{a.label}, tuned parameters", time.time() - t0, table=False)
        (out_dir / "summary_tuned.txt").write_text(s_t)
        log("\nre-evaluation with the tuned parameters:\n" + s_t)
        log("NOTE: the tuned parameters were selected on these same trips, so this re-evaluation is not a held-out "
            "estimate (the tuning objective and these metrics share the trips).")
        tuned = True
    else:
        log("\n[4/6] tune: skipped (--tune-trials 0)")

    # ---- 5. maps ---------------------------------------------------------------------------------------
    log("\n[5/6] maps")
    from viz.folium_map import render_maps
    map_part = "student_test" if "student_test" in a.partitions else a.partitions[0]
    done = render_maps(lp.by_trip, [t for t in trips if t[0] == map_part], out_dir / "maps", final_cfg, a.label,
                       synth, not a.no_png, a.data_root, a.maps_n)
    log(f"{len(done)} maps, {sum(1 for d in done if d[1])} PNGs in {out_dir / 'maps'}")

    # ---- 6. latency ------------------------------------------------------------------------------------
    log("\n[6/6] latency")
    lat = latency_rows(pick_latency_trips(final_df, 5), lp.by_trip, final_cfg_path, a.data_root)
    lat.to_csv(out_dir / "latency.csv", index=False)
    scorer = ("registered model (set_model)" if infer._MODEL is not None
              else f"infer.default_scorer() = {infer.default_scorer()!r} (rf = P2's model, baseline = distance-only)")
    log(lat.to_string(index=False))
    log(f"API-path scorer used for the api_* timings: {scorer}")

    # ---- final summary -----------------------------------------------------------------------------------
    log("\n" + "=" * 78)
    log(f"FINAL SUMMARY  [{a.label}{', tuned' if tuned else ''}]   (classifier-only vs classifier + HMM)")
    log(summarize(final_df, a.label + (", tuned" if tuned else ""), None, table=False).split("\n10 WORST")[0])
    log(f"\nall files: {out_dir}   total time {time.time() - t_all:.1f}s")
    if synth:
        log(SYNTH_BANNER)
    (out_dir / "run_summary.txt").write_text("\n".join(log.lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
