"""Classifier-only vs classifier+HMM comparison (Person 3).

For each trip:   classifier-only = argmax prob per GPS point (no road-network reasoning)
                 HMM             = hmm.viterbi.decode on the same probabilities

Metrics (reference = the trip's .route; used ONLY here, never as decoder input)
  * pointwise accuracy  - only if P1's point->arc labels exist (results/candidates/*_points.csv.gz)
  * route arc precision / recall / F1   (set of arcs, order ignored)
  * LCS similarity      (order-aware): LCS(pred, ref) / max(len(pred), len(ref))
  * route continuity    : fraction of consecutive predicted arcs that are connected in the
                          graph (next.from_node == prev.to_node) and number of breaks

Usage:
  python -m hmm.evaluate --probs <probs.csv[.gz]> --partitions train student_test
Output: results/hmm_comparison.csv  (synthetic probabilities are routed to results/hmm_synthetic/)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import DATA_ROOT, DEV_PARTITIONS, REPO, build_network, load_trip, norm_trip_id
from .probs_io import add_args, load_from_args
from .viterbi import decode_detailed, dedupe, load_config

CANDS_DIR = REPO / "results" / "candidates"      # written by P1's candidates.py (git-ignored)
SYNTH_DIR = REPO / "results" / "hmm_synthetic"
SYNTH_BANNER = "*** SYNTHETIC DEBUGGING DATA - NOT RESULTS. Do not report anything computed from it. ***"


def is_synthetic(path):
    return "synthetic" in Path(path).name.lower()


# --------------------------------------------------------------------------- metrics
def lcs_length(a, b):
    """Longest common subsequence length, bit-parallel (Hyyro): fast for 1000s of arcs."""
    if not a or not b:
        return 0
    masks = {}
    for i, x in enumerate(a):
        masks[x] = masks.get(x, 0) | (1 << i)
    full = (1 << len(a)) - 1
    v = full
    for y in b:
        u = v & masks.get(y, 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")          # zero bits of v = LCS length


def route_metrics(pred, ref, net):
    """Sequence-level metrics of a predicted arc route vs the reference route."""
    sp, sr = set(pred), set(ref)
    inter = len(sp & sr)
    prec = inter / len(sp) if sp else 0.0
    rec = inter / len(sr) if sr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    lcs = lcs_length(list(pred), list(ref))
    sim = lcs / max(len(pred), len(ref), 1)          # penalises extra AND missing arcs
    lcs_rec = lcs / max(len(ref), 1)                 # share of the reference recovered in order
    return {"precision": prec, "recall": rec, "f1": f1, "lcs_sim": sim, "lcs_recall": lcs_rec,
            **continuity(pred, net)}


def continuity(route, net):
    """Fraction of consecutive arc pairs that are directly connected, and the number of breaks."""
    if len(route) < 2:
        return {"continuity": 1.0, "breaks": 0}
    r = np.asarray(route)
    ok = net.to_node[r[:-1]] == net.from_node[r[1:]]
    return {"continuity": float(ok.mean()), "breaks": int((~ok).sum())}


def classifier_only(probs_trip, n_points):
    """argmax per point (ties -> smaller arc id). Returns arc per point (None if no candidates)."""
    p = probs_trip.sort_values(["point_idx", "prob", "arc_id"], ascending=[True, False, True])
    best = p.drop_duplicates("point_idx", keep="first")
    out = [None] * n_points
    for i, a in zip(best["point_idx"].to_numpy(), best["arc_id"].to_numpy()):
        if 0 <= i < n_points:
            out[int(i)] = int(a)
    return out


def pointwise_accuracy(point_arcs, labels):
    """Accuracy over points with a RELIABLE label (P1's label_valid). None-predictions count as wrong."""
    valid = labels["label_valid"].to_numpy().astype(bool)
    true = labels["true_arc"].to_numpy()
    pred = np.array([-1 if a is None else a for a in point_arcs])
    return float((pred[valid] == true[valid]).mean()) if valid.any() else np.nan


# --------------------------------------------------------------------------- one trip
def evaluate_trip(partition, trip_id, probs_trip, cfg, data_root=DATA_ROOT, cands_dir=CANDS_DIR, keep=None):
    """Metrics row for one trip. If `keep` is a dict it receives the decoded objects (res, net, gps, nodes,
    arcs, ref, cls_points) so a caller (hmm.diagnose) can reuse them without decoding twice."""
    t_start = time.time()
    gps, nodes, arcs, ref = load_trip(partition, trip_id, data_root)
    net = build_network(gps, nodes, arcs, cache_key=f"{partition}_{trip_id}")
    n = len(gps)

    t0 = time.time()
    res = decode_detailed(gps, nodes, arcs, probs_trip, cfg, network=net)
    secs = time.time() - t0

    cls_pts = classifier_only(probs_trip, n)
    if keep is not None:
        keep.update(res=res, net=net, gps=gps, nodes=nodes, arcs=arcs, ref=ref, cls_points=cls_pts)
    cls_route = dedupe(a for a in cls_pts if a is not None)
    cm = route_metrics(cls_route, list(ref), net)
    hm = route_metrics(res.route, list(ref), net)
    raw_c, fil_c = continuity(res.raw_route, net), continuity(res.filled_route, net)

    # pointwise accuracy needs P1's labels (derived from the reference route); skip if absent
    lab_f = Path(cands_dir) / f"{partition}_{trip_id}_points.csv.gz"
    cls_acc = hmm_acc = cand_miss = label_valid_pct = np.nan
    if lab_f.exists():
        lab = pd.read_csv(lab_f)
        cls_acc = pointwise_accuracy(cls_pts, lab)
        hmm_acc = pointwise_accuracy(res.point_arcs, lab)
        valid = lab["label_valid"].to_numpy().astype(bool)
        label_valid_pct = float(valid.mean() * 100)
        has = set(zip(probs_trip["point_idx"].to_numpy(), probs_trip["arc_id"].to_numpy()))
        hit = np.array([(i, a) in has for i, a in zip(lab["point_idx"], lab["true_arc"])])
        cand_miss = float((~hit[valid]).mean()) if valid.any() else np.nan

    row = {"trip_id": trip_id, "partition": partition, "n_points": n, "n_ref_arcs": len(ref)}
    for k, v in cm.items():
        row[f"cls_{k}"] = v
    row["cls_point_acc"] = cls_acc
    for k, v in hm.items():
        row[f"hmm_{k}"] = v
    row["hmm_point_acc"] = hmm_acc
    row.update({"hmm_raw_continuity": raw_c["continuity"], "hmm_raw_breaks": raw_c["breaks"],
                "hmm_filled_continuity": fil_c["continuity"], "hmm_filled_breaks": fil_c["breaks"],
                "ref_breaks": continuity(list(ref), net)["breaks"]})
    row.update({"delta_f1": hm["f1"] - cm["f1"], "delta_lcs_sim": hm["lcs_sim"] - cm["lcs_sim"],
                "delta_point_acc": hmm_acc - cls_acc, "delta_continuity": hm["continuity"] - cm["continuity"]})
    # diagnostics to explain trips where the HMM does not help
    gaps = np.diff(gps["t"].to_numpy())
    row.update({"diag_avg_cands": round(len(probs_trip) / n, 2),
                "diag_empty_points": res.stats["n_empty_points"],
                "diag_cand_miss_rate": cand_miss,
                "diag_label_valid_pct": label_valid_pct,   # low = reference route does not cover the track
                "diag_max_gap_s": float(gaps.max()) if len(gaps) else 0.0,
                "diag_restarts": res.stats["n_restarts"],
                "diag_unreach_frac": round(res.stats.get("unreach_frac", 0.0), 4),
                "diag_decode_s": round(secs, 2),
                "diag_trip_total_s": round(time.time() - t_start, 2)})   # load + network + decode + metrics
    return row


# --------------------------------------------------------------------------- many trips
def list_trips(partitions, data_root=DATA_ROOT):
    out = []
    for part in partitions:
        if part not in DEV_PARTITIONS:
            raise SystemExit(f"Partition '{part}' is not allowed in development code (use {DEV_PARTITIONS}).")
        out += [(part, d.name) for d in sorted((Path(data_root) / part).iterdir()) if d.is_dir()]
    return out


def evaluate_many(by_trip, trips, cfg, data_root=DATA_ROOT, cands_dir=CANDS_DIR, verbose=True, on_trip=None):
    """by_trip: {trip_id: probs DataFrame} (from hmm.probs_io.load_probs). A trip with an EMPTY frame
    (no candidates at all) is evaluated normally: no points get an arc and the route is empty.
    A trip that is absent from by_trip is skipped and listed. on_trip(row, keep) is called per trip."""
    rows, skipped = [], []
    for part, trip in trips:
        if trip not in by_trip:
            skipped.append(f"{part}/{trip}")
            continue
        keep = {} if on_trip else None
        rows.append(evaluate_trip(part, trip, by_trip[trip], cfg, data_root, cands_dir, keep))
        if on_trip:
            on_trip(rows[-1], keep)
        if verbose:
            r = rows[-1]
            print(f"  {part}/{trip}  pts={r['n_points']:>6}  F1 cls={r['cls_f1']:.3f} hmm={r['hmm_f1']:.3f}  "
                  f"LCS cls={r['cls_lcs_sim']:.3f} hmm={r['hmm_lcs_sim']:.3f}  "
                  f"breaks cls={r['cls_breaks']} hmm={r['hmm_breaks']}  ({r['diag_decode_s']}s)")
    if verbose and skipped:
        print(f"  ({len(skipped)} trips have no probabilities and were skipped: {skipped[:8]}{'...' if len(skipped) > 8 else ''})")
    return pd.DataFrame(rows)


def add_prob_kind(df, probs_are_oof):
    """Say how trustworthy each trip's probabilities are (train trips may be in-sample)."""
    kind = np.where(df["partition"] == "train",
                    "train: out-of-fold (declared)" if probs_are_oof else "train: UNVERIFIED (maybe in-sample)",
                    "student_test: held-out")
    df.insert(1, "prob_kind", kind)
    return df


def reason_hints(r):
    """Rule-of-thumb flags from the diagnostic columns (hints to investigate, not proof)."""
    h = []
    if r.diag_label_valid_pct < 50:
        h.append(f"reference route covers only {r.diag_label_valid_pct:.0f}% of the track")
    if r.diag_cand_miss_rate > 0.05:
        h.append(f"true arc missing from candidates at {100 * r.diag_cand_miss_rate:.0f}% of points")
    if r.diag_restarts > 0 or r.diag_max_gap_s > 60:
        h.append(f"time gap up to {r.diag_max_gap_s:.0f}s ({int(r.diag_restarts)} restarts)")
    if r.diag_unreach_frac > 0.5:
        h.append(f"{100 * r.diag_unreach_frac:.0f}% of candidate transitions unreachable")
    if r.cls_point_acc > 0.9:
        h.append("classifier-only already >90% accurate")
    if r.ref_breaks > 0:
        h.append(f"reference route itself has {int(r.ref_breaks)} break(s)")
    if r.hmm_precision < r.cls_precision - 0.02:
        h.append("HMM route has extra arcs (lower precision)")
    return "; ".join(h) or "no flag raised - inspect the map"


def summarize(df, label=None, wall_s=None, table=True):
    """Printable summary: mean/median per metric, improved/tied/worse, per-trip table, worst 10, timing."""
    L = [f"=== SUMMARY{f' [{label}]' if label else ''}: {len(df)} trips ==="]
    L.append(f"{'metric':<18}{'cls mean':>10}{'hmm mean':>10}{'d mean':>9} |{'cls med':>9}{'hmm med':>9}{'d med':>9}")
    for c in ["point_acc", "precision", "recall", "f1", "lcs_sim", "lcs_recall", "continuity"]:
        a, b = df[f"cls_{c}"], df[f"hmm_{c}"]
        L.append(f"{c:<18}{a.mean():>10.4f}{b.mean():>10.4f}{b.mean() - a.mean():>+9.4f} |"
                 f"{a.median():>9.4f}{b.median():>9.4f}{(b - a).median():>+9.4f}")
    L.append(f"breaks (total)    cls={int(df.cls_breaks.sum())}  hmm(raw)={int(df.hmm_raw_breaks.sum())}  "
             f"hmm(filled)={int(df.hmm_filled_breaks.sum())}   |  reference route breaks={int(df.ref_breaks.sum())}")
    L.append(f"continuity (mean) cls={df.cls_continuity.mean():.4f}  hmm raw={df.hmm_raw_continuity.mean():.4f}  "
             f"hmm filled={df.hmm_filled_continuity.mean():.4f}")
    L.append("\nHMM vs classifier-only, trips improved / tied / worse (tolerance 1e-9):")
    for c in ["point_acc", "f1", "lcs_sim"]:
        d = df[f"hmm_{c}"] - df[f"cls_{c}"]
        L.append(f"  {c:<10} improved={int((d > 1e-9).sum()):>3}  tied={int((d.abs() <= 1e-9).sum()):>3}  "
                 f"worse={int((d < -1e-9).sum()):>3}")

    if table:
        t = df[["partition", "trip_id", "n_points", "cls_point_acc", "hmm_point_acc", "cls_f1", "hmm_f1", "delta_f1",
                "cls_lcs_sim", "hmm_lcs_sim", "cls_breaks", "hmm_raw_breaks", "hmm_filled_breaks", "diag_label_valid_pct"]]
        L.append("\nPer-trip table (sorted by delta_f1; full columns are in the CSV):")
        L.append(t.sort_values("delta_f1").round(3).to_string(index=False))

    L.append("\n10 WORST trips for the HMM (lowest delta_f1) with hint flags:")
    for _, r in df.sort_values("delta_f1").head(10).iterrows():
        L.append(f"  {r.partition}/{r.trip_id}  pts={int(r.n_points)}  F1 {r.cls_f1:.3f}->{r.hmm_f1:.3f} "
                 f"({r.delta_f1:+.3f})  acc {r.cls_point_acc:.3f}->{r.hmm_point_acc:.3f}\n      hints: {reason_hints(r)}")

    big = df[df.n_points > 5000].sort_values("n_points", ascending=False)
    if len(big):
        L.append("\nTiming: trips over 5000 points (decode only / load+network+decode+metrics):")
        for _, r in big.iterrows():
            L.append(f"  {r.partition}/{r.trip_id}  pts={int(r.n_points):>6}  decode={r.diag_decode_s:>6.1f}s  total={r.diag_trip_total_s:>6.1f}s")
    else:
        L.append("\nTiming: no trip has more than 5000 points.")
    L.append(f"Sum of per-trip totals: {df.diag_trip_total_s.sum():.1f}s" + (f" | wall clock for the run: {wall_s:.1f}s" if wall_s else ""))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", required=True, help="csv / csv.gz / parquet: trip_id, point_idx, arc_id, prob")
    ap.add_argument("--partitions", nargs="+", default=list(DEV_PARTITIONS))
    ap.add_argument("--trips", nargs="*", help="restrict to these trip ids")
    ap.add_argument("--limit", type=int, help="only the first N trips")
    ap.add_argument("--config", default=None, help="decoder config json (default hmm/config.json)")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--cands-dir", default=str(CANDS_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None, help='name of the score source, e.g. "distance-only baseline"; stored in the CSV')
    ap.add_argument("--quiet", action="store_true", help="no per-trip progress lines")
    add_args(ap)
    a = ap.parse_args()

    synth = is_synthetic(a.probs)
    out = Path(a.out) if a.out else (SYNTH_DIR / "hmm_comparison_SYNTHETIC.csv" if synth
                                     else REPO / "results" / "hmm_comparison.csv")
    if synth and "hmm_synthetic" not in str(out):
        raise SystemExit("Refusing to write synthetic-probability output outside results/hmm_synthetic/.")
    if synth:
        print(SYNTH_BANNER)

    lp = load_from_args(a, a.partitions, a.data_root, a.cands_dir)          # validation + dtype fixing in one place
    trips = [t for t in lp.trips]
    if "train" in a.partitions and not a.probs_are_oof:
        print("NOTE: train trips are included but --probs-are-oof was not given: if the model was fitted on "
              "train trips, their metrics are in-sample (optimistic). Column prob_kind records this.")
    if a.trips:
        want = {norm_trip_id(t) for t in a.trips}
        trips = [t for t in trips if t[1] in want]
    if a.limit:
        trips = trips[:a.limit]
    if not Path(a.cands_dir).exists():
        print(f"NOTE: {a.cands_dir} not found -> pointwise accuracy skipped (needs P1's point labels).")

    if a.label:
        print(f"Score source: {a.label}")
    cfg = load_config(a.config)
    t_run = time.time()
    df = evaluate_many(lp.by_trip, trips, cfg, a.data_root, a.cands_dir, verbose=not a.quiet)
    wall = time.time() - t_run
    add_prob_kind(df, a.probs_are_oof)
    if a.label:
        df.insert(0, "score_source", a.label)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n" + summarize(df, a.label, wall))
    if synth:
        print(SYNTH_BANNER)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
