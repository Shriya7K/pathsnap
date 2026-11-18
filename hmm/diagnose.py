"""Diagnostics that explain WHERE and WHY the HMM does or does not help (used by hmm.run_all).

Per trip, from the same decode that hmm.evaluate did (no second decode):
  * point outcomes of classifier-only and of the HMM, over points with a reliable label (label_valid):
      correct   - predicted arc == labelled arc
      twin      - the same street in the OPPOSITE direction (from/to nodes swapped)
      adjacent  - directly connected to the labelled arc
      other     - anything else
      none      - no candidate for that point
  * effect of fill_path: route metrics with and without the inserted shortest-path arcs
  * how many extra route arcs (not in the reference) are reverse twins, decoded elsewhere, or inserted by fill_path
  * for HMM-wrong points: share where the chosen arc is closer to the GPS point than the labelled arc (needs P1's
    candidate distances; says whether a better distance-based score could even help)
The reference route / P1 labels are used here only to describe errors, never as decoder input.
"""
import numpy as np
import pandas as pd

from .evaluate import route_metrics

OUTCOMES = ["correct", "twin", "adjacent", "other", "none"]


def outcome(pred, true, arcs):
    if pred is None:
        return "none"
    if pred == true:
        return "correct"
    if arcs[pred][0] == arcs[true][1] and arcs[pred][1] == arcs[true][0]:
        return "twin"
    if arcs[pred][1] == arcs[true][0] or arcs[pred][0] == arcs[true][1]:
        return "adjacent"
    return "other"


def _shares(point_arcs, lab, arcs):
    valid = lab["label_valid"].to_numpy().astype(bool)
    true = lab["true_arc"].to_numpy()
    c = dict.fromkeys(OUTCOMES, 0)
    for i in np.nonzero(valid)[0]:
        c[outcome(point_arcs[i], int(true[i]), arcs)] += 1
    n = max(int(valid.sum()), 1)
    return {k: v / n for k, v in c.items()}, c


def diagnose_trip(partition, trip_id, keep, cands_dir):
    """keep = dict filled by hmm.evaluate.evaluate_trip(keep=...). Returns one row (dict)."""
    res, net, arcs, ref = keep["res"], keep["net"], keep["arcs"], keep["ref"]
    refl = ref.tolist()
    row = {"partition": partition, "trip_id": trip_id}

    # fill_path effect (route metrics on the two variants)
    mr, mf = route_metrics(res.raw_route, refl, net), route_metrics(res.filled_route, refl, net)
    for k in ("precision", "recall", "f1", "lcs_sim", "breaks"):
        row[f"raw_{k}"], row[f"filled_{k}"] = mr[k], mf[k]
    row["fill_f1_gain"] = mf["f1"] - mr["f1"]

    # extra route arcs: where do they come from?
    refset = set(refl)
    twins = {(int(arcs[a][1]), int(arcs[a][0])) for a in refset}
    decoded = {a for a in res.point_arcs if a is not None}
    extra = [a for a in res.route if a not in refset]
    e_twin = sum(1 for a in extra if a in decoded and (int(arcs[a][0]), int(arcs[a][1])) in twins)
    e_fill = sum(1 for a in extra if a not in decoded)
    row.update({"route_arcs": len(res.route), "ref_arcs": len(refset), "extra_arcs": len(extra),
                "extra_twin_of_ref": e_twin, "extra_from_fill": e_fill,
                "extra_decoded_other": len(extra) - e_twin - e_fill})

    # point outcomes (needs P1's point labels)
    f = f"{cands_dir}/{partition}_{trip_id}_points.csv.gz"
    try:
        lab = pd.read_csv(f)
    except FileNotFoundError:
        return row
    for name, pa in (("cls", keep["cls_points"]), ("hmm", res.point_arcs)):
        sh, _ = _shares(pa, lab, arcs)
        for k in OUTCOMES:
            row[f"{name}_{k}"] = sh[k]
    # for HMM-wrong labelled points: is the chosen arc nearer to the GPS point than the labelled arc?
    try:
        c = pd.read_csv(f"{cands_dir}/{partition}_{trip_id}_cands.csv.gz", usecols=["point_idx", "arc_id", "dist"])
        d = {(int(i), int(a)): x for i, a, x in zip(c["point_idx"], c["arc_id"], c["dist"])}
        valid = lab["label_valid"].to_numpy().astype(bool)
        true = lab["true_arc"].to_numpy()
        near = tot = 0
        for i in np.nonzero(valid)[0]:
            p = res.point_arcs[i]
            if p is None or p == int(true[i]):
                continue
            a, b = d.get((int(i), int(true[i]))), d.get((int(i), int(p)))
            if a is not None and b is not None:
                tot += 1
                near += b < a
        row["hmm_wrong_chosen_nearer_share"] = near / tot if tot else np.nan
    except FileNotFoundError:
        pass
    return row


def summarize_diagnostics(d):
    """Printable aggregate of the per-trip diagnostic rows (DataFrame)."""
    L = ["=== DIAGNOSE (reference / labels used only to describe errors) ==="]
    if "cls_correct" in d:
        L.append("point outcomes, mean over trips (label_valid points):   correct   twin  adjacent  other   none")
        for name in ("cls", "hmm"):
            L.append(f"  {('classifier-only' if name == 'cls' else 'HMM'):<16}"
                     + "".join(f"{100 * d[f'{name}_{k}'].mean():>9.1f}%" for k in OUTCOMES))
    L.append("fill_path effect (route metrics, mean over trips):          raw   filled")
    for k in ("precision", "recall", "f1", "lcs_sim"):
        L.append(f"  {k:<14}{d[f'raw_{k}'].mean():>31.4f}{d[f'filled_{k}'].mean():>9.4f}")
    L.append(f"  breaks (total) {int(d.raw_breaks.sum()):>29d}{int(d.filled_breaks.sum()):>9d}")
    L.append(f"  trips where filling raises F1: {int((d.fill_f1_gain > 1e-9).sum())}   lowers F1: "
             f"{int((d.fill_f1_gain < -1e-9).sum())}   unchanged: {int((d.fill_f1_gain.abs() <= 1e-9).sum())}")
    ex = d.extra_arcs.sum()
    if ex:
        L.append(f"route arcs not in the reference: {int(ex)} = reverse twins {int(d.extra_twin_of_ref.sum())} + "
                 f"fill_path insertions {int(d.extra_from_fill.sum())} + other decoded {int(d.extra_decoded_other.sum())}")
    if "hmm_wrong_chosen_nearer_share" in d:
        L.append(f"HMM-wrong points whose chosen arc is nearer to the GPS point than the labelled arc "
                 f"(median over trips): {100 * d.hmm_wrong_chosen_nearer_share.median():.0f}%")
    return "\n".join(L)
