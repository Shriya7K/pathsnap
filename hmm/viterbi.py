"""HMM / Viterbi map-matching decoder (Person 3).

Turns classifier probabilities per (GPS point, candidate arc) into one connected route.
See hmm/README.md for the model; short version:

  hidden state at point i   = which candidate arc the vehicle is on
  emission  e_i(k)          = log(max(prob_ik, floor)) / temperature
  transition T_i(j -> k)    = -|route_dist - gps_dist| / beta_eff  - switch_penalty*[arc changes]
                              (floored at -unreach_penalty; unreachable pairs get -unreach_penalty)
  Viterbi:  delta_i(k) = e_i(k) + max_j [ delta_{i-1}(j) + T_i(j,k) ],  psi_i(k) = argmax_j

Public API (the contract teammates code against):
  decode(gps, nodes, arcs, probs, config=None, trip_id=None) -> (point_arcs, route)
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import TripNetwork, build_network, norm_trip_id

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path=None, **overrides):
    """Load decoder parameters from hmm/config.json (keys starting with '_' are comments)."""
    with open(path or CONFIG_PATH) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------- core Viterbi
def viterbi(emissions, trans_fn):
    """Generic log-space Viterbi over a chain of steps with varying numbers of states.

    emissions : list of 1-D arrays; emissions[t][k] = log-emission of state k at step t
    trans_fn  : trans_fn(t) -> matrix (K_{t-1} x K_t) of log-transition scores into step t (t>=1)
    Returns the best state index at every step (list[int]).
    Pure function (no road network inside) so it can be unit-tested by hand."""
    n = len(emissions)
    if n == 0:
        return []
    delta = np.asarray(emissions[0], dtype=float)
    back = [None]
    for t in range(1, n):
        scores = delta[:, None] + trans_fn(t)        # scores[j,k] = delta_{t-1}(j) + T(j->k)
        psi = scores.argmax(axis=0)                  # best predecessor of every state k
        delta = scores[psi, np.arange(scores.shape[1])] + emissions[t]
        back.append(psi)
    path = [int(delta.argmax())]                     # best final state, then follow back-pointers
    for t in range(n - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return path[::-1]


# --------------------------------------------------------------------------- result container
@dataclass
class DecodeResult:
    point_arcs: list                  # arc id (python int) or None per GPS point
    raw_route: list                   # decoded arcs, consecutive duplicates removed
    filled_route: list                # raw_route with shortest-path arcs inserted where possible
    route: list                       # the one returned by decode(): filled if cfg['fill_path'] else raw
    stats: dict = field(default_factory=dict)


def dedupe(seq):
    """Remove consecutive duplicates, keep order."""
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(int(x))
    return out


# --------------------------------------------------------------------------- decoder
def _candidate_table(probs, n_points, n_arcs, trip_id):
    """Clean the classifier output: filter to this trip, drop invalid rows, one row per
    (point, arc) keeping the highest prob. Returns DataFrame sorted by point_idx."""
    p = probs
    if trip_id is not None and "trip_id" in p.columns:
        want = norm_trip_id(trip_id)
        p = p[p["trip_id"].map(norm_trip_id) == want]
    p = p[["point_idx", "arc_id", "prob"]].copy()
    p = p.replace([np.inf, -np.inf], np.nan).dropna()
    p["point_idx"] = p["point_idx"].astype(np.int64)
    p["arc_id"] = p["arc_id"].astype(np.int64)
    p = p[(p.point_idx >= 0) & (p.point_idx < n_points) & (p.arc_id >= 0) & (p.arc_id < n_arcs)]
    p = p.sort_values(["point_idx", "arc_id", "prob"], ascending=[True, True, False])
    p = p.drop_duplicates(["point_idx", "arc_id"], keep="first")
    return p.reset_index(drop=True)


def decode_detailed(gps, nodes, arcs, probs, config=None, trip_id=None, network=None,
                    cache_key=None):
    """Full decoder. gps: DataFrame(lon, lat, t); nodes (N,2); arcs (M,2);
    probs: DataFrame(trip_id, point_idx, arc_id, prob). Returns DecodeResult."""
    cfg = load_config() if config is None else {**load_config(), **config}
    n = len(gps)
    net = network or build_network(gps, nodes, arcs, cache_key=cache_key)
    t = np.asarray(gps["t"], dtype=float)
    xy = net.project(gps["lon"].to_numpy(), gps["lat"].to_numpy())

    cand = _candidate_table(probs, n, net.n_arcs, trip_id)
    stats = {"n_points": n, "n_candidate_rows": len(cand), "n_empty_points": 0,
             "n_restarts": 0, "n_transitions": 0, "n_unreachable": 0}

    point_arcs = [None] * n
    if len(cand) == 0 or n == 0:
        stats["n_empty_points"] = n
        return DecodeResult(point_arcs, [], [], [], stats)

    # --- per-candidate quantities, computed for all rows at once
    pt = cand["point_idx"].to_numpy()
    arc = cand["arc_id"].to_numpy()
    emit = np.log(np.maximum(cand["prob"].to_numpy(dtype=float), cfg["emission_floor"])) / cfg["temperature"]
    off = net.project_on_arc(xy[pt], arc)                       # where along the arc the point lands

    # rows of each point = slice [start[i], start[i+1])
    start = np.searchsorted(pt, np.arange(n + 1))
    active = [i for i in range(n) if start[i + 1] > start[i]]    # points that have candidates
    stats["n_empty_points"] = n - len(active)

    # --- split the active points into independent segments at very large time gaps (restart)
    segments, cur = [], [active[0]]
    for a, b in zip(active[:-1], active[1:]):
        dt = t[b] - t[a]
        if np.isfinite(dt) and dt > cfg["restart_gap_s"]:
            segments.append(cur)
            cur = []
            stats["n_restarts"] += 1
        cur.append(b)
    segments.append(cur)

    from_node, arc_len = net.from_node, net.arc_len

    def transition(p, q):
        """Log-transition matrix (K_p x K_q) between consecutive active points p -> q."""
        sp, sq = slice(start[p], start[p + 1]), slice(start[q], start[q + 1])
        A, OA, B, OB = arc[sp], off[sp], arc[sq], off[sq]
        dt = t[q] - t[p]
        dt = 1.0 if (not np.isfinite(dt) or dt <= 0) else dt
        d_gps = float(np.hypot(*(xy[q] - xy[p])))
        # gap handling: allowed path length and tolerance both grow with the time gap
        limit = min(cfg["path_cap_m"], cfg["max_path_m"] + cfg["gap_scale_mps"] * dt)
        beta = cfg["beta"] + cfg["gap_scale_mps"] * max(dt - 1.0, 0.0)
        unreach = cfg["unreach_penalty"]
        fB = from_node[B].tolist()
        T = np.full((len(A), len(B)), -unreach)
        for a in range(len(A)):
            reach = net.reach(int(net.to_node[A[a]]), limit)
            D = np.fromiter((reach.get(f, math.inf) for f in fB), dtype=float, count=len(fB))
            d_route = (arc_len[A[a]] - OA[a]) + D + OB        # rest of A + road between + into B
            same = B == A[a]
            d_route = np.where(same, np.abs(OB - OA[a]), d_route)   # same arc: just the offset change
            ok = d_route <= limit                              # finite and within the bound
            score = -np.abs(d_route - d_gps) / beta - cfg["switch_penalty"] * (~same)
            T[a] = np.where(ok, np.maximum(score, -unreach), -unreach)
            stats["n_unreachable"] += int((~ok).sum())
        stats["n_transitions"] += T.size
        return T

    # --- Viterbi per segment
    for seg in segments:
        em = [emit[start[i]:start[i + 1]] for i in seg]
        best = viterbi(em, lambda s: transition(seg[s - 1], seg[s]))
        for i, k in zip(seg, best):
            point_arcs[i] = int(arc[start[i] + k])

    raw = dedupe(a for a in point_arcs if a is not None)
    filled = fill_route(raw, net, cfg["fill_max_m"])
    stats["unreach_frac"] = stats["n_unreachable"] / max(stats["n_transitions"], 1)
    return DecodeResult(point_arcs, raw, filled, filled if cfg["fill_path"] else raw, stats)


def fill_route(route, net, max_m):
    """Insert shortest-path arcs between consecutive arcs that are not directly connected.
    If no path exists within max_m, the break is left as it is (we never fake continuity)."""
    out = []
    for a in route:
        if out and not net.connected(out[-1], a):
            mid = net.shortest_arc_path(out[-1], a, max_m)
            if mid:
                out.extend(int(x) for x in mid)
        out.append(int(a))
    return dedupe(out)


def decode(gps, nodes, arcs, probs, config=None, trip_id=None, network=None, cache_key=None):
    """THE interface used by the API.
    gps: DataFrame(lon, lat, t); nodes: ndarray (N,2); arcs: ndarray (M,2);
    probs: DataFrame(trip_id, point_idx, arc_id, prob).
    Returns (point_arcs, route): arc id (or None) per GPS point, and the arc-id route in
    travel order with consecutive duplicates removed."""
    r = decode_detailed(gps, nodes, arcs, probs, config, trip_id, network, cache_key)
    return r.point_arcs, r.route
