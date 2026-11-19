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


# ===========================================================================
# Object-Oriented HMM Decoder & Viterbi Trellis (Full Pipeline Interface)
# ===========================================================================
"""Hidden Markov Model (HMM) and Viterbi Trellis Decoder for PathSnap.

Implements the Newson & Krumm (2009) global sequence map-matching algorithm,
fusing Person 2's ML classifier emission probabilities with road-network
topological transition probabilities.
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
from pyproj import Transformer


@dataclass
class MatchResult:
    trip_id: str
    point_indices: List[int]
    matched_arc_ids: List[int]
    matched_coords_wgs84: List[Tuple[float, float]]  # (latitude, longitude)
    raw_coords_wgs84: List[Tuple[float, float]]      # (latitude, longitude)
    confidences: List[float]
    unique_route_arcs: List[int]
    total_distance_m: float
    accuracy_against_truth: Optional[float] = None


class HMMMapMatcher:
    """Hidden Markov Model Map Matcher with Viterbi Trellis Decoding."""

    def __init__(
        self,
        beta: float = 10.0,            # Transition distance scale parameter (meters)
        sigma_z: float = 15.0,         # Geometric GPS measurement noise (meters)
        max_dist_gap: float = 1500.0,  # Max allowable network routing distance
        temporal_gap_thresh: float = 60.0, # Break in sequential HMM if delta_t > threshold
    ):
        self.beta = beta
        self.sigma_z = sigma_z
        self.max_dist_gap = max_dist_gap
        self.temporal_gap_thresh = temporal_gap_thresh

    def build_network_graph(self, nodes_xy: np.ndarray, a_from: np.ndarray, a_to: np.ndarray) -> nx.DiGraph:
        """Construct directed road network graph with metric edge weights."""
        G = nx.DiGraph()
        n_nodes = len(nodes_xy)
        for i in range(n_nodes):
            G.add_node(i, x=nodes_xy[i, 0], y=nodes_xy[i, 1])

        dx = nodes_xy[a_to, 0] - nodes_xy[a_from, 0]
        dy = nodes_xy[a_to, 1] - nodes_xy[a_from, 1]
        lengths = np.hypot(dx, dy)

        for arc_id in range(len(a_from)):
            u, v, w = int(a_from[arc_id]), int(a_to[arc_id]), float(lengths[arc_id])
            G.add_edge(u, v, arc_id=arc_id, weight=w)

        return G

    def match(
        self,
        trip_id: str,
        gps_xy: np.ndarray,           # (N, 2) UTM metric GPS coordinates
        gps_ts: np.ndarray,           # (N,) timestamps
        nodes_xy: np.ndarray,         # (M, 2) UTM metric node coordinates
        a_from: np.ndarray,           # (A,) arc start node indices
        a_to: np.ndarray,             # (A,) arc end node indices
        candidates_df: pd.DataFrame,  # Candidate rows with point_idx, arc_id, predicted_probability, perpendicular_distance
        epsg: int,
    ) -> MatchResult:
        """Execute Viterbi sequence decoding over candidate trellis."""
        n_pts = len(gps_xy)
        if n_pts == 0 or len(candidates_df) == 0:
            return MatchResult(
                trip_id=trip_id,
                point_indices=[],
                matched_arc_ids=[],
                matched_coords_wgs84=[],
                raw_coords_wgs84=[],
                confidences=[],
                unique_route_arcs=[],
                total_distance_m=0.0,
            )

        # 1. Build road network graph
        G = self.build_network_graph(nodes_xy, a_from, a_to)
        shortest_path_cache: Dict[Tuple[int, int], float] = {}

        def get_shortest_path_length(u: int, v: int) -> float:
            if u == v:
                return 0.0
            pair = (u, v)
            if pair in shortest_path_cache:
                return shortest_path_cache[pair]
            try:
                length = nx.shortest_path_length(G, source=u, target=v, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                length = 1e6
            shortest_path_cache[pair] = length
            return length

        # 2. Organize candidates by observation index
        dist_col = "perpendicular_distance" if "perpendicular_distance" in candidates_df.columns else "dist"
        prob_col = "predicted_probability" if "predicted_probability" in candidates_df.columns else "prob"
        if prob_col not in candidates_df.columns:
            # Fallback to inverse distance if ML probability is not present
            candidates_df[prob_col] = np.exp(-0.5 * (candidates_df[dist_col] / self.sigma_z) ** 2)

        grouped_cands = {}
        for p_idx, grp in candidates_df.groupby("point_idx"):
            grouped_cands[int(p_idx)] = grp.to_dict("records")

        # Handle missing candidate points by assigning fallback closest arc or dummy
        all_point_indices = sorted(list(range(n_pts)))
        obs_candidates: List[List[Dict]] = []
        for i in all_point_indices:
            cands = grouped_cands.get(i, [])
            if not cands:
                # If no candidates within R, use nearest arc from previous observation or default
                if obs_candidates and obs_candidates[-1]:
                    prev_arc = obs_candidates[-1][0]["arc_id"]
                    cands = [{"arc_id": prev_arc, dist_col: 50.0, prob_col: 0.1, "point_idx": i}]
                else:
                    cands = [{"arc_id": 0, dist_col: 100.0, prob_col: 0.1, "point_idx": i}]
            obs_candidates.append(cands)

        # 3. Viterbi Trellis Forward Pass
        T = len(obs_candidates)
        # V[t][j] = log probability of best path to candidate j at step t
        V: List[np.ndarray] = [None] * T
        backpointers: List[np.ndarray] = [None] * T

        # Step 0: Initial emissions
        cands_0 = obs_candidates[0]
        v0 = np.zeros(len(cands_0))
        for j, c in enumerate(cands_0):
            p_ml = max(float(c.get(prob_col, 0.5)), 1e-4)
            d_perp = float(c.get(dist_col, 20.0))
            # Joint emission: log(P_ML) - (d^2 / 2*sigma^2)
            log_emiss = np.log(p_ml) - 0.5 * ((d_perp / self.sigma_z) ** 2)
            v0[j] = log_emiss
        V[0] = v0

        # Steps 1 to T-1
        for t in range(1, T):
            cands_prev = obs_candidates[t - 1]
            cands_curr = obs_candidates[t]
            K_prev = len(cands_prev)
            K_curr = len(cands_curr)

            # Euclidean GPS distance
            p_prev = gps_xy[t - 1]
            p_curr = gps_xy[t]
            d_gps = float(np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]))

            # Temporal check: break sequential connection if temporal gap is huge
            dt = float(gps_ts[t] - gps_ts[t - 1]) if len(gps_ts) > t else 1.0
            is_break = dt > self.temporal_gap_thresh

            # Fast single-source Dijkstra from unique prior end-nodes
            cutoff_dist = max(d_gps * 2.5, 300.0)
            u_nodes = set(int(a_to[int(c["arc_id"])]) for c in cands_prev)
            dist_maps: Dict[int, Dict[int, float]] = {}
            for u in u_nodes:
                try:
                    dist_maps[u] = nx.single_source_dijkstra_path_length(G, u, cutoff=cutoff_dist, weight="weight")
                except Exception:
                    dist_maps[u] = {}

            vt = np.zeros(K_curr)
            bp = np.zeros(K_curr, dtype=int)

            for j in range(K_curr):
                arc_curr = int(cands_curr[j]["arc_id"])
                p_ml = max(float(cands_curr[j].get(prob_col, 0.5)), 1e-4)
                d_perp = float(cands_curr[j].get(dist_col, 20.0))
                log_emiss = np.log(p_ml) - 0.5 * ((d_perp / self.sigma_z) ** 2)

                best_log_prob = -1e9
                best_prev_idx = 0
                v = int(a_from[arc_curr])

                for i in range(K_prev):
                    arc_prev = int(cands_prev[i]["arc_id"])

                    if is_break:
                        log_trans = 0.0
                    else:
                        if arc_prev == arc_curr:
                            d_net = d_gps
                        else:
                            u = int(a_to[arc_prev])
                            if u == v:
                                d_net = d_gps * 1.01  # Direct topological connection
                            else:
                                route_dist = dist_maps.get(u, {}).get(v, None)
                                d_net = route_dist if route_dist is not None else (d_gps + 200.0)

                        diff = abs(d_net - d_gps)
                        log_trans = - (diff / self.beta)

                    total_score = V[t - 1][i] + log_trans
                    if total_score > best_log_prob:
                        best_log_prob = total_score
                        best_prev_idx = i

                vt[j] = best_log_prob + log_emiss
                bp[j] = best_prev_idx

            # Prevent numerical underflow by normalizing log probabilities
            vt -= np.max(vt)
            V[t] = vt
            backpointers[t] = bp

        # 4. Viterbi Backtrack
        matched_arcs: List[int] = [0] * T
        confidences: List[float] = [0.0] * T

        # Optimal ending state
        best_end_idx = int(np.argmax(V[-1]))
        curr_idx = best_end_idx

        for t in range(T - 1, -1, -1):
            cand = obs_candidates[t][curr_idx]
            matched_arcs[t] = int(cand["arc_id"])
            # Posterior confidence across candidates at step t
            probs = np.exp(V[t] - np.max(V[t]))
            probs /= (probs.sum() + 1e-9)
            confidences[t] = round(float(probs[curr_idx]), 4)
            if t > 0:
                curr_idx = backpointers[t][curr_idx]

        # 5. Coordinate Projection & Snapping
        tr_inv = Transformer.from_crs(epsg, 4326, always_xy=True)
        raw_lons, raw_lats = tr_inv.transform(gps_xy[:, 0], gps_xy[:, 1])
        raw_coords = list(zip(raw_lats.tolist(), raw_lons.tolist()))

        # Compute snapped coordinates along matched arcs
        matched_coords = []
        for t in range(T):
            arc = matched_arcs[t]
            u, v = a_from[arc], a_to[arc]
            x1, y1 = nodes_xy[u]
            x2, y2 = nodes_xy[v]
            px, py = gps_xy[t]
            # Orthogonal projection onto road segment
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            if L2 > 0:
                s = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
                snap_x = x1 + s * dx
                snap_y = y1 + s * dy
            else:
                snap_x, snap_y = x1, y1

            snap_lon, snap_lat = tr_inv.transform(snap_x, snap_y)
            matched_coords.append((float(snap_lat), float(snap_lon)))

        # Unique route sequence (collapsed consecutive duplicates)
        unique_arcs = []
        for a in matched_arcs:
            if not unique_arcs or unique_arcs[-1] != a:
                unique_arcs.append(a)

        # Total snapped route distance in meters
        total_dist = 0.0
        for i in range(1, len(matched_coords)):
            # Local metric distance
            dx = gps_xy[i, 0] - gps_xy[i - 1, 0]
            dy = gps_xy[i, 1] - gps_xy[i - 1, 1]
            total_dist += float(np.hypot(dx, dy))

        return MatchResult(
            trip_id=trip_id,
            point_indices=all_point_indices,
            matched_arc_ids=matched_arcs,
            matched_coords_wgs84=matched_coords,
            raw_coords_wgs84=raw_coords,
            confidences=confidences,
            unique_route_arcs=unique_arcs,
            total_distance_m=round(total_dist, 2),
        )
