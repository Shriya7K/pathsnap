"""Viterbi tests: hand-computed example, brute force, road-network behaviour, edge cases."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmm.graph import TripNetwork  # noqa: E402
from hmm.viterbi import decode, decode_detailed, load_config, viterbi  # noqa: E402

LON0, LAT0 = 77.5, 13.0


# ------------------------------------------------------------------ pure Viterbi, by hand
def test_viterbi_hand_computed():
    """3 steps, 2 states, log-scores chosen by hand.
    e0=[0,-1]  e1=[-2,0]  e2=[0,-3];  stay = 0, switch = -1.5
    Per-step argmax of the emissions is (0,1,0): total 0 -1.5 +0 -1.5 +0 = -3.0
    All 8 paths: 000:-2.0  001:-6.5  010:-3.0  011:-4.5  100:-5.5  101:-9.0  110:-2.5  111:-4.0
    -> the best path is (0,0,0) = -2.0, NOT the greedy one: Viterbi is a global optimum.
    Recurrence by hand: delta0=[0,-1]; delta1=[max(0+0,-1-1.5)-2, max(0-1.5,-1+0)+0]=[-2,-1];
    delta2=[max(-2+0,-1-1.5)+0, max(-2-1.5,-1+0)-3]=[-2,-4] -> end in state 0 (-2)."""
    e = [np.array([0., -1.]), np.array([-2., 0.]), np.array([0., -3.])]
    T = np.array([[0., -1.5], [-1.5, 0.]])
    assert viterbi(e, lambda t: T) == [0, 0, 0]
    assert [int(x.argmax()) for x in e] == [0, 1, 0]        # greedy differs


def test_viterbi_matches_brute_force_random_varying_state_counts():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ks = rng.integers(1, 5, size=rng.integers(1, 6))     # number of states varies per step
        e = [rng.normal(size=k) for k in ks]
        Ts = [None] + [rng.normal(size=(ks[i - 1], ks[i])) for i in range(1, len(ks))]
        got = viterbi(e, lambda t: Ts[t])

        def score(p):
            return sum(e[i][s] for i, s in enumerate(p)) + sum(Ts[i][p[i - 1], p[i]] for i in range(1, len(p)))
        best = max(itertools.product(*[range(k) for k in ks]), key=score)
        assert abs(score(got) - score(best)) < 1e-9


def test_viterbi_empty_and_single():
    assert viterbi([], lambda t: None) == []
    assert viterbi([np.array([-1., -0.1, -3.])], lambda t: None) == [1]


# ------------------------------------------------------------------ tiny road network
def two_parallel_streets():
    """Street 1: n0-n1-n2-n3 eastward (arcs 0,1,2). Street 2: m0..m3 ~20 m north (arcs 3,4,5).
    The streets are NOT connected to each other. Each arc is ~100 m long."""
    dx = 0.0009                                              # ~100 m of longitude at lat 13
    s1 = [[LON0 + i * dx, LAT0] for i in range(4)]
    s2 = [[LON0 + i * dx, LAT0 + 0.00018] for i in range(4)]
    nodes = np.array(s1 + s2)
    arcs = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]])
    return nodes, arcs


def make_gps(nodes, arcs, ids, dt=10.0, t0=0.0):
    """One noise-free GPS point at the midpoint of each listed arc."""
    pts = [(nodes[arcs[a][0]] + nodes[arcs[a][1]]) / 2 for a in ids]
    return pd.DataFrame({"lon": [p[0] for p in pts], "lat": [p[1] for p in pts],
                         "t": [t0 + i * dt for i in range(len(ids))]})


def probs_df(rows):
    """rows: list of (point_idx, arc_id, prob)."""
    return pd.DataFrame(rows, columns=["point_idx", "arc_id", "prob"]).assign(trip_id="00000001")


def test_hmm_repairs_wrong_parallel_street():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 1, 2])
    # classifier prefers the parallel street's arc 4 at point 1
    p = probs_df([(0, 0, .8), (0, 3, .2), (1, 1, .4), (1, 4, .6), (2, 2, .8), (2, 5, .2)])
    cls = p.sort_values("prob", ascending=False).drop_duplicates("point_idx").sort_values("point_idx").arc_id.tolist()
    assert cls == [0, 4, 2]                                  # classifier-only jumps to the other street
    point_arcs, route = decode(gps, nodes, arcs, p, config={"fill_path": False})
    assert point_arcs == [0, 1, 2]                           # arc 4 unreachable from arc 0 -> rejected
    assert route == [0, 1, 2]


def test_same_arc_stays_and_duplicates_removed():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 0, 0, 1], dt=1.0)
    gps["lon"] = gps["lon"] + [0, 1e-6, 2e-6, 0]             # tiny jitter
    p = probs_df([(i, a, .6 if a == 0 else .4) for i in range(3) for a in (0, 1)]
                 + [(3, 0, .1), (3, 1, .9)])
    point_arcs, route = decode(gps, nodes, arcs, p)
    assert point_arcs == [0, 0, 0, 1]
    assert route == [0, 1]                                   # consecutive duplicates removed


# ------------------------------------------------------------------ edge cases
def test_empty_candidates_everywhere():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 1, 2])
    empty = pd.DataFrame({"trip_id": [], "point_idx": [], "arc_id": [], "prob": []})
    assert decode(gps, nodes, arcs, empty) == ([None, None, None], [])


def test_point_without_candidates_is_skipped_chain_continues():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 1, 2])
    p = probs_df([(0, 0, .9), (0, 3, .1), (2, 2, .9), (2, 5, .1)])      # point 1 has none
    point_arcs, route = decode(gps, nodes, arcs, p)
    assert point_arcs == [0, None, 2]
    assert route == [0, 1, 2]                                # arc 1 filled in by shortest path


def test_invalid_rows_are_ignored_not_crashing():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 1])
    p = probs_df([(0, 0, .9), (0, 999, .5), (0, -1, .5), (7, 0, .5), (1, 1, np.nan), (1, 1, .8), (1, 1, .2)])
    point_arcs, route = decode(gps, nodes, arcs, p)
    assert point_arcs == [0, 1] and route == [0, 1]


def test_huge_time_gap_restarts_and_never_crashes():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 1, 2], dt=10.0)
    gps.loc[2, "t"] = 1e6                                    # a ~12-day gap before the last point
    p = probs_df([(0, 0, .9), (0, 3, .1), (1, 1, .9), (1, 4, .1), (2, 2, .9), (2, 5, .1)])
    r = decode_detailed(gps, nodes, arcs, p, {"fill_path": True})
    assert r.stats["n_restarts"] == 1
    assert r.point_arcs == [0, 1, 2]


def test_gap_widens_allowed_path_distance():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 2], dt=1.0)              # midpoints of arcs 0 and 2: 200 m apart
    p = probs_df([(0, 0, .5), (1, 2, .5)])
    cfg = {"max_path_m": 100, "gap_scale_mps": 10}
    short = decode_detailed(gps, nodes, arcs, p, cfg)
    gps2 = gps.copy()
    gps2.loc[1, "t"] = 120.0
    long_ = decode_detailed(gps2, nodes, arcs, p, cfg)
    assert short.stats["n_unreachable"] == 1                 # dt=1 s: bound ~110 m < 200 m route
    assert long_.stats["n_unreachable"] == 0                 # dt=120 s: bound ~1300 m


def test_never_dead_ends_when_everything_is_unreachable():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 4])                      # arcs on different, unconnected streets
    p = probs_df([(0, 0, .9), (1, 4, .9)])
    point_arcs, route = decode(gps, nodes, arcs, p)
    assert point_arcs == [0, 4]                              # penalised, not impossible


# ------------------------------------------------------------------ honesty about continuity
def test_filled_route_only_claims_connections_the_graph_supports():
    nodes, arcs = two_parallel_streets()
    net = TripNetwork(nodes, arcs, LON0, LAT0)
    gps = make_gps(nodes, arcs, [0, 2, 4, 5], dt=10.0)       # street 1, then a jump to street 2
    p = probs_df([(0, 0, .9), (1, 2, .9), (2, 4, .9), (3, 5, .9)])
    r = decode_detailed(gps, nodes, arcs, p, {"fill_path": True}, network=net)
    assert r.route == [0, 1, 2, 4, 5]                        # arc 1 inserted (a real path exists) ...
    breaks = [(a, b) for a, b in zip(r.route, r.route[1:]) if not net.connected(a, b)]
    assert breaks == [(2, 4)]                                # ... the street change stays an honest break
    raw_breaks = [(a, b) for a, b in zip(r.raw_route, r.raw_route[1:]) if not net.connected(a, b)]
    assert (0, 2) in raw_breaks and len(raw_breaks) == 2     # unfilled route has the extra break


def test_fill_flag_switches_returned_route():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 2], dt=10.0)
    p = probs_df([(0, 0, .9), (1, 2, .9)])
    assert decode(gps, nodes, arcs, p, config={"fill_path": True})[1] == [0, 1, 2]
    assert decode(gps, nodes, arcs, p, config={"fill_path": False})[1] == [0, 2]


# ------------------------------------------------------------------ interface contract
def test_output_format_matches_contract():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 1, 2])
    p = probs_df([(0, 0, .9), (1, 1, .9), (1, 4, .1), (2, 2, .9)])
    out = decode(gps, nodes, arcs, p)
    assert isinstance(out, tuple) and len(out) == 2
    point_arcs, route = out
    assert isinstance(point_arcs, list) and len(point_arcs) == len(gps)
    assert all(a is None or type(a) is int for a in point_arcs)       # python ints (JSON-safe), not numpy
    assert isinstance(route, list) and all(type(a) is int for a in route)
    assert all(a != b for a, b in zip(route, route[1:]))              # no consecutive duplicates
    assert all(0 <= a < len(arcs) for a in route)


def test_trip_id_filter_uses_only_that_trips_rows():
    nodes, arcs = two_parallel_streets()
    gps = make_gps(nodes, arcs, [0, 1])
    p1 = probs_df([(0, 0, .9), (1, 1, .9)])
    p2 = probs_df([(0, 3, .9), (1, 4, .9)]).assign(trip_id="00000002")
    both = pd.concat([p1, p2])
    assert decode(gps, nodes, arcs, both, trip_id=1)[0] == [0, 1]     # int id is zero-padded to match
    assert decode(gps, nodes, arcs, both, trip_id="00000002")[0] == [3, 4]


def test_config_json_has_all_decoder_parameters():
    cfg = load_config()
    for k in ("emission_floor", "temperature", "beta", "switch_penalty", "unreach_penalty",
              "max_path_m", "gap_scale_mps", "path_cap_m", "restart_gap_s", "fill_path", "fill_max_m"):
        assert k in cfg
