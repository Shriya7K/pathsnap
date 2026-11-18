"""Graph tests on a tiny synthetic grid (hand-checkable)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmm import graph  # noqa: E402
from hmm.graph import TripNetwork  # noqa: E402

LON0, LAT0, STEP = 77.5, 13.0, 0.001          # 0.001 deg of latitude ~ 110.6 m


def make_grid(n=3, lon0=LON0):
    """n x n nodes, bidirectional arcs between 4-neighbours. node id = r*n + c."""
    nodes = np.array([[lon0 + c * STEP, LAT0 + r * STEP] for r in range(n) for c in range(n)])
    arcs = []
    for r in range(n):
        for c in range(n):
            u = r * n + c
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < n and cc < n:
                    v = rr * n + cc
                    arcs += [(u, v), (v, u)]
    return nodes, np.array(arcs)


def arc_id(arcs, u, v):
    return [i for i, (a, b) in enumerate(arcs) if (a, b) == (u, v)][0]


def test_arc_counts_and_lengths_in_metres():
    nodes, arcs = make_grid(3)
    net = TripNetwork(nodes, arcs, LON0, LAT0)
    assert net.n_nodes == 9 and net.n_arcs == 24            # 12 undirected edges x 2 directions
    north = arc_id(arcs, 0, 3)
    assert 108 < net.arc_len[north] < 113                   # 0.001 deg lat ~ 110.6 m, NOT 0.001
    assert net.epsg == 32643                                # UTM zone 43N


def test_adjacency_rule_to_equals_from():
    nodes, arcs = make_grid(3)
    net = TripNetwork(nodes, arcs, LON0, LAT0)
    for a in range(net.n_arcs):
        succ = set(net.successors(a).tolist())
        expected = {b for b in range(net.n_arcs) if arcs[b][0] == arcs[a][1]}
        assert succ == expected
        assert all(net.connected(a, b) for b in succ)
    a01 = arc_id(arcs, 0, 1)
    assert len(net.successors(a01)) == 3                    # node 1 (edge node) has 3 neighbours
    assert net.connected(a01, a01)                          # same arc counts as connected
    assert not net.connected(a01, arc_id(arcs, 7, 8))       # far away arc is not


def test_bounded_dijkstra_respects_limit():
    nodes, arcs = make_grid(3)
    net = TripNetwork(nodes, arcs, LON0, LAT0)
    d = net.reach(0, 250)                                   # node 0 is a corner, edge ~110 m
    assert d[0] == 0
    assert abs(d[1] - net.arc_len[arc_id(arcs, 0, 1)]) < 1e-9
    assert 2 in d and abs(d[2] - 2 * d[1]) < 1.0            # two edges away
    assert 8 not in net.reach(0, 250)                       # opposite corner is 4 edges (~440 m) away
    assert 8 in net.reach(0, 600) and 430 < net.reach(0, 600)[8] < 460


def test_shortest_arc_path_is_a_connected_chain():
    nodes, arcs = make_grid(3)
    net = TripNetwork(nodes, arcs, LON0, LAT0)
    a, b = arc_id(arcs, 0, 1), arc_id(arcs, 7, 8)
    path = net.shortest_arc_path(a, b, 2000)
    chain = [a] + path + [b]
    assert all(net.connected(x, y) for x, y in zip(chain, chain[1:]))
    assert net.shortest_arc_path(a, b, 50) is None          # too far for a 50 m bound
    nxt = int(net.successors(a)[0])
    assert net.shortest_arc_path(a, nxt, 1000) == []        # directly adjacent -> nothing in between


def test_components_two_disjoint_grids():
    n1, a1 = make_grid(2)
    n2, a2 = make_grid(2, lon0=LON0 + 0.05)
    nodes, arcs = np.vstack([n1, n2]), np.vstack([a1, a2 + len(n1)])
    net = TripNetwork(nodes, arcs, LON0, LAT0)
    s = net.stats()
    assert s["weak_components"] == 2 and s["strong_components"] == 2
    assert s["n_arcs"] == 16 and s["largest_weak_pct_arcs"] == 50.0


def test_one_way_street_is_not_traversable_backwards():
    nodes = np.array([[LON0, LAT0], [LON0 + 0.001, LAT0], [LON0 + 0.002, LAT0]])
    arcs = np.array([[0, 1], [1, 2]])                       # one-way eastwards
    net = TripNetwork(nodes, arcs, LON0, LAT0)
    assert net.connected(0, 1) and not net.connected(1, 0)
    assert 0 not in net.reach(2, 1000)                      # cannot drive back west


def test_projection_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "CACHE_DIR", tmp_path)
    nodes, arcs = make_grid(3)
    a = TripNetwork(nodes, arcs, LON0, LAT0, cache_key="t")
    assert (tmp_path / "t.npz").exists()
    b = TripNetwork(nodes, arcs, LON0, LAT0, cache_key="t")           # loaded from cache
    assert np.allclose(a.node_xy, b.node_xy)
    c = TripNetwork(nodes + 1e-4, arcs, LON0, LAT0, cache_key="t")    # changed data -> cache ignored
    assert not np.allclose(a.node_xy, c.node_xy)
