"""Metric tests."""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmm.evaluate import classifier_only, continuity, lcs_length, pointwise_accuracy, route_metrics  # noqa: E402
from hmm.graph import TripNetwork  # noqa: E402


def lcs_dp(a, b):
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            m[i + 1][j + 1] = m[i][j] + 1 if a[i] == b[j] else max(m[i][j + 1], m[i + 1][j])
    return m[-1][-1]


def test_lcs_matches_dynamic_programming():
    random.seed(0)
    for _ in range(200):
        a = [random.randint(0, 5) for _ in range(random.randint(0, 20))]
        b = [random.randint(0, 5) for _ in range(random.randint(0, 20))]
        assert lcs_length(a, b) == lcs_dp(a, b)


def line_net():
    nodes = np.array([[77.5 + i * 0.001, 13.0] for i in range(6)])
    arcs = np.array([[i, i + 1] for i in range(5)])         # arc i: node i -> i+1 (a chain)
    return TripNetwork(nodes, arcs, 77.5, 13.0)


def test_route_metrics_known_values():
    net = line_net()
    m = route_metrics([0, 1, 3], [0, 1, 2], net)             # 2 of 3 arcs right
    for k in ("precision", "recall", "f1", "lcs_sim"):
        assert abs(m[k] - 2 / 3) < 1e-12
    assert m["breaks"] == 1 and abs(m["continuity"] - 0.5) < 1e-12      # 1->3 is a break, 0->1 is fine


def test_continuity_edges():
    net = line_net()
    assert continuity([0, 1, 2, 3], net) == {"continuity": 1.0, "breaks": 0}
    assert continuity([2], net) == {"continuity": 1.0, "breaks": 0}
    assert continuity([3, 1], net)["breaks"] == 1


def test_classifier_only_and_pointwise_accuracy():
    p = pd.DataFrame({"point_idx": [0, 0, 1, 1, 3], "arc_id": [5, 6, 7, 8, 9], "prob": [.2, .8, .5, .5, 1.0]})
    got = classifier_only(p, 4)
    assert got == [6, 7, None, 9]                            # tie -> smaller arc id; empty point -> None
    lab = pd.DataFrame({"true_arc": [6, 8, 1, 9], "label_valid": [1, 1, 0, 1]})
    assert abs(pointwise_accuracy(got, lab) - 2 / 3) < 1e-12  # point 2 (unreliable label) is ignored
