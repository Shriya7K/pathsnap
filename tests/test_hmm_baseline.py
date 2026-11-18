"""Tests for the distance-only baseline scorer."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmm.baseline_scores import SCORE_COLUMNS, distance_softmax, load_candidates_label_free  # noqa: E402
from hmm.graph import REPO  # noqa: E402


def cands():
    return pd.DataFrame({
        "point_idx": [0, 0, 0, 1, 1, 2],
        "arc_id":    [10, 11, 12, 10, 13, 14],
        "dist":      [2.0, 15.0, 40.0, 30.0, 30.0, 99.0],
    })


def test_probabilities_sum_to_one_per_point():
    p = distance_softmax(cands(), sigma=20.0)
    sums = p.groupby("point_idx")["prob"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)
    assert list(p.columns) == ["point_idx", "arc_id", "prob"]
    assert (p["prob"] > 0).all() and (p["prob"] <= 1).all()


def test_matches_hand_computed_softmax():
    p = distance_softmax(cands(), sigma=20.0)
    z = np.exp(-np.array([2.0, 15.0, 40.0]) / 20.0)
    assert np.allclose(p[p.point_idx == 0]["prob"].to_numpy(), z / z.sum())
    assert p[p.point_idx == 2]["prob"].iloc[0] == pytest.approx(1.0)        # single candidate -> 1
    two = p[p.point_idx == 1]["prob"].to_numpy()
    assert np.allclose(two, [0.5, 0.5])                                      # equal distances -> equal prob


def test_nearer_arc_gets_higher_probability_and_sigma_sharpens():
    soft = distance_softmax(cands(), sigma=100.0)
    sharp = distance_softmax(cands(), sigma=5.0)
    for p in (soft, sharp):
        q = p[p.point_idx == 0].set_index("arc_id")["prob"]
        assert q[10] > q[11] > q[12]
    assert sharp[sharp.point_idx == 0]["prob"].max() > soft[soft.point_idx == 0]["prob"].max()


def test_no_label_columns_are_used():
    """Poison every label-type column: the output must be identical to scoring without them."""
    clean = cands()
    poisoned = clean.assign(label=[1, 0, 0, 0, 1, 1], label_valid=0, route_pos=-999,
                            true_arc=12345, d_true=np.nan)
    a = distance_softmax(clean, 20.0)
    b = distance_softmax(poisoned, 20.0)
    pd.testing.assert_frame_equal(a, b)
    assert SCORE_COLUMNS == ["point_idx", "arc_id", "dist"]                  # the only inputs
    distance_softmax(clean[SCORE_COLUMNS], 20.0)                              # works with just those three


def test_large_distances_do_not_overflow():
    p = distance_softmax(pd.DataFrame({"point_idx": [0, 0], "arc_id": [1, 2], "dist": [5000.0, 5100.0]}), 1.0)
    assert np.isfinite(p["prob"]).all() and abs(p["prob"].sum() - 1) < 1e-12


def test_does_not_modify_the_input():
    c = cands()
    before = c.copy()
    distance_softmax(c, 20.0)
    pd.testing.assert_frame_equal(c, before)


@pytest.mark.skipif(not (REPO / "results" / "candidates").exists(), reason="P1 candidate files not generated")
def test_real_loader_returns_only_scoring_columns():
    files = sorted((REPO / "results" / "candidates").glob("student_test_*_cands.csv.gz"))
    if not files:
        pytest.skip("no student_test candidate files")
    trip = files[0].name.replace("student_test_", "").replace("_cands.csv.gz", "")
    c = load_candidates_label_free("student_test", trip)
    assert list(c.columns) == SCORE_COLUMNS
