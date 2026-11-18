"""Tests for hmm/infer.py (the API entry point) on a small synthetic grid trip."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmm import infer  # noqa: E402
from hmm.graph import REPO, TripNetwork, load_trip  # noqa: E402
from hmm.infer import route_for_inputs, score_candidates  # noqa: E402

@pytest.fixture(autouse=True)
def baseline_by_default(monkeypatch):
    """The tests below were written for the distance-only baseline; the real-model tests ask for scorer='rf'."""
    monkeypatch.setenv("PATHSNAP_SCORER", "baseline")


LON0, LAT0, STEP = 77.5, 13.0, 0.001          # grid spacing ~ 108-111 m


def grid(rows=3, cols=4):
    nodes = np.array([[LON0 + c * STEP, LAT0 + r * STEP] for r in range(rows) for c in range(cols)])
    arcs = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                arcs += [(u, u + 1), (u + 1, u)]
            if r + 1 < rows:
                arcs += [(u, u + cols), (u + cols, u)]
    return nodes, np.array(arcs)


def drive_east_along_bottom_row(nodes, n_pts=30):
    """GPS points every ~10 m along row 0 (node 0 -> node 3), ~4 m north of the road, 1 Hz."""
    lon = np.linspace(nodes[0][0], nodes[3][0], n_pts)
    lat = np.full(n_pts, LAT0 + 0.00004)
    return pd.DataFrame({"lon": lon, "lat": lat, "t": np.arange(1.0, n_pts + 1)})


def test_returns_agreed_format_on_grid_trip():
    nodes, arcs = grid()
    gps = drive_east_along_bottom_row(nodes)
    point_arcs, route = route_for_inputs(gps, nodes, arcs)
    assert isinstance(point_arcs, list) and len(point_arcs) == len(gps)
    assert all(a is None or type(a) is int for a in point_arcs)
    assert isinstance(route, list) and len(route) > 0 and all(type(a) is int for a in route)
    assert all(a != b for a, b in zip(route, route[1:]))                  # consecutive duplicates removed
    assert all(0 <= a < len(arcs) for a in route)


def test_route_follows_the_driven_road_and_is_connected():
    nodes, arcs = grid()
    gps = drive_east_along_bottom_row(nodes)
    _, route = route_for_inputs(gps, nodes, arcs)
    net = TripNetwork(nodes, arcs, LON0, LAT0)
    east = [i for i, (u, v) in enumerate(arcs) if (u, v) in ((0, 1), (1, 2), (2, 3))]
    assert route == east                                                  # exactly the three eastward arcs
    assert all(net.connected(a, b) for a, b in zip(route, route[1:]))


def test_empty_and_single_point_inputs_do_not_crash():
    nodes, arcs = grid()
    assert route_for_inputs(pd.DataFrame({"lon": [], "lat": [], "t": []}), nodes, arcs) == ([], [])
    one = pd.DataFrame({"lon": [nodes[1][0]], "lat": [LAT0 + 0.00004], "t": [1.0]})
    point_arcs, route = route_for_inputs(one, nodes, arcs)
    assert len(point_arcs) == 1 and len(route) <= 1


def test_far_away_points_have_no_candidates_and_give_none():
    nodes, arcs = grid()
    gps = pd.DataFrame({"lon": [LON0 + 0.5, LON0 + 0.5001], "lat": [LAT0 + 0.5, LAT0 + 0.5], "t": [1.0, 2.0]})
    assert route_for_inputs(gps, nodes, arcs) == ([None, None], [])


def test_works_from_any_working_directory(tmp_path, monkeypatch):
    nodes, arcs = grid()
    gps = drive_east_along_bottom_row(nodes)
    monkeypatch.chdir(tmp_path)                                           # config path must still resolve
    _, route = route_for_inputs(gps, nodes, arcs, config_path="hmm/config.json")
    assert len(route) > 0


def test_score_candidates_is_the_single_swap_point(monkeypatch):
    """Replacing score_candidates changes the result - proves it is the only scoring hook."""
    nodes, arcs = grid()
    gps = drive_east_along_bottom_row(nodes)
    seen = {}

    def my_model(cands_df):
        seen["cols"] = set(cands_df.columns)
        return pd.DataFrame({"point_idx": cands_df["point_idx"], "arc_id": cands_df["arc_id"], "prob": 0.5})

    monkeypatch.setattr(infer, "score_candidates", my_model)
    route_for_inputs(gps, nodes, arcs)
    assert {"point_idx", "arc_id", "dist", "rank", "x", "y", "ts"} <= seen["cols"]
    assert not seen["cols"] & {"label", "label_valid", "true_arc", "route_pos", "d_true"}   # label-free input


def test_score_candidates_baseline_output():
    c = pd.DataFrame({"point_idx": [0, 0], "arc_id": [1, 2], "dist": [3.0, 60.0]})
    p = score_candidates(c)
    assert list(p.columns) == ["point_idx", "arc_id", "prob"] and abs(p.prob.sum() - 1) < 1e-12


@pytest.mark.skipif(not (REPO / "data" / "student_test").exists()
                    or not (REPO / "results" / "candidates").exists(), reason="needs data + P1 candidates")
def test_real_trip_matches_the_batch_baseline_path():
    """route_for_inputs (label-free candidate_gen) == batch path (P1 candidate files + baseline scores)."""
    from hmm.baseline_scores import distance_softmax, load_candidates_label_free
    from hmm.viterbi import decode
    trip = "00000052"                                                      # 262 points: quick
    gps, nodes, arcs, _ = load_trip("student_test", trip, with_route=False)   # .route is NOT read
    pa_infer, route_infer = route_for_inputs(gps, nodes, arcs)
    probs = distance_softmax(load_candidates_label_free("student_test", trip))
    pa_batch, route_batch = decode(gps, nodes, arcs, probs)
    assert pa_infer == pa_batch and route_infer == route_batch


# ------------------------------------------------------------------ model object (predict_proba-style) swap
class FakeModel:
    """Stands in for P2's model: scikit-learn style predict_proba on the label-free candidate table."""

    def __init__(self, two_columns=True):
        self.two_columns, self.seen = two_columns, None

    def predict_proba(self, cands_df):
        self.seen = cands_df.copy()
        p1 = np.exp(-cands_df["dist"].to_numpy() / 15.0)            # a made-up scorer: nearer = likelier
        return np.column_stack([1 - p1, p1]) if self.two_columns else p1


def test_fake_model_object_is_used_via_model_argument():
    nodes, arcs = grid()
    gps = drive_east_along_bottom_row(nodes)
    m = FakeModel()
    point_arcs, route = route_for_inputs(gps, nodes, arcs, model=m)
    assert m.seen is not None and {"point_idx", "arc_id", "dist", "rank", "x", "y", "ts"} <= set(m.seen.columns)
    assert not set(m.seen.columns) & {"label", "label_valid", "true_arc", "route_pos", "d_true"}
    assert route == [i for i, (u, v) in enumerate(arcs) if (u, v) in ((0, 1), (1, 2), (2, 3))]
    assert len(point_arcs) == len(gps)


def test_set_model_registers_for_the_whole_process_and_can_be_reset():
    nodes, arcs = grid()
    gps = drive_east_along_bottom_row(nodes)
    m = FakeModel(two_columns=False)                                # (n,) output is also accepted
    infer.set_model(m)
    try:
        _, route = route_for_inputs(gps, nodes, arcs)               # same signature as before
        assert m.seen is not None and len(route) == 3
    finally:
        infer.set_model(None)
    m.seen = None
    route_for_inputs(gps, nodes, arcs)
    assert m.seen is None                                           # back to the distance-only baseline


@pytest.mark.parametrize("bad", [lambda n: np.zeros(n - 1), lambda n: np.zeros((n, 3)),
                                 lambda n: np.full(n, np.nan)])
def test_malformed_model_output_is_rejected_with_a_clear_error(bad):
    class Bad:
        def predict_proba(self, df):
            return bad(len(df))

    nodes, arcs = grid()
    with pytest.raises(ValueError):
        route_for_inputs(drive_east_along_bottom_row(nodes), nodes, arcs, model=Bad())


# ------------------------------------------------------------------ P2's real model as the scorer (scorer="rf")
needs_rf = pytest.mark.skipif(not infer.MODEL_PATH.exists(), reason="models/final_model.joblib not built")


def test_scorer_selection_by_argument_and_environment(monkeypatch):
    assert infer.default_scorer() == "baseline"                      # autouse fixture sets the variable
    monkeypatch.setenv("PATHSNAP_SCORER", "rf")
    assert infer.default_scorer() == "rf"
    monkeypatch.delenv("PATHSNAP_SCORER")
    assert infer.default_scorer() == "rf"                            # the default is P2's model
    monkeypatch.setenv("PATHSNAP_SCORER", "nonsense")
    with pytest.raises(ValueError):
        infer.default_scorer()
    nodes, arcs = grid()
    with pytest.raises(ValueError):
        route_for_inputs(drive_east_along_bottom_row(nodes), nodes, arcs, scorer="nonsense")


@needs_rf
def test_rf_scorer_on_a_small_grid_trip_gives_normalised_probabilities_and_a_connected_route():
    nodes, arcs = grid()
    gps = drive_east_along_bottom_row(nodes)
    cands = infer.generate_candidates(gps, nodes, arcs)
    p = score_candidates(cands, scorer="rf")
    assert list(p.columns) == ["point_idx", "arc_id", "prob"] and len(p) == len(cands)
    assert np.allclose(p.groupby("point_idx").prob.sum(), 1.0) and p.prob.between(0, 1).all()
    point_arcs, route = route_for_inputs(gps, nodes, arcs, scorer="rf")
    assert len(point_arcs) == len(gps) and len(route) > 0 and all(type(a) is int for a in route)


@needs_rf
def test_rf_scorer_input_has_no_label_columns_and_features_come_from_p2(monkeypatch):
    nodes, arcs = grid()
    seen = {}
    real = infer.score_rf

    def spy(td, cands_df):
        seen["cols"] = set(cands_df.columns)
        return real(td, cands_df)

    monkeypatch.setattr(infer, "score_rf", spy)
    route_for_inputs(drive_east_along_bottom_row(nodes), nodes, arcs, scorer="rf")
    assert not seen["cols"] & {"label", "label_valid", "true_arc", "route_pos", "d_true"}
    assert infer.load_rf()["features"][:2] == ["dist", "log_dist"]   # list read from P2's metadata json


@needs_rf
@pytest.mark.skipif(not (REPO / "data" / "student_test").exists()
                    or not (REPO / "results" / "predictions" / "student_test_00000052.csv.gz").exists(),
                    reason="needs data + P2 prediction files")
def test_rf_inference_matches_p2_batch_predictions_on_a_real_trip():
    """route_for_inputs(rf) == decode() on P2's own prediction file for the trip (renormalised per point)."""
    from hmm.import_p2 import import_dir
    from hmm.viterbi import decode
    trip = "00000052"
    gps, nodes, arcs, _ = load_trip("student_test", trip, with_route=False)
    pa_infer, route_infer = route_for_inputs(gps, nodes, arcs, scorer="rf")
    allp = import_dir(REPO / "results" / "predictions", "student_test", verbose=False)
    probs = allp[allp.trip_id == trip][["point_idx", "arc_id", "prob"]]
    pa_batch, route_batch = decode(gps, nodes, arcs, probs)
    assert pa_infer == pa_batch and route_infer == route_batch
