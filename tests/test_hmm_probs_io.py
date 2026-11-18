"""probs_io / check_probs tests on tiny SYNTHETIC grid trips (hmm/fixture_trips.py). No real data needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmm.check_probs import main as check_main  # noqa: E402
from hmm.evaluate import evaluate_many  # noqa: E402
from hmm.fixture_trips import build_fixture  # noqa: E402
from hmm.probs_io import ProbsError, load_probs  # noqa: E402
from hmm.viterbi import load_config  # noqa: E402

DEV = ["train", "student_test"]


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("fx"))


@pytest.fixture(scope="module")
def fx_empty(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("fx_empty"), empty_trip=True)


def load(fx, path, **kw):
    return load_probs(path, DEV, fx["data_root"], fx["cands_dir"], **kw)


def write(df, path, fmt):
    p = Path(str(path) + "." + fmt)
    if fmt == "parquet":
        pytest.importorskip("pyarrow")
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)
    return p


def canon(df):
    return df[["trip_id", "point_idx", "arc_id", "prob"]].sort_values(["trip_id", "point_idx", "arc_id"]).reset_index(drop=True)


# ------------------------------------------------------------------ file formats and dtypes
@pytest.mark.parametrize("fmt", ["csv", "csv.gz", "parquet"])
def test_p2_style_files_load_in_every_format(fx, tmp_path, fmt):
    p = write(fx["probs"], tmp_path / "p", fmt)
    lp = load(fx, p)
    pd.testing.assert_frame_equal(canon(lp.df), canon(fx["probs"]))
    assert len(lp.trips) == 5 and lp.report["trips_present_vs_expected"] == {"train": (3, 3), "student_test": (2, 2)}


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_trip_id_saved_as_int_is_zero_padded(fx, tmp_path, fmt):
    df = fx["probs"].assign(trip_id=lambda d: d["trip_id"].astype(int))     # 1, 2, 3, 11, 12
    p = write(df, tmp_path / "p", fmt)
    lp = load(fx, p)
    assert set(lp.df["trip_id"]) == {"00000001", "00000002", "00000003", "00000011", "00000012"}
    assert all(isinstance(t, str) and len(t) == 8 for t in lp.df["trip_id"])
    assert any("normalised" in c for c in lp.changes)                        # reported, not silent


def test_trip_id_float_and_float_strings(fx, tmp_path):
    df = fx["probs"].assign(trip_id=lambda d: d["trip_id"].astype(int).astype(float))
    lp = load(fx, write(df, tmp_path / "p", "parquet"))
    assert "00000012" in set(lp.df["trip_id"])


def test_point_idx_and_arc_id_become_ints(fx, tmp_path):
    df = fx["probs"].assign(point_idx=lambda d: d["point_idx"].astype(float), arc_id=lambda d: d["arc_id"].astype(float))
    lp = load(fx, write(df, tmp_path / "p", "csv"))
    assert lp.df["point_idx"].dtype == np.int64 and lp.df["arc_id"].dtype == np.int64
    bad = fx["probs"].assign(point_idx=lambda d: d["point_idx"] + 0.5)
    with pytest.raises(ProbsError, match="non-integer"):
        load(fx, write(bad, tmp_path / "bad", "csv"))


def test_extra_columns_kept_and_shuffled_rows_give_same_result(fx, tmp_path):
    df = fx["probs"].assign(model="x", fold=np.where(fx["probs"]["trip_id"].isin(["00000001", "00000002", "00000003"]), 1, -1))
    df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    lp = load(fx, write(df, tmp_path / "p", "csv"))
    assert {"model", "fold"} <= set(lp.df.columns)
    pd.testing.assert_frame_equal(canon(lp.df), canon(fx["probs"]))


# ------------------------------------------------------------------ renamed columns
def test_renamed_columns_are_an_error_with_suggestions_unless_accepted(fx, tmp_path):
    df = fx["probs"].rename(columns={"prob": "proba", "arc_id": "candidate_arc"})
    p = write(df, tmp_path / "p", "csv")
    with pytest.raises(ProbsError) as e:
        load(fx, p)
    msg = str(e.value)
    assert "missing required column" in msg and "proba" in msg and "candidate_arc" in msg and "--accept-renames" in msg
    lp = load(fx, p, accept_renames=True)
    pd.testing.assert_frame_equal(canon(lp.df), canon(fx["probs"]))
    assert any("renamed" in c for c in lp.changes)


def test_missing_column_without_a_similar_one(fx, tmp_path):
    p = write(fx["probs"].drop(columns=["prob"]), tmp_path / "p", "csv")
    with pytest.raises(ProbsError, match="no similar column"):
        load(fx, p, accept_renames=True)


# ------------------------------------------------------------------ problems in the values
def test_missing_candidates_reported_and_fill_missing_adds_zero_rows(fx, tmp_path):
    df = fx["probs"].sample(frac=0.9, random_state=0)
    p = write(df, tmp_path / "p", "csv")
    lp = load(fx, p)
    r = lp.report
    assert r["candidate_rows_missing"] == len(fx["probs"]) - len(df) and r["candidate_rows_missing_pct"] > 9
    assert any("missing from the file" in w for w in lp.warnings) and len(lp.df) == len(df)   # not modified
    lp2 = load(fx, p, fill_missing=True)
    assert len(lp2.df) == len(fx["probs"]) and (lp2.df["prob"] == 0).sum() == r["candidate_rows_missing"]
    assert any("--fill-missing" in c for c in lp2.changes)


def test_extra_rows_not_in_candidates_are_reported(fx, tmp_path):
    extra = fx["probs"].iloc[:3].assign(arc_id=lambda d: d["arc_id"] + 1000, prob=0.1)
    lp = load(fx, write(pd.concat([fx["probs"], extra]), tmp_path / "p", "csv"))
    assert lp.report["rows_not_in_candidates"] == 3


def test_duplicates_block_unless_dropped(fx, tmp_path):
    dup = pd.concat([fx["probs"], fx["probs"].iloc[:4].assign(prob=0.001)])
    p = write(dup, tmp_path / "p", "csv")
    with pytest.raises(ProbsError, match="duplicate"):
        load(fx, p)
    lp = load(fx, p, drop_duplicates=True)
    assert len(lp.df) == len(fx["probs"]) and any("dropped 4" in c for c in lp.changes)
    kept = canon(lp.df).iloc[:4]["prob"].to_numpy()
    assert np.allclose(kept, canon(fx["probs"]).iloc[:4]["prob"].to_numpy())      # highest prob kept


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.2, 1.5])
def test_nan_inf_and_out_of_range_block(fx, tmp_path, bad):
    df = fx["probs"].copy()
    df.loc[df.index[5], "prob"] = bad
    with pytest.raises(ProbsError):
        load(fx, write(df, tmp_path / "p", "csv"))


def test_unnormalised_probs_warn_and_renormalise_only_with_flag(fx, tmp_path):
    df = fx["probs"].assign(prob=lambda d: d["prob"] * 0.5)
    p = write(df, tmp_path / "p", "csv")
    lp = load(fx, p)
    assert lp.report["points_sum_not_1"] == lp.report["points_total"] > 0
    assert np.allclose(lp.df["prob"].to_numpy(), canon(df)["prob"].to_numpy())          # untouched
    lp2 = load(fx, p, renormalise=True)
    sums = lp2.df.groupby(["trip_id", "point_idx"])["prob"].sum()
    assert np.allclose(sums, 1.0) and any("renormalised" in c for c in lp2.changes)


def test_all_zero_point_and_single_candidate_point_do_not_crash(fx, tmp_path):
    df = fx["probs"].copy()
    t = "00000001"
    df.loc[(df.trip_id == t) & (df.point_idx == 5), "prob"] = 0.0                       # all-zero point
    keep = df[(df.trip_id == t) & (df.point_idx == 7)].sort_values("prob").index[-1]
    df = df.drop(df[(df.trip_id == t) & (df.point_idx == 7)].index.difference([keep]))  # single-candidate point
    lp = load(fx, write(df, tmp_path / "p", "csv"), renormalise=True)
    assert lp.report["points_all_zero"] == 1 and lp.report["points_single_candidate"] >= 1
    rows = evaluate_many(lp.by_trip, lp.trips, load_config(), fx["data_root"], fx["cands_dir"], verbose=False)
    assert len(rows) == 5 and np.isfinite(rows["hmm_f1"]).all()


def test_empty_trip_with_no_candidates_is_valid(fx_empty, tmp_path):
    lp = load(fx_empty, write(fx_empty["probs"], tmp_path / "p", "csv"))
    empty = fx_empty["trips"][-1][1]
    assert empty in lp.empty_trips and len(lp.by_trip[empty]) == 0
    rows = evaluate_many(lp.by_trip, lp.trips, load_config(), fx_empty["data_root"], fx_empty["cands_dir"], verbose=False)
    r = rows[rows.trip_id == empty].iloc[0]
    assert len(rows) == 6 and r.hmm_f1 == 0 and r.cls_f1 == 0 and r.diag_empty_points == r.n_points


def test_missing_expected_trip_blocks_unless_partial(fx, tmp_path):
    df = fx["probs"][fx["probs"].trip_id != "00000011"]
    p = write(df, tmp_path / "p", "csv")
    with pytest.raises(ProbsError, match="have no rows"):
        load(fx, p)
    lp = load(fx, p, allow_partial=True)
    assert lp.report["missing_trips"] == ["student_test/00000011"]


# ------------------------------------------------------------------ evaluator_test and out-of-fold declarations
def test_evaluator_test_rows_are_rejected(fx, tmp_path):
    for col in ("partition", "split"):
        df = fx["probs"].assign(**{col: "student_test"})
        df.loc[df.index[:3], col] = "evaluator_test"
        with pytest.raises(ProbsError, match="evaluator_test"):
            load(fx, write(df, tmp_path / f"p_{col}", "csv"))


def test_unknown_trip_ids_are_rejected_as_possible_evaluator_trips(fx, tmp_path):
    df = pd.concat([fx["probs"], fx["probs"].iloc[:5].assign(trip_id="00000999")])
    with pytest.raises(ProbsError, match="evaluator_test"):
        load(fx, write(df, tmp_path / "p", "csv"))


def test_evaluator_partition_argument_is_refused(fx, tmp_path):
    with pytest.raises(ProbsError, match="evaluator_test"):
        load_probs(write(fx["probs"], tmp_path / "p", "csv"), ["evaluator_test"], fx["data_root"], fx["cands_dir"])


def test_oof_declarations_are_detected(fx, tmp_path):
    is_train = fx["probs"].trip_id.isin(["00000001", "00000002", "00000003"])
    fold = fx["probs"].assign(fold=np.where(is_train, 2, -1))
    assert load(fx, write(fold, tmp_path / "a", "csv")).declares_oof
    split = fx["probs"].assign(split=np.where(is_train, "oof", "held_out"))
    assert load(fx, write(split, tmp_path / "b", "csv")).declares_oof
    ins = fx["probs"].assign(split=np.where(is_train, "in_sample", "held_out"))
    lp = load(fx, write(ins, tmp_path / "c", "csv"))
    assert lp.declares_insample and not lp.declares_oof
    assert not load(fx, write(fx["probs"], tmp_path / "d", "csv")).declares_oof


# ------------------------------------------------------------------ the check_probs command
def cli(fx, path, *extra):
    return check_main(["--probs", str(path), "--data-root", str(fx["data_root"]), "--cands-dir", str(fx["cands_dir"]), *extra])


def test_check_probs_exit_codes(fx, tmp_path, capsys):
    ok = write(fx["probs"], tmp_path / "ok", "csv")
    assert cli(fx, ok) == 0 and "result: OK" in capsys.readouterr().out
    bad = write(fx["probs"].assign(prob=np.nan), tmp_path / "bad", "csv")
    assert cli(fx, bad) == 1 and "FAILED" in capsys.readouterr().out
    assert cli(fx, tmp_path / "nope.csv") == 1                                       # missing file
    ren = write(fx["probs"].rename(columns={"prob": "p"}), tmp_path / "ren", "csv")
    assert cli(fx, ren) == 1 and cli(fx, ren, "--accept-renames") == 0


def test_candidates_missing_is_blocking_for_check_probs(fx, tmp_path, capsys):
    p = write(fx["probs"], tmp_path / "p", "csv")
    code = check_main(["--probs", str(p), "--data-root", str(fx["data_root"]), "--cands-dir", str(tmp_path / "none")])
    assert code == 1 and "candidate files not found" in capsys.readouterr().out
