"""hmm/import_p2.py on SYNTHETIC per-trip files in P2's layout (extra feature columns, no trip_id column)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmm import import_p2  # noqa: E402
from hmm.import_p2 import ImportError_, import_dir  # noqa: E402


def trip_df(n_points=3, k=3, raw=None):
    """P2-style file body: trip, point_idx, arc_id, two fake features, probability."""
    rows = [(p, 10 * p + a) for p in range(n_points) for a in range(k)]
    df = pd.DataFrame(rows, columns=["point_idx", "arc_id"])
    df.insert(0, "trip", "test_x")
    df["dist"], df["rank"] = 5.0, 0
    df["probability"] = raw if raw is not None else 0.2          # sums to 0.6 per point (not 1)
    return df


def write(d, name, df):
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / name, index=False)


@pytest.fixture
def d(tmp_path):
    return tmp_path / "pred"


def test_normal_case_two_trips(d):
    write(d, "student_test_00000003.csv.gz", trip_df())
    write(d, "student_test_00000007.csv", trip_df(2, 2, raw=[0.1, 0.3, 0.5, 0.5]))
    out = import_dir(d, "student_test", expect_trips=2, verbose=False)
    assert list(out.columns) == ["partition", "trip_id", "point_idx", "arc_id", "prob", "prob_raw"]
    assert set(out.trip_id) == {"00000003", "00000007"} and (out.partition == "student_test").all()
    assert len(out) == 9 + 4
    assert np.allclose(out.groupby(["trip_id", "point_idx"]).prob.sum(), 1.0)
    a = out[(out.trip_id == "00000007") & (out.point_idx == 0)]
    assert np.allclose(a.prob, [0.25, 0.75]) and np.allclose(a.prob_raw, [0.1, 0.3])   # raw kept, prob scaled
    assert not out.duplicated(["trip_id", "point_idx", "arc_id"]).any()
    assert not out.attrs["warnings"]


def test_trip_id_comes_from_filename_not_from_file_content(d):
    write(d, "student_test_00000042.csv.gz", trip_df())                # content says trip = test_x
    assert set(import_dir(d, "student_test", verbose=False).trip_id) == {"00000042"}


def test_unnormalised_probs_are_renormalised(d, capsys):
    write(d, "student_test_00000001.csv.gz", trip_df(raw=0.01))
    out = import_dir(d, "student_test")
    assert np.allclose(out.prob, 1 / 3) and np.allclose(out.prob_raw, 0.01)
    assert "mean 0.0300" in capsys.readouterr().out                    # raw per-point sum printed


def test_all_zero_point_becomes_uniform_and_is_counted(d, capsys):
    raw = [0.0, 0.0, 0.0, 0.2, 0.6, 0.2]
    write(d, "student_test_00000001.csv.gz", trip_df(2, 3, raw=raw))
    out = import_dir(d, "student_test")
    assert np.allclose(out[out.point_idx == 0].prob, 1 / 3) and (out[out.point_idx == 0].prob_raw == 0).all()
    assert np.allclose(out[out.point_idx == 1].prob, [0.2, 0.6, 0.2])
    assert "all-zero points     : 1" in capsys.readouterr().out


def test_duplicates_refused(d):
    df = pd.concat([trip_df(), trip_df().iloc[:1]])
    write(d, "student_test_00000001.csv.gz", df)
    with pytest.raises(ImportError_, match="duplicate"):
        import_dir(d, "student_test", verbose=False)


@pytest.mark.parametrize("bad, msg", [(np.nan, "NaN"), (np.inf, "infinite"), (1.5, "outside"), (-0.1, "outside")])
def test_bad_probability_values_refused(d, bad, msg):
    df = trip_df()
    df.loc[2, "probability"] = bad
    write(d, "student_test_00000001.csv.gz", df)
    with pytest.raises(ImportError_, match=msg):
        import_dir(d, "student_test", verbose=False)


def test_missing_column_refused(d):
    write(d, "student_test_00000001.csv.gz", trip_df().drop(columns=["probability"]))
    with pytest.raises(ImportError_, match="missing column"):
        import_dir(d, "student_test", verbose=False)


def test_evaluator_test_filename_and_partition_refused(d):
    write(d, "student_test_00000001.csv.gz", trip_df())
    write(d, "evaluator_test_00000001.csv.gz", trip_df())
    with pytest.raises(ImportError_, match="evaluator"):
        import_dir(d, "student_test", verbose=False)
    with pytest.raises(ImportError_, match="refused"):
        import_dir(d, "evaluator_test", verbose=False)
    with pytest.raises(ImportError_, match="evaluator"):
        import_p2.parse_name("evaluator_test_00000009.csv.gz")


def test_wrong_trip_count_warns_but_still_imports(d, capsys):
    write(d, "student_test_00000001.csv.gz", trip_df())
    out = import_dir(d, "student_test", expect_trips=10)
    assert len(out) == 9 and out.attrs["warnings"] and "WARNING: --expect-trips 10 but found 1" in capsys.readouterr().out


def test_other_partition_files_are_ignored_and_train_oof_layout_works(d):
    write(d, "student_test_00000001.csv.gz", trip_df())
    write(d, "train_00000005.csv.gz", trip_df())
    tr = import_dir(d, "train", verbose=False)
    assert set(tr.trip_id) == {"00000005"} and (tr.partition == "train").all()
    assert set(import_dir(d, "student_test", verbose=False).trip_id) == {"00000001"}


def test_bad_filename_and_empty_dir_refused(d):
    write(d, "student_test_abc.csv.gz", trip_df())
    with pytest.raises(ImportError_, match="file name"):
        import_dir(d, "student_test", verbose=False)
    with pytest.raises(ImportError_, match="not found"):
        import_dir(d / "nope", "student_test", verbose=False)
    (d.parent / "empty").mkdir()
    with pytest.raises(ImportError_, match="no student_test"):
        import_dir(d.parent / "empty", "student_test", verbose=False)


def test_cli_writes_file_and_returns_codes(d, tmp_path):
    write(d, "student_test_00000001.csv.gz", trip_df())
    out = tmp_path / "p2" / "probs.csv.gz"
    assert import_p2.main(["--in-dir", str(d), "--partition", "student_test", "--out", str(out)]) == 0
    back = pd.read_csv(out, dtype={"trip_id": str})
    assert list(back.columns) == ["partition", "trip_id", "point_idx", "arc_id", "prob", "prob_raw"]
    assert back.trip_id.iloc[0] == "00000001"                            # zero padding survives
    assert import_p2.main(["--in-dir", str(d), "--partition", "evaluator_test", "--out", str(out)]) == 1
