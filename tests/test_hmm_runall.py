"""tune out-of-fold guard and the one-command runner, on SYNTHETIC grid trips (hmm/fixture_trips.py)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmm import run_all, tune  # noqa: E402
from hmm.fixture_trips import build_fixture  # noqa: E402
from hmm.viterbi import CONFIG_PATH  # noqa: E402

TRAIN = ["00000001", "00000002", "00000003"]


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx_run")
    f = build_fixture(d)
    f["probs"].to_csv(d / "probs.csv", index=False)
    f["file"] = d / "probs.csv"
    return f


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def common(fx):
    return ["--data-root", str(fx["data_root"]), "--cands-dir", str(fx["cands_dir"])]


# ------------------------------------------------------------------ tune: out-of-fold guard
def test_tune_refuses_train_trips_without_oof_flag(fx, tmp_path, capsys):
    log = tmp_path / "log.csv"
    with pytest.raises(SystemExit) as e:
        tune.main(["--probs", str(fx["file"]), "--n-trials", "1", "--log", str(log), *common(fx)])
    assert e.value.code == 2 and not log.exists()
    out = capsys.readouterr().out
    assert "REFUSING TO TUNE" in out and "--probs-are-oof" in out


def test_tune_runs_with_oof_flag_and_logs_it(fx, tmp_path, capsys):
    cfg_before = md5(CONFIG_PATH)
    log = tmp_path / "log.csv"
    tune.main(["--probs", str(fx["file"]), "--n-trials", "2", "--seed", "1", "--probs-are-oof", "--log", str(log),
               *common(fx)])
    d = pd.read_csv(log)
    assert len(d) == 2 and bool(d["train_probs_oof"].all()) and d.loc[0, "is_baseline"]
    assert "relying on --probs-are-oof" in capsys.readouterr().out          # file itself had no fold column
    assert md5(CONFIG_PATH) == cfg_before                                    # no --write-config -> untouched


def test_tune_on_student_test_only_needs_no_flag(fx, tmp_path):
    log = tmp_path / "log.csv"
    tune.main(["--probs", str(fx["file"]), "--partitions", "student_test", "--n-trials", "1", "--log", str(log),
               *common(fx)])
    assert len(pd.read_csv(log)) == 1


def test_tune_refuses_when_file_declares_in_sample_even_with_flag(fx, tmp_path, capsys):
    df = fx["probs"].assign(split=np.where(fx["probs"].trip_id.isin(TRAIN), "in_sample", "held_out"))
    p = tmp_path / "p.csv"
    df.to_csv(p, index=False)
    with pytest.raises(SystemExit) as e:
        tune.main(["--probs", str(p), "--n-trials", "1", "--probs-are-oof", "--log", str(tmp_path / "l.csv"), *common(fx)])
    assert e.value.code == 2 and "in-sample" in capsys.readouterr().out


def test_tune_accepts_file_with_fold_column_plus_flag_without_warning(fx, tmp_path, capsys):
    df = fx["probs"].assign(fold=np.where(fx["probs"].trip_id.isin(TRAIN), 0, -1))
    p = tmp_path / "p.csv"
    df.to_csv(p, index=False)
    tune.main(["--probs", str(p), "--n-trials", "1", "--probs-are-oof", "--log", str(tmp_path / "l.csv"), *common(fx)])
    assert "relying on --probs-are-oof" not in capsys.readouterr().out


def test_fold_column_without_flag_still_refuses_but_hints(fx, tmp_path, capsys):
    df = fx["probs"].assign(fold=np.where(fx["probs"].trip_id.isin(TRAIN), 0, -1))
    p = tmp_path / "p.csv"
    df.to_csv(p, index=False)
    with pytest.raises(SystemExit):
        tune.main(["--probs", str(p), "--n-trials", "1", "--log", str(tmp_path / "l.csv"), *common(fx)])
    assert "looks out-of-fold" in capsys.readouterr().out


def test_tune_never_accepts_evaluator_test(fx, tmp_path):
    with pytest.raises(SystemExit):
        tune.main(["--probs", str(fx["file"]), "--partitions", "evaluator_test", "--n-trials", "1",
                   "--log", str(tmp_path / "l.csv"), *common(fx)])


# ------------------------------------------------------------------ run_all
def run(fx, tmp_path, label, *extra, probs=None):
    return run_all.main(["--probs", str(probs or fx["file"]), "--label", label, "--out-root", str(tmp_path / "results"),
                         "--no-png", *common(fx), *extra])


def test_run_all_end_to_end_with_tuning_in_its_own_folder(fx, tmp_path):
    cfg_before = md5(CONFIG_PATH)
    assert run(fx, tmp_path, "Synthetic Fixture", "--oof", "--tune-trials", "2") == 0
    out = tmp_path / "results" / "hmm_synthetic_fixture"
    for f in ["check_probs_report.txt", "comparison.csv", "summary.txt", "diagnose_by_trip.csv", "diagnose_summary.txt",
              "tuning_log.csv", "tuned_config.json", "comparison_tuned.csv", "summary_tuned.txt", "latency.csv",
              "run_summary.txt"]:
        assert (out / f).exists(), f
    assert list((out / "maps").glob("*.html"))
    assert len(pd.read_csv(out / "tuning_log.csv")) == 2 and len(pd.read_csv(out / "comparison.csv")) == 5
    assert (pd.read_csv(out / "comparison.csv")["score_source"] == "Synthetic Fixture").all()
    assert md5(CONFIG_PATH) == cfg_before                                    # hmm/config.json untouched
    text = (out / "run_summary.txt").read_text()
    assert "FINAL SUMMARY" in text and "improved" in text and "was NOT modified" in text
    assert not (tmp_path / "results" / "hmm_baseline").exists()              # other result folders untouched


def test_run_all_without_tuning_skips_that_step(fx, tmp_path):
    assert run(fx, tmp_path, "plain run") == 0
    out = tmp_path / "results" / "hmm_plain_run"
    assert not (out / "tuning_log.csv").exists() and (out / "comparison.csv").exists()


def test_run_all_refuses_to_tune_train_without_oof_before_doing_any_work(fx, tmp_path):
    assert run(fx, tmp_path, "no oof", "--tune-trials", "2") == 2
    out = tmp_path / "results" / "hmm_no_oof"
    assert not (out / "comparison.csv").exists() and "cannot tune" in (out / "run_summary.txt").read_text()


def test_run_all_stops_at_the_first_blocking_error(fx, tmp_path):
    bad = tmp_path / "bad.csv"
    fx["probs"].assign(prob=np.nan).to_csv(bad, index=False)
    assert run(fx, tmp_path, "bad file", probs=bad) == 1
    out = tmp_path / "results" / "hmm_bad_file"
    assert not (out / "comparison.csv").exists() and "Stopped at step 1" in (out / "run_summary.txt").read_text()


def test_run_all_never_overwrites_and_reserves_baseline(fx, tmp_path):
    assert run(fx, tmp_path, "twice") == 0
    assert run(fx, tmp_path, "twice") == 1                                   # exists, no --overwrite
    assert run(fx, tmp_path, "twice", "--overwrite") == 0
    assert run(fx, tmp_path, "Baseline") == 1                                # reserved: hmm_baseline*


def test_run_all_write_config_rules(fx, tmp_path):
    assert run(fx, tmp_path, "wc", "--write-config") == 1                    # needs --tune-trials
    assert run(fx, tmp_path, "synthetic wc", "--oof", "--tune-trials", "1", "--write-config") == 1   # synthetic refused


def test_run_all_rejects_evaluator_rows(fx, tmp_path):
    p = tmp_path / "p.csv"
    fx["probs"].assign(partition="evaluator_test").to_csv(p, index=False)
    assert run(fx, tmp_path, "evaluator", probs=p) == 1
    assert not (tmp_path / "results" / "hmm_evaluator" / "comparison.csv").exists()


def test_tune_objective_f1_exclude_trips_and_extra_logged_metrics(fx, tmp_path):
    log, trips_out = tmp_path / "l.csv", tmp_path / "trips.csv"
    tune.main(["--probs", str(fx["file"]), "--n-trials", "2", "--seed", "3", "--probs-are-oof", "--objective", "f1",
               "--exclude-trips", "00000002", "--max-trips", "2", "--log", str(log), "--trips-out", str(trips_out),
               *common(fx)])
    d, t = pd.read_csv(log), pd.read_csv(trips_out, dtype={"trip_id": str})
    assert len(t) == 2 and "00000002" not in set(t.trip_id) and (t.seed == 3).all()
    assert (d.objective_name == "f1").all() and np.allclose(d.objective, d.mean_f1)       # objective = mean F1
    for c in ["mean_f1_raw", "mean_precision", "mean_recall", "mean_point_acc"]:           # fill_path cannot hide
        assert c in d.columns and d[c].between(0, 1).all()
    # same seed -> same trip subset
    t2 = tmp_path / "trips2.csv"
    tune.main(["--probs", str(fx["file"]), "--n-trials", "1", "--seed", "3", "--probs-are-oof", "--objective", "f1",
               "--exclude-trips", "00000002", "--max-trips", "2", "--log", str(tmp_path / "l2.csv"),
               "--trips-out", str(t2), *common(fx)])
    assert pd.read_csv(t2, dtype={"trip_id": str}).equals(t)
