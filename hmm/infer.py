"""Inference entry point for the API (Person 4 calls route_for_inputs).

    from hmm.infer import route_for_inputs
    point_arcs, route = route_for_inputs(gps_df, nodes, arcs, config_path="hmm/config.json")

  gps_df  : DataFrame with columns lon, lat, t   (one row per GPS point, in time order)
  nodes   : ndarray (N, 2)  lon, lat   (row index = node id)
  arcs    : ndarray (M, 2)  from_node, to_node   (row index = arc id)
  returns : point_arcs  list[int | None], one per GPS point (None = no candidate)
            route       list[int] arc ids in travel order, consecutive duplicates removed

Pipeline (reads NO reference data, never a .route file):
  1. candidates   : P1's label-free data_processing/candidate_gen.py
  2. score        : score_candidates(cands_df)   <-- P2's random forest (models/final_model.joblib, 14 features built
                    by P2's own build_features), renormalised per GPS point. The distance-only baseline stays
                    selectable: route_for_inputs(..., scorer="baseline") or environment PATHSNAP_SCORER=baseline.
  3. decode       : hmm.viterbi.decode (HMM + Viterbi) with parameters from config_path
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_scores import DEFAULT_SIGMA_M, distance_softmax
from .graph import REPO
from .viterbi import decode

DEFAULT_CONFIG = "hmm/config.json"
MODEL_PATH = REPO / "models" / "final_model.joblib"
METADATA_PATH = REPO / "models" / "final_model_metadata.json"
SCORERS = ("rf", "baseline")


def default_scorer():
    """'rf' (P2's model) unless the environment variable PATHSNAP_SCORER says 'baseline'."""
    v = os.environ.get("PATHSNAP_SCORER", "rf").strip().lower()
    if v not in SCORERS:
        raise ValueError(f"PATHSNAP_SCORER must be one of {SCORERS}, got {v!r}")
    return v


_RF = {}               # cache: {"model": ..., "features": [...], "pos": int}


def load_rf():
    """Load P2's model + the feature list from her metadata json (cached). Not editable here: P2 owns both files."""
    if not _RF:
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} / {METADATA_PATH} not found. Run P2's data_processing/train_final_model.py, "
                "or use the distance-only baseline (scorer='baseline' or PATHSNAP_SCORER=baseline).")
        import joblib
        model = joblib.load(MODEL_PATH)
        feats = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["features"]
        if model.n_features_in_ != len(feats) or 1 not in model.classes_:
            raise ValueError("P2's model and metadata do not match (feature count / positive class)")
        _RF.update(model=model, features=feats, pos=list(model.classes_).index(1))
    return _RF


def score_rf(td, cands_df, timing=None):
    """P2's random forest on the label-free features of P2's build_features(td); returns point_idx, arc_id, prob.
    prob = predict_proba(class 1), renormalised so each GPS point's candidates sum to 1 (all-zero point -> uniform)."""
    sys.path.insert(0, str(REPO))
    from data_processing.features import build_features      # P2's code, reused unchanged
    rf = load_rf()
    t0 = time.perf_counter()
    F = build_features(td)                                    # one row per (point_idx, arc_id)
    if len(F) == 0:
        return pd.DataFrame({"point_idx": [], "arc_id": [], "prob": []})
    X = F[rf["features"]].astype(np.float32)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("non-finite features")
    t1 = time.perf_counter()
    p = np.concatenate([rf["model"].predict_proba(X.iloc[i:i + 5000])[:, rf["pos"]] for i in range(0, len(X), 5000)])
    t2 = time.perf_counter()
    if timing is not None:                                    # optional stage timing (hmm/latency.py)
        timing["features_s"], timing["model_s"] = t1 - t0, t2 - t1
    out = pd.DataFrame({"point_idx": F["point_idx"].to_numpy(), "arc_id": F["arc_id"].to_numpy(), "prob": p})
    s = out.groupby("point_idx")["prob"].transform("sum")
    n = out.groupby("point_idx")["prob"].transform("size")
    out["prob"] = np.where(s > 0, out["prob"] / s.where(s > 0, 1.0), 1.0 / n)
    return out


# --------------------------------------------------------------------------- step 2: the swap point
_MODEL = None          # P2's model object, once registered with set_model(); None = distance-only baseline


def set_model(model):
    """Register P2's model for the whole process (call once at API start-up). set_model(None) goes back to
    the distance-only baseline.

    Contract: `model.predict_proba(cands_df)` receives the label-free candidate DataFrame (columns point_idx,
    arc_id, dist, rank, x, y, ts - one row per point/candidate) and returns either a 1-D array of probabilities
    (one per row, same order) or an (n, 2) array in the scikit-learn convention (column 1 = P(candidate is the
    true arc)). P2's wrapper does its own feature engineering inside predict_proba."""
    global _MODEL
    _MODEL = model


def score_with_model(model, cands_df):
    """cands_df -> DataFrame(point_idx, arc_id, prob) using model.predict_proba (validated)."""
    p = np.asarray(model.predict_proba(cands_df), dtype=float)
    if p.ndim == 2:
        if p.shape[1] != 2:
            raise ValueError(f"predict_proba must return shape (n,) or (n, 2), got {p.shape}")
        p = p[:, 1]
    if p.shape != (len(cands_df),):
        raise ValueError(f"predict_proba returned {p.shape[0]} values for {len(cands_df)} candidate rows")
    if not np.isfinite(p).all():
        raise ValueError("predict_proba returned NaN / inf")
    return pd.DataFrame({"point_idx": cands_df["point_idx"].to_numpy(), "arc_id": cands_df["arc_id"].to_numpy(),
                         "prob": np.clip(p, 0.0, 1.0)})


def score_candidates(cands_df, scorer=None):
    """Probability that the vehicle is on each candidate arc.

    IN : DataFrame, one row per (GPS point, candidate arc) with columns
         point_idx, arc_id, dist (m, point->arc), rank, x, y (projected metres), ts (seconds).
         Label-free: it never contains label information. (generate_candidates also attaches the trip
         geometry as cands_df.attrs["trip_data"], which P2's build_features needs.)
    OUT: DataFrame with columns point_idx, arc_id, prob.   (prob sums to 1 per GPS point)

    Order of choice: a model registered with set_model() > scorer ("rf" = P2's random forest, "baseline" =
    DISTANCE-ONLY BASELINE softmax(-dist/sigma)); scorer defaults to PATHSNAP_SCORER, else "rf"."""
    if _MODEL is not None:
        return score_with_model(_MODEL, cands_df)
    scorer = scorer or default_scorer()
    if scorer == "baseline":
        return distance_softmax(cands_df, DEFAULT_SIGMA_M)
    if scorer != "rf":
        raise ValueError(f"scorer must be one of {SCORERS}, got {scorer!r}")
    td = cands_df.attrs.get("trip_data")
    if td is None:
        raise ValueError("scorer='rf' needs the trip geometry (use generate_candidates, or scorer='baseline')")
    return score_rf(td, cands_df)


# --------------------------------------------------------------------------- step 1: candidates
def generate_candidates(gps_df, nodes, arcs):
    """Run P1's candidate_gen.generate on in-memory arrays.
    generate() reads a trip folder (.track/.nodes/.arcs), so we write the arrays to a temporary
    folder in the same tab-separated format and call it unchanged. No .route file is ever written or read."""
    sys.path.insert(0, str(REPO))
    from data_processing.candidate_gen import generate      # P1's code (not edited here)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        np.savetxt(d / "trip.track", np.column_stack([gps_df["lon"], gps_df["lat"], gps_df["t"]]),
                   fmt="%.9f", delimiter="\t")
        np.savetxt(d / "trip.nodes", np.asarray(nodes, dtype=float), fmt="%.9f", delimiter="\t")
        np.savetxt(d / "trip.arcs", np.asarray(arcs, dtype=np.int64), fmt="%d", delimiter="\t")
        td = generate(d)
    c = td.cands.copy()                                       # point_idx, arc_id, dist, rank
    c["x"] = td.xy[c["point_idx"].to_numpy(), 0]              # extra context columns for the model
    c["y"] = td.xy[c["point_idx"].to_numpy(), 1]
    c["ts"] = td.ts[c["point_idx"].to_numpy()]
    c.attrs["trip_data"] = td                                 # geometry for P2's build_features (label-free)
    return c


# --------------------------------------------------------------------------- the API function
def route_for_inputs(gps_df, nodes, arcs, config_path=DEFAULT_CONFIG, model=None, scorer=None):
    """Candidates -> scores -> HMM/Viterbi. See the module docstring for the contract.
    model (optional): a predict_proba-style object used instead of everything else.
    scorer (optional): "rf" (default, P2's model) or "baseline" (distance-only); default from PATHSNAP_SCORER."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = REPO / config_path                         # allow calls from any working directory
    from .viterbi import load_config
    cfg = load_config(cfg_path)

    gps = pd.DataFrame({"lon": np.asarray(gps_df["lon"], float), "lat": np.asarray(gps_df["lat"], float),
                        "t": np.asarray(gps_df["t"], float)})
    nodes = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs, dtype=np.int64)

    if len(gps) == 0:
        return [], []
    cands = generate_candidates(gps, nodes, arcs)
    probs = score_with_model(model, cands) if model is not None else score_candidates(cands, scorer) if scorer else score_candidates(cands)
    return decode(gps, nodes, arcs, probs, config=cfg)
