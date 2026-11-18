"""The ONE loader/validator for classifier probability files (Person 3).

hmm.evaluate, hmm.tune, viz.folium_map, hmm.check_probs and hmm.run_all all call load_probs(), so
validation and dtype fixing happen in exactly one place.

File format (csv, csv.gz or parquet): one row per (GPS point, candidate arc)
    trip_id, point_idx, arc_id, prob            required
    partition, fold, split, <anything else>     optional (extra columns are kept, ignored by the decoder)

What is ALWAYS normalised (and reported): trip_id -> zero-padded 8-char string (54 -> "00000054"),
point_idx / arc_id -> int64, prob -> float.
What changes data ONLY behind an explicit flag (and is printed): accept_renames, renormalise,
fill_missing, drop_duplicates.
BLOCKING problems raise ProbsError (CLIs exit non-zero): missing columns, evaluator_test rows or unknown
trips, non-integer ids, NaN/inf/out-of-range probabilities, duplicates, missing expected trips.
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import DATA_ROOT, DEV_PARTITIONS, REPO, norm_trip_id

CANDS_DIR = REPO / "results" / "candidates"
REQUIRED = ["trip_id", "point_idx", "arc_id", "prob"]
# column names P2 might plausibly use instead of ours (suggested, applied only with --accept-renames)
ALIASES = {
    "prob": ["proba", "probability", "p", "pred_prob", "y_prob", "p_true", "prob_1", "predicted_prob"],
    "arc_id": ["candidate_arc", "candidate_arc_id", "arc", "arcid", "cand_arc"],
    "trip_id": ["trip", "tripid", "trip_name", "trip_no"],
    "point_idx": ["point", "pt_idx", "point_id", "gps_idx", "point_index", "obs_idx"],
}
SUM_TOL = 1e-3
INSAMPLE_WORDS = ("insample", "in_sample", "in-sample", "fit", "trained")   # split values meaning "not out-of-fold"
OOF_WORDS = ("oof", "out_of_fold", "out-of-fold")


class ProbsError(Exception):
    """Blocking problem(s) with a probability file. .problems is the list of messages."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(f"  - {p}" for p in self.problems))


@dataclass
class LoadedProbs:
    df: pd.DataFrame                    # trip_id, point_idx, arc_id, prob (+ any extra columns)
    by_trip: dict                       # trip_id -> DataFrame (includes empty frames for trips with no candidates)
    empty_trips: set                    # trips for which P1 has zero candidates (valid to have no rows)
    trips: list                         # [(partition, trip_id)] expected trips that are present
    report: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    changes: list = field(default_factory=list)
    text: str = ""
    declares_oof: bool = False          # file has a fold column / split value saying train rows are out-of-fold
    declares_insample: bool = False     # file says train rows are in-sample


def _norm_id(x):
    """54, 54.0, '54', '00000054' -> '00000054'; anything else is returned unchanged (and rejected later)."""
    t = str(x).strip()
    if t.replace('.', '', 1).isdigit() and t.count('.') <= 1 and float(t).is_integer():
        return str(int(float(t))).zfill(8)
    return t


# --------------------------------------------------------------------------- reading
def read_raw(path):
    p = Path(path)
    if not p.exists():
        raise ProbsError([f"file not found: {p}"])
    try:
        if p.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(p)
        # trip ids are read as text so '00000054' keeps its meaning; .csv and .csv.gz (compression inferred)
        return pd.read_csv(p, dtype={c: str for c in ["trip_id"] + ALIASES["trip_id"]})
    except ImportError as e:
        raise ProbsError([f"cannot read {p.name}: {e}. Install pyarrow for parquet (pip install pyarrow)."])
    except Exception as e:                          # corrupt / wrong format
        raise ProbsError([f"cannot read {p.name} as csv/csv.gz/parquet: {e}"])


def _suggest_renames(columns):
    """{missing_required_column: [matching existing columns]} using ALIASES (case-insensitive)."""
    low = {c.lower().strip(): c for c in columns}
    out = {}
    for req in REQUIRED:
        if req in columns:
            continue
        out[req] = [low[a] for a in ALIASES[req] if a in low]
    return out


# --------------------------------------------------------------------------- the loader
def load_probs(path, partitions=DEV_PARTITIONS, data_root=DATA_ROOT, cands_dir=CANDS_DIR,
               accept_renames=False, renormalise=False, fill_missing=False, drop_duplicates=False,
               allow_partial=False, candidates="require"):
    """Read, normalise and validate a probability file. See the module docstring.

    candidates: "require" (candidate files must exist), "auto" (check them if they exist, else warn),
                "skip" (no candidate cross-check)."""
    problems, warns, changes, rep = [], [], [], {}
    partitions = list(partitions)
    for part in partitions:
        if part not in DEV_PARTITIONS:
            raise ProbsError([f"partition '{part}' is refused: development code only handles {DEV_PARTITIONS} "
                              f"(evaluator_test is never read)."])

    df = read_raw(path)
    rep["rows_in_file"] = len(df)
    rep["columns"] = list(df.columns)

    # ---- columns
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sugg = _suggest_renames(df.columns)
        fixable = all(len(sugg[m]) == 1 for m in missing)
        if accept_renames and fixable:
            ren = {sugg[m][0]: m for m in missing}
            df = df.rename(columns=ren)
            changes.append(f"renamed columns {ren} (--accept-renames)")
        else:
            hint = "; ".join(f"'{m}' <- {sugg[m] if sugg[m] else 'no similar column found'}" for m in missing)
            msg = (f"missing required column(s) {missing}. Columns in file: {list(df.columns)}. "
                   f"Likely renames: {hint}.")
            if fixable and not accept_renames:
                msg += " Re-run with --accept-renames to apply them, or ask P2 to use the agreed names."
            raise ProbsError([msg])

    # ---- evaluator_test guard (partition / split columns)
    for col in ("partition", "split"):
        if col in df.columns:
            vals = df[col].astype(str).str.lower()
            if vals.str.contains("evaluator").any():
                n = int(vals.str.contains("evaluator").sum())
                raise ProbsError([f"{n} rows are labelled evaluator_test (column '{col}'). evaluator_test is never "
                                  f"used in development; remove those rows."])

    # ---- dtype normalisation
    bad_id = {}
    for c in ("point_idx", "arc_id"):
        num = pd.to_numeric(df[c], errors="coerce")
        bad = num.isna() | (num != np.floor(num.fillna(0)))
        if bad.any():
            bad_id[c] = int(bad.sum())
        df[c] = num
    if bad_id:
        raise ProbsError([f"non-integer / missing values in id columns: {bad_id}"])
    df["point_idx"] = df["point_idx"].astype(np.int64)
    df["arc_id"] = df["arc_id"].astype(np.int64)
    if (df["point_idx"] < 0).any() or (df["arc_id"] < 0).any():
        raise ProbsError(["negative point_idx or arc_id values"])
    tid = df["trip_id"]
    if tid.isna().any():
        raise ProbsError([f"{int(tid.isna().sum())} rows with empty trip_id"])
    original = tid.astype(str).str.strip()
    df["trip_id"] = tid.map(_norm_id)
    n_changed = int((original != df["trip_id"]).sum())
    if n_changed:
        changes.append(f"trip_id normalised to zero-padded 8-char strings in {n_changed} rows "
                       f"(file dtype {tid.dtype}; always applied)")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce").astype(float)

    # ---- expected trips (from the data folders; only dev partitions are ever listed)
    expected, part_of = {}, {}
    for part in DEV_PARTITIONS:
        d = Path(data_root) / part
        ids = sorted(x.name for x in d.iterdir() if x.is_dir()) if d.exists() else []
        expected[part] = ids
        for t in ids:
            part_of[t] = part
    known = set(part_of)
    unknown = sorted(set(df["trip_id"]) - known)
    if unknown:
        problems.append(f"{len(unknown)} trip_id(s) are not in train/student_test under {data_root} "
                        f"(possibly evaluator_test, which is refused): {unknown[:10]}{'...' if len(unknown) > 10 else ''}")
        raise ProbsError(problems)

    if "partition" in df.columns:                      # declared partition must agree with where the trip lives
        wrong = df[df["partition"].astype(str).str.lower() != df["trip_id"].map(part_of)]
        if len(wrong):
            raise ProbsError([f"'partition' column disagrees with the data folders for {wrong['trip_id'].nunique()} "
                              f"trip(s), e.g. {wrong['trip_id'].iloc[0]}"])

    # rows of trips outside the selected partitions are ignored (selection, reported)
    sel_trips = {t for p in partitions for t in expected[p]}
    outside = ~df["trip_id"].isin(sel_trips)
    if outside.any():
        changes.append(f"ignored {int(outside.sum())} rows of {df.loc[outside, 'trip_id'].nunique()} trips outside "
                       f"the selected partitions {partitions}")
        df = df[~outside].copy()

    # ---- value problems
    n_nan = int((~np.isfinite(df["prob"])).sum())
    if n_nan:
        problems.append(f"{n_nan} rows have NaN / inf / non-numeric prob")
    fin = df["prob"][np.isfinite(df["prob"])]
    n_range = int(((fin < 0) | (fin > 1 + 1e-9)).sum())
    if n_range:
        problems.append(f"{n_range} rows have prob outside [0, 1] (min {fin.min():.4g}, max {fin.max():.4g}); "
                        f"are these logits / scores instead of probabilities?")
    key = ["trip_id", "point_idx", "arc_id"]
    dup = df.duplicated(key, keep=False)
    rep["duplicate_rows"] = int(dup.sum() - df[dup].drop_duplicates(key).shape[0]) if dup.any() else 0
    if dup.any():
        if drop_duplicates:
            before = len(df)
            df = df.sort_values("prob", ascending=False, kind="stable").drop_duplicates(key, keep="first")
            changes.append(f"dropped {before - len(df)} duplicate (trip, point, arc) rows keeping the highest prob "
                           f"(--drop-duplicates)")
        else:
            problems.append(f"{rep['duplicate_rows']} duplicate (trip_id, point_idx, arc_id) rows; "
                            f"fix in the file or pass --drop-duplicates (keeps the highest prob)")

    # ---- fold / split declarations about out-of-fold-ness of TRAIN trips
    tr = df[df["trip_id"].map(part_of) == "train"]
    declares_oof = declares_in = False
    if len(tr):
        if "fold" in tr.columns and tr["fold"].notna().all() and (pd.to_numeric(tr["fold"], errors="coerce") >= 0).all():
            declares_oof = True
        if "split" in tr.columns:
            s = tr["split"].astype(str).str.lower()
            if s.map(lambda v: any(w in v for w in OOF_WORDS)).all():
                declares_oof = True
            if s.map(lambda v: any(w in v for w in INSAMPLE_WORDS)).any():
                declares_in = True
    rep["train_rows"] = int(len(tr))

    # ---- candidate cross-check against P1 (label-free columns only)
    empty_trips, present = set(), sorted(set(df["trip_id"]))
    cand_ok = False
    if candidates != "skip":
        cdir = Path(cands_dir)
        have = cdir.exists() and any(cdir.glob("*_cands.csv.gz"))
        if not have:
            msg = (f"P1 candidate files not found in {cdir}; cannot check candidate coverage "
                   f"(generate them with: python data_processing/candidates.py <data_root>)")
            if candidates == "require":
                problems.append(msg)
            else:
                warns.append(msg)
        else:
            cand_ok = True

    if cand_ok:
        cand_parts = []
        for t in sorted(sel_trips):
            c = _read_cands(part_of[t], t, cdir)
            if c is None:
                warns.append(f"no candidate file for {part_of[t]}/{t}; skipped in the coverage check")
                continue
            c = c.assign(trip_id=t)
            if c.empty:
                empty_trips.add(t)
            cand_parts.append(c)
        if not cand_parts:
            warns.append("no usable candidate files for the selected trips; candidate coverage not checked")
            cand_ok = False
    if cand_ok:
        cands = pd.concat(cand_parts, ignore_index=True)
        cands = cands.astype({"point_idx": np.int64, "arc_id": np.int64})   # empty CSVs come back as object columns
        checked = set(cands["trip_id"]) | empty_trips
        d_chk = df[df["trip_id"].isin(checked)]
        cands = cands.assign(_c=1)
        m = d_chk[key].assign(_p=1).merge(cands[key + ["_c"]], on=key, how="outer", indicator=True)
        n_cand = int(len(cands))
        n_missing = int((m["_merge"] == "right_only").sum())
        n_extra = int((m["_merge"] == "left_only").sum())
        rep.update({"candidate_rows_expected": n_cand, "candidate_rows_missing": n_missing,
                    "candidate_rows_missing_pct": round(100 * n_missing / max(n_cand, 1), 3),
                    "rows_not_in_candidates": n_extra})
        if n_extra:
            warns.append(f"{n_extra} rows are not in P1's candidate set (extra rows; the decoder will still use "
                         f"them if the arc id is valid)")
        if n_missing:
            warns.append(f"{n_missing} of {n_cand} P1 candidate rows ({rep['candidate_rows_missing_pct']}%) are "
                         f"missing from the file" + ("" if fill_missing else
                         " (the decoder cannot choose them; --fill-missing adds them with prob 0)"))
        # points that have P1 candidates but zero rows in the file
        pts_c = cands[["trip_id", "point_idx"]].drop_duplicates()
        pts_p = d_chk[["trip_id", "point_idx"]].drop_duplicates().assign(_p=1)
        lost = pts_c.merge(pts_p, on=["trip_id", "point_idx"], how="left")
        rep["points_without_any_row"] = int(lost["_p"].isna().sum())
        rep["points_with_candidates"] = int(len(pts_c))
        if fill_missing and n_missing:
            add = m[m["_merge"] == "right_only"][key].assign(prob=0.0)
            df = pd.concat([df, add], ignore_index=True)
            changes.append(f"added {len(add)} missing candidate rows with prob 0 (--fill-missing)")

    # ---- trips present vs expected
    pres_set = set(df["trip_id"])
    trips_rep, miss_trips = {}, []
    for part in partitions:
        exp = expected[part]
        got = [t for t in exp if t in pres_set or t in empty_trips]
        trips_rep[part] = (len(got), len(exp))
        miss_trips += [(part, t) for t in exp if t not in pres_set and t not in empty_trips]
    rep["trips_present_vs_expected"] = trips_rep
    rep["missing_trips"] = [f"{p}/{t}" for p, t in miss_trips]
    if miss_trips and not allow_partial:
        problems.append(f"{len(miss_trips)} expected trip(s) have no rows: "
                        f"{rep['missing_trips'][:10]}{'...' if len(miss_trips) > 10 else ''} "
                        f"(use --allow-partial for a deliberate subset)")
    if len(df) == 0 and not empty_trips:
        problems.append("no usable rows after validation")

    # ---- probability sums per point (before any renormalisation)
    finite = df[np.isfinite(df["prob"])]
    sums = finite.groupby(["trip_id", "point_idx"])["prob"].transform("sum") if len(finite) else pd.Series(dtype=float)
    pt_sum = finite.assign(_s=sums).drop_duplicates(["trip_id", "point_idx"])["_s"] if len(finite) else pd.Series(dtype=float)
    rep["points_total"] = int(len(pt_sum))
    rep["points_sum_not_1"] = int((np.abs(pt_sum - 1) > SUM_TOL).sum())
    rep["points_all_zero"] = int((pt_sum == 0).sum())
    rep["points_single_candidate"] = int((finite.groupby(["trip_id", "point_idx"]).size() == 1).sum()) if len(finite) else 0
    if rep["points_sum_not_1"]:
        warns.append(f"{rep['points_sum_not_1']} of {rep['points_total']} points have probabilities that do not sum "
                     f"to 1 (tolerance {SUM_TOL})" + ("" if renormalise else " (--renormalise rescales per point)"))
    if rep["points_all_zero"]:
        warns.append(f"{rep['points_all_zero']} points have all-zero probabilities (cannot be renormalised; the decoder "
                     f"treats their candidates as equally unlikely)")
    if renormalise and len(finite):
        s = df.groupby(["trip_id", "point_idx"])["prob"].transform("sum")
        need = (np.abs(s - 1) > SUM_TOL) & (s > 0)
        if need.any():
            df.loc[need, "prob"] = df.loc[need, "prob"] / s[need]
            changes.append(f"renormalised {int(df.loc[need].drop_duplicates(['trip_id', 'point_idx']).shape[0])} points "
                           f"to sum to 1 (--renormalise)")

    if problems:
        raise ProbsError(problems)

    df = df.sort_values(["trip_id", "point_idx", "arc_id"], kind="stable").reset_index(drop=True)
    by_trip = {k: g for k, g in df.groupby("trip_id")}
    empty_frame = df.iloc[0:0]
    for t in empty_trips:
        by_trip.setdefault(t, empty_frame)
    trips = [(part_of[t], t) for t in sorted(by_trip, key=lambda x: (DEV_PARTITIONS.index(part_of[x]), x))
             if part_of[t] in partitions]
    out = LoadedProbs(df=df, by_trip=by_trip, empty_trips=empty_trips, trips=trips, report=rep,
                      warnings=warns, changes=changes, declares_oof=declares_oof, declares_insample=declares_in)
    out.text = format_report(out, path)
    return out


def _read_cands(partition, trip, cdir):
    """P1's candidates for one trip via data_processing/load_candidates.py, reduced to the label-free
    key columns. None if the file does not exist."""
    import sys
    f = Path(cdir) / f"{partition}_{trip}_cands.csv.gz"
    if not f.exists() or not (Path(cdir) / f"{partition}_{trip}_points.csv.gz").exists():
        return None
    sys.path.insert(0, str(REPO / "data_processing"))
    import load_candidates as lc                       # P1's loader (not edited)
    lc.CAND_DIR = Path(cdir)
    return lc.load_trip(partition, trip)[["point_idx", "arc_id"]]   # label columns dropped immediately


def format_report(lp, path=""):
    r = lp.report
    L = [f"=== probability file check: {path} ===",
         f"rows in file: {r['rows_in_file']}   columns: {r['columns']}",
         "trips present vs expected: " + ", ".join(f"{p} {a}/{b}" for p, (a, b) in r["trips_present_vs_expected"].items())]
    if r["missing_trips"]:
        L.append(f"missing trips: {r['missing_trips']}")
    if "candidate_rows_expected" in r:
        L.append(f"P1 candidate rows: {r['candidate_rows_expected']}  missing from file: {r['candidate_rows_missing']} "
                 f"({r['candidate_rows_missing_pct']}%)  extra rows not in candidates: {r['rows_not_in_candidates']}  "
                 f"points with candidates but no rows: {r['points_without_any_row']}/{r['points_with_candidates']}")
    L.append(f"duplicates: {r['duplicate_rows']}   points: {r['points_total']}   sum != 1 (tol {SUM_TOL}): "
             f"{r['points_sum_not_1']}   all-zero: {r['points_all_zero']}   single-candidate: {r['points_single_candidate']}")
    if lp.empty_trips:
        L.append(f"trips with no candidates at all in P1's files (no rows needed): {sorted(lp.empty_trips)}")
    L.append(f"train trips: file declares out-of-fold = {lp.declares_oof}, declares in-sample = {lp.declares_insample}")
    for c in lp.changes:
        L.append(f"CHANGED: {c}")
    for w in lp.warnings:
        L.append(f"WARNING: {w}")
    L.append("result: OK (no blocking problems)")
    return "\n".join(L)


# --------------------------------------------------------------------------- shared CLI plumbing
def add_args(ap):
    """Flags shared by every script that loads a probability file."""
    g = ap.add_argument_group("probability file handling (hmm/probs_io.py)")
    g.add_argument("--accept-renames", action="store_true", help="apply obvious column renames (proba -> prob, ...)")
    g.add_argument("--renormalise", action="store_true", help="rescale each point's probs to sum to 1")
    g.add_argument("--fill-missing", action="store_true", help="add P1 candidates missing from the file with prob 0")
    g.add_argument("--drop-duplicates", action="store_true", help="keep the highest prob of duplicate rows")
    g.add_argument("--allow-partial", action="store_true", help="allow expected trips to have no rows (dev subsets)")
    g.add_argument("--probs-are-oof", action="store_true",
                   help="train-trip probs are out-of-fold (or from a non-learned baseline), i.e. NOT from a model fitted on them")
    return ap


def load_from_args(a, partitions, data_root=DATA_ROOT, cands_dir=CANDS_DIR, candidates="auto"):
    """load_probs() with the shared flags; prints the report/changes; exits 1 on blocking problems."""
    try:
        lp = load_probs(a.probs, partitions, data_root, cands_dir, accept_renames=a.accept_renames,
                        renormalise=a.renormalise, fill_missing=a.fill_missing, drop_duplicates=a.drop_duplicates,
                        allow_partial=a.allow_partial, candidates=candidates)
    except ProbsError as e:
        print(f"BLOCKING PROBLEM(S) with {a.probs}:\n{e}")
        raise SystemExit(1)
    for c in lp.changes:
        print(f"CHANGED: {c}")
    for w in lp.warnings:
        print(f"WARNING: {w}")
    return lp


def require_oof_for_train(partitions, probs_are_oof, lp):
    """Tuning guard: returns None if fine, otherwise the refusal message."""
    if "train" not in partitions:
        return None
    if lp.declares_insample:
        return ("the file declares its train-trip probabilities as in-sample (split column). Tuning on in-sample "
                "probabilities fits the decoder to an over-confident classifier. Ask P2 for out-of-fold probabilities.")
    if not probs_are_oof:
        extra = (" The file has a fold/split column that looks out-of-fold; if so, confirm with --probs-are-oof."
                 if lp.declares_oof else "")
        return ("train trips are included but --probs-are-oof was not given. A classifier's probabilities on its own "
                "training trips are over-confident, so tuning on them is not valid. Pass --probs-are-oof ONLY if the "
                "train probabilities are out-of-fold (or from a non-learned baseline), or tune with "
                "--partitions student_test." + extra)
    return None
