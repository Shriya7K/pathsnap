# HMM / Viterbi route reconstruction (Person 3)

The classifier scores each (GPS point, candidate arc) independently, so its best guesses can
jump between unconnected roads. The HMM adds two things the classifier does not have: **time order**
and **road-network connectivity**. Viterbi then picks the *single best sequence* of arcs.

## Model

**Hidden states.** At GPS point *i* the hidden state is "which candidate arc the vehicle is on".
The states are exactly the candidate arcs P1/P2 supply for that point (their `arc_id` is kept,
so decoded states map straight back to arc ids). Different points have different numbers of states.

**Emission** (how well a state explains the observation): the classifier probability.

    e_i(k) = log( max(prob_ik, emission_floor) ) / temperature

The floor (1e-6) avoids log 0. `temperature > 1` softens over-confident classifiers.

**Transition** between consecutive points *i-1 -> i*, from arc *a* (point offset o_a) to arc *b* (offset o_b):

* `d_gps`  = straight-line distance between the two GPS points (UTM metres)
* `d_route` = road distance between the two projected positions:
  same arc: `|o_b - o_a|`; otherwise `(len_a - o_a) + shortest_road_path(a.to_node -> b.from_node) + o_b`
* Newson & Krumm idea: a vehicle's road distance should be close to the straight-line distance.

      T(a->b) = - |d_route - d_gps| / beta_eff  -  switch_penalty * [a != b]        if d_route <= L
      T(a->b) = - unreach_penalty                                                   otherwise (unreachable)

  and reachable scores are floored at `-unreach_penalty`. Unreachable is a *small non-zero
  probability* (e^-12 by default), never zero, so the chain can never dead-end. Staying on the same
  arc is naturally the best score (no switch penalty, d_route ~ d_gps).

**Gap handling.** `dt` = time between the two points.
* allowed path length `L = min(path_cap_m, max_path_m + gap_scale_mps * dt)` and tolerance
  `beta_eff = beta + gap_scale_mps * (dt - 1)`: a longer gap allows a longer path and a looser match.
* `dt > restart_gap_s`: the chain is **restarted** (a new independent Viterbi segment; no transition
  is assumed across the gap).
* A point with **no candidates** is skipped (`point_arcs[i] = None`); the transition goes from the
  last point that had candidates, using the real time difference. Rows with unknown arc ids, NaN or
  out-of-range point indices are dropped; duplicate (point, arc) rows keep the max probability.

**Viterbi recurrence** (log space, so products become sums):

    delta_0(k)   = e_0(k)
    delta_i(k)   = e_i(k) + max_j [ delta_{i-1}(j) + T_i(j -> k) ]
    psi_i(k)     = argmax_j [ delta_{i-1}(j) + T_i(j -> k) ]          (back-pointer)
    best end     = argmax_k delta_n(k);  then follow psi backwards to get the arc of every point.

Per step this is one `K_prev x K_cur` matrix operation; Dijkstra results are cached per start node
(consecutive points share candidate arcs), so a 19k-point trip takes ~17 s.

**Route and connectivity.** The route = decoded arcs with consecutive duplicates removed (`raw_route`).
Because points are 1 s apart and arcs can be short, consecutive decoded arcs are sometimes not
*directly* adjacent. With `fill_path = true` the shortest-path arcs between them are inserted
(`filled_route`), **only if a road path exists within `fill_max_m`**; otherwise the break stays and is
counted. `evaluate.py` reports continuity for both (`hmm_raw_*`, `hmm_filled_*`).

## Parameters (`hmm/config.json`, loaded at inference by `load_config()`)

| key | meaning | default* |
|---|---|---|
| `emission_floor` | min probability before log | 1e-6 |
| `temperature` | divides the log-probability (>1 flattens) | 1.0 |
| `beta` (m) | tolerance of `abs(d_route - d_gps)` | 10 |
| `switch_penalty` | extra log-cost for changing arc | 0.5 |
| `unreach_penalty` | log-cost of an unreachable transition (and floor) | 12 |
| `max_path_m` (m) | base bounded-path distance | 200 |
| `gap_scale_mps` (m/s) | widening of `L` and `beta` per second of gap | 30 |
| `path_cap_m` (m) | hard cap on `L` | 4000 |
| `restart_gap_s` (s) | gap that restarts the chain | 600 |
| `fill_path`, `fill_max_m` | insert shortest-path arcs between disconnected decoded arcs | true, 3000 |

\*The shipped values are **untuned defaults** (`_meta.status` in the JSON says so) until
`python -m hmm.tune ... --write-config` is run on P2's real probabilities.
`beta, switch_penalty, unreach_penalty, max_path_m, temperature, gap_scale_mps` are tuned.

## Interface

```python
from hmm.viterbi import decode
point_arcs, route = decode(gps_df, nodes, arcs, probs_df, trip_id="00000001")
# gps_df: lon, lat, t   nodes: (N,2)   arcs: (M,2)   probs_df: trip_id, point_idx, arc_id, prob
# point_arcs: list[int|None] per GPS point;  route: list[int] arc ids, consecutive duplicates removed
```
Pass `cache_key="<partition>_<trip>"` (or a prebuilt `TripNetwork` via `network=`) to reuse the projected network.
`decode_detailed` returns the same plus `raw_route`, `filled_route` and diagnostics.

## Reproduce

```bash
python -m pytest tests -q                                           # unit tests
python -m hmm.graph --partitions train student_test                 # results/hmm_graph_stats.csv
python -m hmm.evaluate --probs <P2_probs.csv.gz> --partitions train student_test   # results/hmm_comparison.csv
python -m hmm.tune --probs <P2_probs.csv.gz> --n-trials 40 --seed 0 --write-config # results/hmm_tuning_log.csv + config.json
python -m viz.folium_map --probs <P2_probs.csv.gz> --partition student_test --n 5 --png   # results/maps/
```
Synthetic debugging (NOT results): `python -m hmm.fake_scores ...` writes `hmm_synthetic_probs.csv.gz`;
any file with "synthetic" in its name is automatically routed to `results/hmm_synthetic/`, bannered,
and refused by `--write-config`.

## API entry point: `hmm/infer.py` (for P4)

```python
from hmm.infer import route_for_inputs

point_arcs, route = route_for_inputs(gps_df, nodes, arcs, config_path="hmm/config.json")
```

| argument | type | meaning |
|---|---|---|
| `gps_df` | `pandas.DataFrame` with columns `lon`, `lat`, `t` | one row per GPS point, time order (t in seconds) |
| `nodes` | `numpy.ndarray`, shape `(N, 2)` | lon, lat per node (row index = node id) |
| `arcs` | `numpy.ndarray`, shape `(M, 2)` | from_node, to_node per arc (row index = arc id) |
| `config_path` | `str`, default `"hmm/config.json"` | decoder parameters; a relative path is also resolved against the repo root |

Returns `(point_arcs, route)`: `point_arcs` is a `list[int | None]` with one entry per GPS point
(`None` = no candidate arc for that point); `route` is a `list[int]` of arc ids in travel order with
consecutive duplicates removed. Empty `gps_df` returns `([], [])`.

Pipeline: (1) candidates from P1's label-free `data_processing/candidate_gen.py` (it reads a trip
folder, so the arrays are written to a temporary folder in the same format; no `.route` is written or
read) -> (2) `score_candidates(cands_df)` -> (3) `hmm.viterbi.decode`. It reads no reference data.

**Scorer (step 2).** By default `score_candidates` is **P2's random forest**: it loads `models/final_model.joblib` and the
feature list in `models/final_model_metadata.json`, builds the 14 features with P2's own
`data_processing/features.py::build_features` (reused, not copied), takes `predict_proba` of class 1 and renormalises each GPS
point's candidates to sum to 1. The **distance-only baseline** stays selectable:
`route_for_inputs(..., scorer="baseline")` or the environment variable `PATHSNAP_SCORER=baseline` (values `rf` / `baseline`).
Priority: `model=` argument > `set_model(model)` > `scorer` > `PATHSNAP_SCORER` > `rf`.
If the joblib is missing (it is git-ignored, 6 MB) the call raises a `FileNotFoundError` that says how to rebuild it
(`python data_processing/train_final_model.py`) or to use the baseline.
`python -m hmm.latency` times candidates / features / model / decode per trip (`results/hmm_latency.csv`).

## Distance-only baseline (stand-in for P2's classifier)

`hmm/baseline_scores.py`: `prob = softmax(-dist / sigma)` over each GPS point's candidate arcs
(sigma in metres, default 20). It reads only `point_idx, arc_id, dist` from P1's candidates. It is a real,
non-ML score source and is labelled "distance-only baseline" in the CSV (`score_source`), logs and maps.
Because the two directions of a two-way street are at the same distance, the baseline gives them equal
probability, so direction can only come from the HMM transitions.

```bash
python -m hmm.baseline_scores --partitions train student_test --sigma 20   # results/hmm_baseline/baseline_probs.csv.gz
python -m hmm.evaluate --probs results/hmm_baseline/baseline_probs.csv.gz --partitions train student_test        --label "distance-only baseline" --out results/hmm_baseline_comparison.csv
python -m hmm.tune --probs results/hmm_baseline/baseline_probs.csv.gz --label "distance-only baseline"        --max-trips 6 --n-trials 4 --seed 0 --log results/hmm_baseline_tuning_log.csv     # smoke test only; no --write-config
python -m viz.folium_map --probs results/hmm_baseline/baseline_probs.csv.gz --partition student_test --n 5        --label "distance-only baseline" --png
```
(P1's candidate files must exist in `results/candidates/`: `python data_processing/candidates.py data`.)

## Plugging in P2's model (exact steps)

**What to tell P2 (copy-paste):**
> Please send ONE file (`.csv`, `.csv.gz` or `.parquet`) with one row per (GPS point, candidate arc) from P1's
> candidate set (`data_processing/load_candidates.py`, K=30, R=100 m), columns:
> `trip_id` (8-char zero-padded string, e.g. `00000054`; an int is also accepted), `point_idx` (int), `arc_id` (int),
> `prob` (float in [0, 1], the probability that this candidate is the true arc).
> It must cover ALL 80 train and ALL 10 student_test trips and nothing from evaluator_test.
> **Train-trip rows must be OUT-OF-FOLD** (each train trip scored by a model that was not fitted on it, e.g. grouped
> K-fold by trip) because I tune the HMM on them; add a `fold` column (int fold index for train rows, -1 for student_test)
> or `split = "oof"` / `"held_out"`. student_test rows come from the final model. Extra columns are fine.
> For the API please give me an object with `predict_proba(cands_df)` (see below) or tell me how to load your model.

**Then (one command):**
```bash
python -m hmm.check_probs --probs <P2_file>                       # pre-flight: fix anything it lists, or add the fix flags
python -m hmm.run_all --probs <P2_file> --label "<model name>" --oof --tune-trials 40
```
`run_all` = check_probs -> evaluate (90 trips) -> diagnose -> tune (only if `--tune-trials > 0`) -> re-evaluate with the
tuned parameters -> 5 maps + PNGs -> latency, all in `results/hmm_<label_slug>/` (never overwrites; `--overwrite` to reuse a
folder). `hmm/config.json` changes only with `--write-config` (and `--tune-trials > 0`). Add `--tune-max-trips N` for a
quicker tuning subset, `--no-png` if no browser is installed. Tuning needs `--oof` when train trips are included
(or `--tune-partitions student_test`); without `--oof` the run stops before doing any work.

**Fix flags** (nothing is modified unless you pass them, and every change is printed): `--accept-renames`
(proba -> prob, candidate_arc -> arc_id, ...), `--renormalise` (rescale each point's probs to sum to 1),
`--fill-missing` (candidates missing from the file get prob 0), `--drop-duplicates` (keep the highest prob),
`--allow-partial` (deliberate subset of trips). Always applied and reported: `trip_id` -> zero-padded string,
ids -> int.

**Blocking problems** (exit code 1, run stops): missing required columns (error lists likely renames), evaluator_test
rows or trip ids that are not in train/student_test, non-integer ids, NaN/inf or probs outside [0, 1], duplicate rows,
missing expected trips, P1 candidate files missing. **Warnings** (reported, not blocking): candidate rows missing from the file
(% reported), rows not in P1's candidates, sums != 1 (tolerance 1e-3), all-zero points, single-candidate points.
Points with all-zero probabilities, trips with no candidates and single-candidate points are handled without crashing.

**Out-of-fold handling.** `hmm/tune.py` refuses to tune on train trips unless `--probs-are-oof` (`--oof` in run_all) is given,
and refuses even then if the file's `split` column says `in_sample`. A `fold` column or `split = oof` is detected and reported
(`declares out-of-fold`); the flag is still required, as an explicit confirmation. Without the flag, evaluation still runs
but prints a note and marks train rows `prob_kind = "train: UNVERIFIED (maybe in-sample)"` in the CSV.

**For the API (P4):** the file route above is for evaluation and tuning. For serving, P2 provides a model object and P4
registers it once at start-up:
```python
from hmm.infer import set_model, route_for_inputs
set_model(model)      # model.predict_proba(cands_df) -> (n,) or (n, 2) array; cands_df columns: point_idx, arc_id, dist, rank, x, y, ts
point_arcs, route = route_for_inputs(gps_df, nodes, arcs, config_path="hmm/config.json")
```
(or pass `model=` to `route_for_inputs` for a single call). No code in hmm/ changes. Latency of the API path is in
`results/hmm_<label>/latency.csv`; it includes model time only if `set_model` was called before the run.

## Final plug-in steps with P2's random forest (what was actually run; results in `report_notes/p3_results.md`)
```bash
python data_processing/candidates.py data ; git checkout results/candidate_summary.csv   # P1 candidates (label files)
python data_processing/build_training_data.py                                            # P2: training features
python data_processing/train_final_model.py                                              # P2: models/final_model.joblib
python data_processing/build_test_features.py ; python data_processing/predict_test_probabilities.py   # results/predictions/
python -m hmm.p2_oof                                                                     # 5-fold OOF probs for the 80 train trips
python -m hmm.import_p2 --in-dir results/predictions --partition student_test --out results/p2/student_test_probs.csv.gz --expect-trips 10
python -m hmm.import_p2 --in-dir results/p2_oof --partition train --out results/p2/train_oof_probs.csv.gz --expect-trips 80
python -m hmm.merge_p2 --student results/p2/student_test_probs.csv.gz --train-oof results/p2/train_oof_probs.csv.gz --folds results/p2_oof_meta/folds.csv --out results/p2/probs_all.csv.gz
python -m hmm.check_probs --probs results/p2/probs_all.csv.gz --probs-are-oof
python -m hmm.run_all --probs results/p2/probs_all.csv.gz --label "random forest (P2)" --maps-n 0 --no-png        # evaluate + diagnose
python -m hmm.compare_models --baseline results/hmm_baseline_comparison.csv --rf results/hmm_rf_default_comparison.csv --out results/hmm_model_comparison.csv
python -m hmm.tune --probs results/p2/probs_all.csv.gz --partitions train --probs-are-oof --label "random forest (P2)" --objective f1 --max-trips 30 --exclude-trips 00000054 00000033 --seed 0 --n-trials 30 --trips-out results/hmm_rf_tuning_trips.csv --log results/hmm_rf_tuning_log.csv --write-config
python -m hmm.compare_configs --probs results/p2/probs_all.csv.gz --default-config results/hmm_default_config.json --tuned-config hmm/config.json --tuning-trips results/hmm_rf_tuning_trips.csv --out results/hmm_rf_tuned_vs_default.csv
python -m hmm.latency --out results/hmm_latency.csv
python -m viz.folium_map ...   # see report_notes/p3_results.md for the exact map commands
```
`hmm/p2_oof.py` imports P2's `FEATURE_COLUMNS`, `BEST_PARAMS`, `load_training_sample`, `get_trip_files` unchanged. Because
`results/training_features/` keeps only `label_valid == 1` rows, the full label-free feature table of each train trip is rebuilt
with P2's `generate()` + `build_features()` (cached in `results/p2_oof_features/`) so every candidate of a held-out trip is scored.

## Integrity notes
* Ground truth (`.route`, P1's labels) is used only for evaluation and dev maps, never by the decoder.
* Only `train` and `student_test` are touched; every dev script refuses `evaluator_test`.
* Limitations: arcs are treated as straight segments node-to-node; candidate misses cannot be fixed by
  the HMM (it only chooses among supplied candidates); where the reference route does not cover the
  track (e.g. train/00000054: only ~11% of points within 50 m of it) route precision is meaningless
  for that trip; `diag_label_valid_pct` in the comparison table flags this.
