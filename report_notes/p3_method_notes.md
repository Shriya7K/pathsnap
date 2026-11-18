# P3 method notes: HMM / Viterbi route reconstruction

Facts taken from the code (`hmm/graph.py`, `hmm/viterbi.py`, `hmm/evaluate.py`, `hmm/infer.py`,
`hmm/baseline_scores.py`, `hmm/config.json`). No results, no analysis.

## Inputs and interface
- Decoder input: GPS DataFrame (`lon`, `lat`, `t`), `nodes` array (N,2), `arcs` array (M,2) (row index = arc id),
  and a probability table with columns `trip_id`, `point_idx`, `arc_id`, `prob` (one row per point and candidate).
- Output of `decode(...)`: `(point_arcs, route)`. `point_arcs` is one arc id (Python int) or `None` per GPS point.
  `route` is the list of arc ids in travel order with consecutive duplicates removed.
- Parameters are read from `hmm/config.json` by `load_config()`.
- Probability table cleaning: rows are filtered to the trip, rows with NaN/inf, out-of-range `point_idx` or
  out-of-range `arc_id` are dropped, duplicate (point, arc) rows keep the highest `prob`.

## Road graph (`hmm/graph.py`)
- One `TripNetwork` per trip. Node coordinates are projected from lon/lat to the UTM zone of the track mean
  (EPSG 326xx north / 327xx south). All distances are in metres in that projection.
- Arc length = straight-line distance between the arc's two projected end nodes.
- Directed arc adjacency: arc A -> arc B when `A.to_node == B.from_node`. Stored as arcs sorted by `from_node`
  with a pointer array (CSR).
- Bounded shortest path: Dijkstra with a binary heap over nodes, edge weight = arc length, stopped at a distance
  limit. Results are cached per start node; limits are rounded up to multiples of 250 m so cached results can be reused.
- `shortest_arc_path(a, b, limit)`: arcs strictly between `a` and `b` on the shortest path from `a.to_node` to
  `b.from_node`, or `None` if none exists within the limit.
- The projected node coordinates can be cached to disk (`hmm/cache/<key>.npz`), validated by a checksum of nodes and arcs.
- `stats()`: node/arc counts, arc length mean/max, zero-length arcs, mean out-degree, weakly and strongly connected
  components (and the share of arcs in the largest one).

## Hidden states
- At each GPS point the hidden state is the candidate arc the vehicle is on.
- The state set of a point is the set of `arc_id`s supplied for that `point_idx` (the arc id is kept with each state).
  Different points can have different numbers of states.

## Emission
- `e_i(k) = log(max(prob_ik, emission_floor)) / temperature`.
- `emission_floor` avoids `log(0)`. `temperature` divides the log-probability.

## Transition (between consecutive points that have candidates)
- Offset of a point on an arc: distance from the arc start to the projection of the point on the arc segment (clipped to the segment).
- `d_gps`: straight-line distance between the two projected GPS points.
- `d_route`:
  - same arc: `|offset_to - offset_from|`;
  - different arc: `(length_from - offset_from) + shortest_road_distance(from.to_node -> to.from_node) + offset_to`.
- Reachable if `d_route <= L` (L defined under gap handling):
  `T = -|d_route - d_gps| / beta_eff - switch_penalty * [arc changes]`, then floored at `-unreach_penalty`.
- Not reachable (no path within L): `T = -unreach_penalty` (a finite penalty, never minus infinity).
- The Newson and Krumm style term is the exponential in `|d_route - d_gps|` with scale `beta`.

## Gap handling
- `dt` = time difference between the two points (if `dt <= 0` or not finite, 1 s is used).
- Allowed path length: `L = min(path_cap_m, max_path_m + gap_scale_mps * dt)`.
- Tolerance: `beta_eff = beta + gap_scale_mps * max(dt - 1, 0)`.
- Restart: if `dt > restart_gap_s` between two consecutive points with candidates, the chain is split and each
  part is decoded as an independent Viterbi segment (no transition across the gap).
- Points with no candidates are skipped (`point_arcs[i] = None`); the next transition is computed from the last
  point that had candidates, with the real time and distance difference.

## Viterbi recurrence (log space)
- `delta_0(k) = e_0(k)`
- `delta_i(k) = e_i(k) + max_j [ delta_{i-1}(j) + T_i(j -> k) ]`
- `psi_i(k) = argmax_j [ delta_{i-1}(j) + T_i(j -> k) ]`
- End state `argmax_k delta_n(k)`, then follow `psi` backwards. Implemented per step as one
  `K_previous x K_current` matrix operation (`viterbi()`, a pure function that takes emissions and a transition callback).

## Route and `fill_path`
- `raw_route`: decoded arcs with consecutive duplicates removed.
- If `fill_path` is true, for each consecutive pair of route arcs that are not directly adjacent
  (`a.to_node != b.from_node`), the shortest-path arcs between them are inserted if a path exists within
  `fill_max_m`; otherwise nothing is inserted and the break remains. Result: `filled_route`.
- `decode()` returns `filled_route` when `fill_path` is true, else `raw_route`. `decode_detailed()` returns both
  plus counters (restarts, empty points, unreachable transitions).

## Parameters (`hmm/config.json`)
| key | unit | role |
|---|---|---|
| `emission_floor` | probability | lower bound before log |
| `temperature` | - | divides the log-probability |
| `beta` | m | scale of `abs(d_route - d_gps)` |
| `switch_penalty` | log units | cost added when the arc changes |
| `unreach_penalty` | log units | cost of an unreachable transition; also the lower bound of reachable scores |
| `max_path_m` | m | base bounded-path distance |
| `gap_scale_mps` | m/s | growth of `L` and `beta_eff` per second of time gap |
| `path_cap_m` | m | upper bound of `L` |
| `restart_gap_s` | s | time gap that restarts the chain |
| `fill_path` | bool | insert shortest-path arcs between non-adjacent route arcs |
| `fill_max_m` | m | maximum length of such a fill path |
- Tuned by random search over `beta`, `switch_penalty`, `unreach_penalty`, `max_path_m`, `temperature`, `gap_scale_mps`
  (`hmm/tune.py`): trial 0 is the current config, every trial is appended to a CSV log with its seed,
  objective = mean over trips of 0.5 * (route F1 + LCS similarity), only `train` and `student_test` trips.
  `--write-config` is the only way the tuner changes `config.json`.

## Evaluation definitions (`hmm/evaluate.py`)
- Classifier-only: arg-max probability per point (ties: smaller arc id), consecutive duplicates removed, no filling.
- Pointwise accuracy: over points with `label_valid == 1` in P1's point file; a `None` prediction counts as wrong.
- Route precision / recall / F1: on the sets of arc ids of predicted route and reference route.
- LCS similarity: `LCS(pred, ref) / max(len(pred), len(ref))`; LCS recall: `LCS / len(ref)`
  (longest common subsequence, bit-parallel implementation).
- Continuity: share of consecutive route arc pairs with `a.to_node == b.from_node`; breaks = number of pairs that are not.
- Reference route (`.route`) and P1's label columns are used only for these metrics and for the dev maps.

## Distance-only baseline score source (`hmm/baseline_scores.py`)
- `prob = softmax(-dist / sigma)` over the candidates of each GPS point, `sigma` in metres (default 20).
- Reads only `point_idx`, `arc_id`, `dist` of P1's candidates.

## API path (`hmm/infer.py`)
- `route_for_inputs(gps_df, nodes, arcs, config_path)`:
  1. P1's label-free `candidate_gen.generate` (arrays written to a temporary trip folder in P1's tab-separated format),
  2. `score_candidates(cands_df)` (single replacement point for P2's model; currently the baseline above),
  3. `decode(...)` with the parameters from `config_path`.
- It reads no reference data and no `.route`.

## Probability file handling (`hmm/probs_io.py`, `hmm/check_probs.py`)
- One loader (`load_probs`) is used by `hmm.evaluate`, `hmm.tune`, `viz.folium_map`, `hmm.check_probs` and `hmm.run_all`.
- Accepted formats: `.csv`, `.csv.gz`, `.parquet`. Required columns: `trip_id`, `point_idx`, `arc_id`, `prob`; optional `partition`, `fold`, `split`; other columns are kept and ignored.
- Always applied and reported: `trip_id` -> zero-padded 8-character string; `point_idx`, `arc_id` -> integers.
- Only with a flag (and printed): `--accept-renames`, `--renormalise` (rescale per point), `--fill-missing` (missing P1 candidates get prob 0), `--drop-duplicates` (keep highest prob), `--allow-partial`.
- Blocking: missing required columns, evaluator_test rows (partition/split column) or trip ids not found in train/student_test, non-integer ids, NaN/inf/out-of-[0,1] probabilities, duplicate (trip, point, arc) rows, expected trips without rows, P1 candidate files missing (check_probs / run_all).
- Reported, not blocking: P1 candidate rows missing from the file, rows not in P1's candidates, points whose probabilities do not sum to 1 (tolerance 1e-3), all-zero points, single-candidate points.
- Candidate set = P1's `data_processing/load_candidates.load_trip` (K=30, R=100 m defaults), reduced to `point_idx`, `arc_id`.
- Trips with zero candidates in P1's files may have no rows; they are evaluated with an empty probability table (no point gets an arc, empty route).

## Out-of-fold handling
- `hmm.tune` (and `run_all` with `--tune-trials > 0`) refuses to include train trips unless `--probs-are-oof` / `--oof` is given.
- A `fold` column (non-negative for all train rows) or a `split` value containing "oof" is detected as "declares out-of-fold"; a `split` value such as "in_sample" is detected as "declares in-sample", which is refused even with the flag.
- Evaluation CSVs have a `prob_kind` column per trip: train out-of-fold (declared) / train unverified / student_test held-out.

## One-command runner (`hmm/run_all.py`)
- Steps: check_probs, evaluate (all selected trips), diagnose, tune (optional), re-evaluate with tuned parameters, maps (+PNG) for 5 student_test trips, latency.
- Output folder: `results/hmm_<label_slug>/`; an existing non-empty folder is not reused without `--overwrite`; the label `baseline` is reserved.
- `hmm/config.json` is written only with `--write-config` and `--tune-trials > 0`; tuned parameters are always saved to `tuned_config.json` in the output folder.
- Stops at the first blocking error (exit code 1; 2 for the out-of-fold refusal).

## Diagnostics (`hmm/diagnose.py`)
- Point outcome per labelled point (`label_valid`): correct, twin (same street, opposite direction: from/to nodes swapped), adjacent (directly connected to the labelled arc), other, none.
- Route metrics with and without `fill_path` insertions; extra route arcs split into reverse twins of reference arcs, arcs inserted by `fill_path`, and other decoded arcs.
- Share of HMM-wrong points whose chosen arc is closer (P1 candidate `dist`) than the labelled arc.

## Model object for the API (`hmm/infer.py`)
- `set_model(model)` registers an object with `predict_proba(cands_df)`; `cands_df` columns: `point_idx`, `arc_id`, `dist`, `rank`, `x`, `y`, `ts`. Output: shape `(n,)` or `(n, 2)` (column 1 used); NaN/inf or wrong length raises `ValueError`; values are clipped to [0, 1].
- `route_for_inputs(..., model=...)` uses a model for one call; without any model `score_candidates` is the distance-only baseline.

## Maps (`viz/folium_map.py`)
- Layers: GPS points (blue, thinned to at most 3000), road network near the track (grey, bounding box plus buffer,
  undirected duplicates removed, capped at 25000 segments), reference route (green, only for `train` / `student_test`),
  classifier-only route (orange), HMM route (red). Legend, layer control, bounds fitted to the track.
- PNG export uses a headless Edge/Chrome if one is installed.

## Guards
- `train` and `student_test` are the only partitions the development scripts accept; `evaluator_test` is refused
  by `hmm.graph`, `hmm.evaluate`, `hmm.tune`, `hmm.baseline_scores`, `hmm.fake_scores` and the map CLI.
