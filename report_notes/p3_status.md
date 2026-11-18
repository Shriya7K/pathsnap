# P3 status (HMM and maps)

Facts only. Updated 2026-11-18. Numbers are in report_notes/p3_results.md and the result files named there.

## Implemented (hmm/, viz/, tests/test_hmm_*.py)
| file | purpose |
|---|---|
| hmm/graph.py, hmm/viterbi.py | road graph, HMM decoder (emission, transition, gaps, Viterbi, fill_path) |
| hmm/config.json | decoder parameters; TUNED 2026-11-18 (see its `_meta`); defaults saved in results/hmm_default_config.json |
| hmm/evaluate.py, hmm/diagnose.py | per-trip metrics; error categories, fill_path effect |
| hmm/tune.py | random-search tuner (objective f1 or f1_lcs, logs raw F1 / precision / recall / pointwise accuracy per trial, refuses train trips unless OOF) |
| hmm/probs_io.py, check_probs.py | shared probability loader and validator |
| hmm/import_p2.py, merge_p2.py | P2's per-trip files -> one probability file (+ `fold` column) |
| hmm/p2_oof.py | 5-fold out-of-fold probabilities of P2's random forest for the 80 train trips |
| hmm/run_all.py, compare_models.py, compare_configs.py | one-command runner; model comparison table; default-vs-tuned comparison on unseen trips |
| hmm/infer.py | `route_for_inputs(gps_df, nodes, arcs, config_path, model=None, scorer=None)`; default scorer = P2's random forest, baseline via `scorer="baseline"` / `PATHSNAP_SCORER=baseline` |
| hmm/latency.py | per-trip inference timing (candidates / features / model / decode) |
| hmm/baseline_scores.py | distance-only baseline scores (no ML) |
| hmm/fixture_trips.py, fake_scores.py | SYNTHETIC debugging data only |
| viz/folium_map.py, viz/render_selected.py | Folium maps + PNG; rule-based choice of 5 student_test maps |
| tests/test_hmm_*.py | unit / integration tests |

## Result files (details and tables: report_notes/p3_results.md)
- results/hmm_graph_stats.csv; distance-only baseline: results/hmm_baseline_comparison.csv, results/hmm_baseline_tuning_log.csv (4-trial smoke test), results/maps/*distance_only_baseline*
- random forest, default config: results/hmm_rf_default_comparison.csv, results/hmm_random_forest_p2/ (summary, diagnose, latency), results/hmm_rf_default_run.log, results/hmm_rf_check_probs.txt
- comparison of systems: results/hmm_model_comparison.csv
- tuning: results/hmm_rf_tuning_log.csv, results/hmm_rf_tuning_trips.csv, results/hmm_rf_tune_run.log, results/hmm_rf_tuning_timing_one_pass.csv
- default vs tuned on unseen trips: results/hmm_rf_tuned_vs_default.csv / .txt
- latency: results/hmm_latency.csv / .txt
- maps (tuned): results/maps/student_test_<trip>_random_forest__p2__tuned.html/.png, results/maps/selection_rf_tuned.csv
- OOF: results/p2_oof_meta/oof_fold_top1.csv, folds.csv; stage logs: results/logs_stageA/

## Not committed (git-ignored or large; regenerate with the README commands)
models/final_model.joblib, results/predictions/, results/p2/, results/p2_oof/, results/p2_oof_features/, results/hmm_synthetic/, hmm/cache/, results/hmm_baseline/baseline_probs.csv.gz, large HTML maps, data/.

## Open items
1. models/final_model.joblib is rebuilt locally by P2's script; the metadata matches the committed one exactly, but the original joblib was never in git.
2. Tuned config lowers pointwise accuracy on most unseen trips while raising route F1 / LCS (report_notes/p3_results.md section 6).
3. Only one tuning run (30 trials, seed 0, 30 trips) exists.
