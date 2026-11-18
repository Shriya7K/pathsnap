# P3 results: HMM / Viterbi with P2's random forest

Facts only, no interpretation. Every number below is copied from a saved file named next to it; the commands that produced the files are in the reproduce section. Written 2026-11-18. Reference routes / P1 labels were used only for scoring and dev maps. evaluator_test was never read, tuned on or plotted.

## 1. Rebuild of P2's missing outputs (P2's code run unchanged)
- `candidates.py data`: regenerated 180 files in results/candidates; decompressed content identical to the previous 180 files (180/180). `results/candidate_summary.csv` restored with `git checkout`. Log: results/hmm_stageA1_candidates.log
- `build_training_data.py`: 80 train trips, 3,047,533 valid candidate rows, 185,106 positives (6.07%). Log: results/logs_stageA/A2_build_training_data.log
- `train_final_model.py`: models/final_model.joblib, 6,388,209 bytes. `models/final_model_metadata.json` after the rebuild is **identical** to the committed one (git shows no change): hyperparameters {'n_estimators': 180, 'max_depth': 20, 'min_samples_leaf': 2, 'max_features': 'sqrt'}, training_rows 49940, positive_ratio 0.0631958350020024. (The original joblib is not in git, so its bytes could not be compared; the metadata matches exactly and the API path reproduces the batch predictions, see tests.)
- `build_test_features.py` + `predict_test_probabilities.py`: 10 student_test files, 366,367 predictions. Logs in results/logs_stageA/.

### sha256 (model rebuilt on this machine)
```
d923b22aee4b6921d734c56e052d2a50f5cafdd6c4bdc5a908db8cdbf6ece37c  models/final_model.joblib
232c7440fbac5dab5c870e877258d1ab846b82bac9cd5b6a98462d5de14e667d  results/predictions/student_test_00000000.csv.gz
abe64bc3ef3a01595e2973f24876e14dd7f74ddb32dff79e7f9954393e7ce4f6  results/predictions/student_test_00000007.csv.gz
c89b6c45f3205cc46d4e7e8269dfd5986ba81fa43a2171006969c9f0102f7f04  results/predictions/student_test_00000009.csv.gz
388db263d69719b606f9d7b67817a956bd5363213b5ee5b9e0584edb962ce33f  results/predictions/student_test_00000040.csv.gz
15f9045ceda7156c5e1529e40e1c294cd5eaa6ef7e399fd06915675fd60972fe  results/predictions/student_test_00000052.csv.gz
e3233f3b9d0274311bc5de063be50a46173c60a4f33e7c700234527cd213d799  results/predictions/student_test_00000056.csv.gz
1165d2a0bd2f8b245f5921fcf0182a775e9baefc0b196b2c69ea69feab3a1838  results/predictions/student_test_00000065.csv.gz
2abcb4b728441e73a1b370bfebb4c9548e1bd146fba1da04aeee546e89e74636  results/predictions/student_test_00000076.csv.gz
614ecf1d5117f3d4953152ac3342c1c09c985db458845dc0d60c3800abbeaf06  results/predictions/student_test_00000077.csv.gz
60721085200506ff9eaf346c9a7bc646b43a4633f374ca257918cc7d88441938  results/predictions/student_test_00000091.csv.gz
```
(gzip files embed a timestamp, so these hashes change if the files are regenerated; the decompressed contents are what is reproducible.)

## 2. Out-of-fold probabilities for the 80 train trips (`hmm/p2_oof.py`)
5-fold GroupKFold over the 80 train trips (same splitter call as P2's train_models.py; unshuffled, so deterministic), RandomForest with P2's BEST_PARAMS, class_weight=balanced_subsample, random_state=42, trained on each fold's training trips only; all candidate rows of the held-out trips scored. Asserted: train/held-out disjoint in every fold, each of the 80 trips held out exactly once. Sources: results/p2_oof_meta/oof_fold_top1.csv, results/model_training/fold_results.csv, log results/logs_stageA/A5_p2_oof.log

| fold | heldout_points | OOF Top-1 (this run: 180 trees, depth 20) | P2 fold_results.csv Top-1 (100 trees, depth 18) |
|---|---|---|---|
| 1.0000 | 37619.0000 | 0.8601 | 0.8581 |
| 2.0000 | 33983.0000 | 0.8808 | 0.8802 |
| 3.0000 | 36396.0000 | 0.7774 | 0.7725 |
| 4.0000 | 46353.0000 | 0.6999 | 0.7002 |
| 5.0000 | 30798.0000 | 0.8322 | 0.8336 |

mean OOF Top-1 0.8101; P2 file mean 0.8089.

## 3. Default decoder config, 90 trips (80 train OOF + 10 student_test)
Files: results/hmm_rf_default_comparison.csv (per trip), results/hmm_random_forest_p2/summary.txt (mean/median, improved/tied/worse, worst-10), diagnose_summary.txt, diagnose_by_trip.csv. Probability file check: results/hmm_rf_check_probs.txt (0 missing/extra candidate rows vs P1, file declares out-of-fold, 1 all-zero point made uniform by the importer).
```
=== SUMMARY [random forest (P2)]: 90 trips ===
metric              cls mean  hmm mean   d mean |  cls med  hmm med    d med
point_acc             0.8236    0.9247  +0.1011 |   0.8497   0.9451  +0.0881
precision             0.7430    0.8890  +0.1460 |   0.7506   0.9392  +0.1420
recall                0.8732    0.9847  +0.1114 |   0.8915   0.9940  +0.0950
f1                    0.7940    0.9296  +0.1356 |   0.8074   0.9643  +0.1358
lcs_sim               0.5092    0.8285  +0.3194 |   0.4990   0.9081  +0.3017
lcs_recall            0.8705    0.9843  +0.1138 |   0.8893   0.9936  +0.0988
continuity            0.7104    0.9986  +0.2882 |   0.7432   1.0000  +0.2504
breaks (total)    cls=21615  hmm(raw)=4596  hmm(filled)=64   |  reference route breaks=14
continuity (mean) cls=0.7104  hmm raw=0.8768  hmm filled=0.9986

HMM vs classifier-only, trips improved / tied / worse (tolerance 1e-9):
  point_acc  improved= 90  tied=  0  worse=  0
  f1         improved= 88  tied=  0  worse=  2
  lcs_sim    improved= 86  tied=  0  worse=  4
```
Diagnostics (results/hmm_random_forest_p2/diagnose_summary.txt):
```
=== DIAGNOSE (reference / labels used only to describe errors) ===
point outcomes, mean over trips (label_valid points):   correct   twin  adjacent  other   none
  classifier-only      82.4%      7.4%      2.6%      7.7%      0.0%
  HMM                  92.5%      2.6%      1.9%      3.0%      0.0%
fill_path effect (route metrics, mean over trips):          raw   filled
  precision                              0.9368   0.8890
  recall                                 0.8883   0.9847
  f1                                     0.9093   0.9296
  lcs_sim                                0.8350   0.8285
  breaks (total)                          4596       64
  trips where filling raises F1: 71   lowers F1: 18   unchanged: 1
route arcs not in the reference: 19542 = reverse twins 920 + fill_path insertions 12255 + other decoded 6367
HMM-wrong points whose chosen arc is nearer to the GPS point than the labelled arc (median over trips): 56%
```

## 4. Model comparison, same 90 trips and the 10 student_test trips (default decoder config)
File: results/hmm_model_comparison.csv (mean and median per metric).

**all trips (90 trips), mean**

| system | point_acc | precision | recall | f1 | lcs_sim | continuity |
|---|---|---|---|---|---|---|
| distance-only alone | 0.5783 | 0.5398 | 0.6589 | 0.5874 | 0.3393 | 0.4369 |
| distance-only + HMM | 0.9158 | 0.9294 | 0.9859 | 0.9541 | 0.8936 | 0.9994 |
| random forest alone | 0.8236 | 0.7430 | 0.8732 | 0.7940 | 0.5092 | 0.7104 |
| random forest + HMM | 0.9247 | 0.8890 | 0.9847 | 0.9296 | 0.8285 | 0.9986 |

**student_test (10 trips), mean**

| system | point_acc | precision | recall | f1 | lcs_sim | continuity |
|---|---|---|---|---|---|---|
| distance-only alone | 0.5571 | 0.5388 | 0.6667 | 0.5907 | 0.3386 | 0.4885 |
| distance-only + HMM | 0.9205 | 0.9616 | 0.9902 | 0.9753 | 0.9441 | 1.0000 |
| random forest alone | 0.8403 | 0.7565 | 0.8685 | 0.8031 | 0.5144 | 0.7285 |
| random forest + HMM | 0.9255 | 0.8894 | 0.9875 | 0.9300 | 0.8476 | 1.0000 |

## 5. Decoder tuning (HMM parameters only; train OOF trips only)
Seed 0; 30 trials (trial 0 = default config); objective = mean route F1 after fill_path over 30 train OOF trips; trips 00000054 and 00000033 excluded; student_test not used. Subset: results/hmm_rf_tuning_trips.csv. Every trial: results/hmm_rf_tuning_log.csv (also logs F1 without fill_path, precision, recall, pointwise accuracy). One-pass timing: results/hmm_rf_tuning_timing_one_pass.csv; console log results/hmm_rf_tune_run.log.

| config | objective | mean_f1_raw | mean_precision | mean_recall | mean_point_acc | mean_lcs_sim | total_breaks |
|---|---|---|---|---|---|---|---|
| default (trial 0) | 0.9452 | 0.9257 | 0.9068 | 0.9911 | 0.9266 | 0.8242 | 4 |
| best (trial 2) | 0.9728 | 0.8742 | 0.9582 | 0.9896 | 0.9156 | 0.9455 | 2 |

(objective = mean route F1 with fill_path; mean_f1_raw = the same decode without fill_path.)

Parameters:

| config | beta | switch_penalty | unreach_penalty | max_path_m | temperature | gap_scale_mps |
|---|---|---|---|---|---|---|
| default | 10.0 | 0.5 | 12.0 | 200.0 | 1.0 | 30.0 |
| tuned (written to hmm/config.json) | 15.7436 | 2.918 | 15.4161 | 567.5362 | 2.1569 | 5.0316 |

Top 5 trials by objective:

| trial | objective | mean_f1_raw | mean_precision | mean_recall | mean_point_acc | mean_lcs_sim | total_breaks |
|---|---|---|---|---|---|---|---|
| 2.0000 | 0.9728 | 0.8742 | 0.9582 | 0.9896 | 0.9156 | 0.9455 | 2.0000 |
| 7.0000 | 0.9724 | 0.8928 | 0.9570 | 0.9902 | 0.9236 | 0.9424 | 2.0000 |
| 20.0000 | 0.9718 | 0.8908 | 0.9559 | 0.9902 | 0.9235 | 0.9418 | 2.0000 |
| 9.0000 | 0.9692 | 0.9078 | 0.9509 | 0.9902 | 0.9237 | 0.9302 | 2.0000 |
| 13.0000 | 0.9673 | 0.8820 | 0.9496 | 0.9882 | 0.9189 | 0.9323 | 4.0000 |

## 6. Default vs tuned config on trips the tuner never saw
File: results/hmm_rf_tuned_vs_default.csv (per trip); console output results/hmm_rf_tuned_vs_default.txt. improved/tied/worse = tuned vs default per trip (tolerance 1e-9).
```
30 tuning trips excluded; evaluating 60 unseen trips (10 student_test + 50 train OOF)
--- default config: beta=10.0, switch_penalty=0.5, unreach_penalty=12.0, max_path_m=200.0, temperature=1.0, gap_scale_mps=30.0
--- tuned config: beta=15.7436, switch_penalty=2.918, unreach_penalty=15.4161, max_path_m=567.5362, temperature=2.1569, gap_scale_mps=5.0316

=== student_test (never seen by tuning) (10 trips): HMM decoder, default vs tuned config ===
metric       default mean  tuned mean   d mean |  def med tuned med    d med | improved/tied/worse
point_acc          0.9255      0.9003  -0.0252 |   0.9403    0.9043  -0.0359 | 3/0/7
precision          0.8894      0.9757  +0.0862 |   0.9418    0.9830  +0.0412 | 9/1/0
recall             0.9875      0.9880  +0.0005 |   0.9838    0.9852  +0.0013 | 2/7/1
f1                 0.9300      0.9816  +0.0516 |   0.9661    0.9846  +0.0184 | 9/1/0
lcs_sim            0.8476      0.9680  +0.1204 |   0.9207    0.9783  +0.0576 | 9/1/0
continuity         1.0000      1.0000  +0.0000 |   1.0000    1.0000  +0.0000 | 0/10/0
breaks total: default 0  tuned 0

=== remaining train OOF trips (not in the tuning subset) (50 trips): HMM decoder, default vs tuned config ===
metric       default mean  tuned mean   d mean |  def med tuned med    d med | improved/tied/worse
point_acc          0.9234      0.8963  -0.0272 |   0.9461    0.9095  -0.0365 | 12/0/38
precision          0.8781      0.9219  +0.0438 |   0.9397    0.9838  +0.0441 | 48/1/1
recall             0.9802      0.9782  -0.0020 |   0.9940    0.9924  -0.0016 | 11/19/20
f1                 0.9202      0.9453  +0.0251 |   0.9662    0.9884  +0.0222 | 47/1/2
lcs_sim            0.8273      0.8999  +0.0726 |   0.9187    0.9774  +0.0587 | 48/1/1
continuity         0.9979      0.9985  +0.0006 |   1.0000    1.0000  +0.0000 | 6/38/6
breaks total: default 60  tuned 42

=== all unseen trips (60 trips): HMM decoder, default vs tuned config ===
metric       default mean  tuned mean   d mean |  def med tuned med    d med | improved/tied/worse
point_acc          0.9237      0.8969  -0.0268 |   0.9449    0.9084  -0.0364 | 15/0/45
precision          0.8800      0.9309  +0.0508 |   0.9418    0.9830  +0.0412 | 57/2/1
recall             0.9814      0.9798  -0.0016 |   0.9934    0.9906  -0.0028 | 13/26/21
f1                 0.9218      0.9513  +0.0295 |   0.9662    0.9871  +0.0209 | 56/2/2
lcs_sim            0.8307      0.9113  +0.0806 |   0.9187    0.9775  +0.0588 | 57/2/1
continuity         0.9983      0.9988  +0.0005 |   1.0000    1.0000  +0.0000 | 6/48/6
breaks total: default 60  tuned 42

written results/hmm_rf_tuned_vs_default.csv
```

## 7. API latency, 10 student_test trips (`hmm/latency.py`, tuned config, P2's model, one-off model load excluded)
File: results/hmm_latency.csv, console results/hmm_latency.txt.

| trip_id | n_points | n_candidates | candidates_s | features_s | model_s | decode_s | total_s |
|---|---|---|---|---|---|---|---|
| 00000000 | 2503 | 23096 | 0.3204 | 0.0232 | 0.3755 | 0.8900 | 1.6173 |
| 00000007 | 1354 | 9638 | 0.1235 | 0.0123 | 0.1498 | 0.3792 | 0.6683 |
| 00000009 | 1049 | 27936 | 0.2187 | 0.0263 | 0.4239 | 1.4273 | 2.1016 |
| 00000040 | 2995 | 29619 | 0.2791 | 0.0250 | 0.4271 | 1.1516 | 1.8877 |
| 00000052 | 262 | 5301 | 0.1200 | 0.0102 | 0.1283 | 0.2308 | 0.4925 |
| 00000056 | 3958 | 90851 | 0.4956 | 0.0877 | 1.2929 | 3.7718 | 5.6582 |
| 00000065 | 768 | 20467 | 0.2015 | 0.0200 | 0.3482 | 0.9550 | 1.5299 |
| 00000076 | 1486 | 42442 | 0.4722 | 0.0333 | 0.6349 | 2.7579 | 3.9046 |
| 00000077 | 3911 | 95972 | 0.3517 | 0.0840 | 1.3986 | 3.6122 | 5.4577 |
| 00000091 | 874 | 21045 | 0.1957 | 0.0203 | 0.3342 | 0.9329 | 1.4882 |

Slowest trip: 00000056, 3958 points, 5.66 s. Mean 2.48 s, median 1.75 s.

## 8. Maps (tuned config, student_test, reference route drawn for dev only)
Selection rule (fixed, applied to results/hmm_rf_tuned_vs_default.csv): see viz/render_selected.py; chosen trips in results/maps/selection_rf_tuned.csv. Maps: results/maps/student_test_<trip>_random_forest__p2__tuned.html/.png

| reason | trip_id | classifier_only_f1 | tuned_f1 | gain | n_points |
|---|---|---|---|---|---|
| helps a lot (largest tuned_f1 - classifier_only_f1) | 00000077 | 0.6920 | 0.9781 | 0.2861 | 3911 |
| does not help / fails (smallest gain) | 00000007 | 0.9416 | 0.9918 | 0.0502 | 1354 |
| clean case (highest tuned_f1) | 00000040 | 0.8371 | 0.9952 | 0.1580 | 2995 |
| weakest result (lowest tuned_f1) | 00000076 | 0.6771 | 0.9486 | 0.2715 | 1486 |
| typical (gain closest to the median gain) | 00000000 | 0.8117 | 0.9804 | 0.1687 | 2503 |

## 9. hmm/config.json
`hmm/config.json` holds the tuned parameters of section 5 (its `_meta.status` has date, objective, trials, seed). The default config is saved as results/hmm_default_config.json. The distance-only baseline results (results/hmm_baseline_comparison.csv, results/maps/*distance_only_baseline*) and sections 3-4 were produced with the default config.

## 10. Reproduce
All commands in order: hmm/README.md, section "Final plug-in steps with P2's random forest", plus:
```
python -m hmm.latency --out results/hmm_latency.csv
python -m viz.render_selected --probs results/p2/probs_all.csv.gz --table results/hmm_rf_tuned_vs_default.csv
python -m pytest tests -q
```

## 11. Known limits (facts)
- Trips 00000054 and 00000033: reference route covers 11% / 41% of the track; excluded from tuning, included in evaluation.
- Probabilities of the 80 train trips are out-of-fold (5 folds); student_test probabilities come from the final model trained on all 80 train trips.
- Tuning used 30 random trials on 30 trips; the tuner objective is route F1 after fill_path. The tuned config has lower pointwise accuracy than the default on most unseen trips (section 6).
