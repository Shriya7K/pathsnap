"""Model training and GroupKFold cross-validation for PathSnap Person 2."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
)

from features.config import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    RANDOM_SEED,
    N_FOLDS,
)


def compute_top1_accuracy(df: pd.DataFrame, prob_col: str = "pred_prob") -> float:
    """Compute observation-level Top-1 candidate ranking accuracy.

    Groups predictions by (trip_id, point_idx) and checks whether the candidate
    with the highest predicted probability is the true reference arc (label == 1).
    """
    # Sort by prob descending, then distance_rank ascending for deterministic tie-breaking
    sort_cols = [prob_col]
    ascending = [False]
    if "distance_rank" in df.columns:
        sort_cols.append("distance_rank")
        ascending.append(True)

    sorted_df = df.sort_values(sort_cols, ascending=ascending)
    # Pick the top-ranked candidate for each observation
    top1 = sorted_df.drop_duplicates(subset=["trip_id", "point_idx"], keep="first")
    return float(top1[TARGET_COLUMN].mean() * 100.0)


def get_model_definitions():
    """Return dictionary of 5 model architectures from different families."""
    return {
        "Logistic Regression": {
            "family": "Linear",
            "needs_scale": True,
            "model": LogisticRegression(
                C=1.0,
                max_iter=300,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_SEED,
            ),
            "params": {"C": 1.0, "max_iter": 300, "class_weight": "balanced", "solver": "lbfgs"},
        },
        "K-Nearest Neighbours": {
            "family": "Metric / Instance-based",
            "needs_scale": True,
            "subsample_train": 25000,  # Scalable metric index
            "subsample_val_obs": 2500,  # Representative validation sample (~27k candidates) for fast CV
            "model": KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                algorithm="kd_tree",
                leaf_size=50,
                n_jobs=-1,
            ),
            "params": {"n_neighbors": 15, "weights": "distance", "algorithm": "kd_tree", "leaf_size": 50},
        },
        "Random Forest": {
            "family": "Bagging Ensemble",
            "needs_scale": False,
            "model": RandomForestClassifier(
                n_estimators=60,
                max_depth=14,
                min_samples_split=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            "params": {"n_estimators": 60, "max_depth": 14, "min_samples_split": 20, "class_weight": "balanced"},
        },
        "HistGradientBoosting": {
            "family": "Gradient Boosted Trees",
            "needs_scale": False,
            "model": HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            "params": {"max_iter": 100, "learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 30, "class_weight": "balanced"},
        },
        "Multi-Layer Perceptron": {
            "family": "Neural Network",
            "needs_scale": True,
            "subsample_train": 150000,
            "model": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.001,
                max_iter=40,
                batch_size=2048,
                early_stopping=True,
                random_state=RANDOM_SEED,
            ),
            "params": {"hidden_layer_sizes": (64, 32), "activation": "relu", "alpha": 0.001, "batch_size": 2048},
        },
    }


def train_and_evaluate_cv(
    train_file: Path = Path("results/features/train_features.csv.gz"),
    out_dir: Path = Path("results/models"),
    n_splits: int = N_FOLDS,
    sample_trips: int = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading training features from {train_file}...")
    df = pd.read_csv(train_file)
    print(f"Loaded {len(df):,} candidate rows across {df['trip_id'].nunique()} trips.")

    if sample_trips:
        unique_trips = df["trip_id"].unique()[:sample_trips]
        df = df[df["trip_id"].isin(unique_trips)].reset_index(drop=True)
        print(f"Filtered to sample of {sample_trips} trips ({len(df):,} rows).")

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int8)
    groups = df["trip_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(X, y, groups=groups))

    # Verify no trip overlap across folds
    for f_idx, (tr_idx, val_idx) in enumerate(folds):
        tr_trips = set(groups[tr_idx])
        val_trips = set(groups[val_idx])
        overlap = tr_trips.intersection(val_trips)
        assert len(overlap) == 0, f"Data leakage: trips {overlap} in both train and val of fold {f_idx}"

    print(f"Verified strict GroupKFold: 0 overlapping trips across all {n_splits} folds.")

    model_defs = get_model_definitions()
    detailed_results = []
    summary_results = []

    for name, spec in model_defs.items():
        print(f"\n{'='*60}\nEvaluating Model: {name} ({spec['family']})\n{'='*60}")
        fold_metrics = []
        oof_preds = np.zeros(len(df), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(folds, 1):
            t_train_start = time.time()
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Optional subsampling for scalable KNN neighbor index construction
            if "subsample_train" in spec and len(X_tr) > spec["subsample_train"]:
                np.random.seed(RANDOM_SEED + fold)
                sample_idx = np.random.choice(len(X_tr), size=spec["subsample_train"], replace=False)
                X_tr_fit, y_tr_fit = X_tr[sample_idx], y_tr[sample_idx]
            else:
                X_tr_fit, y_tr_fit = X_tr, y_tr

            # Scaling (fitted strictly on training fold only)
            if spec["needs_scale"]:
                scaler = StandardScaler()
                X_tr_fit = scaler.fit_transform(X_tr_fit)
                X_val_eval = scaler.transform(X_val)
            else:
                X_val_eval = X_val

            # Fit model
            model = spec["model"].__class__(**spec["model"].get_params())
            model.fit(X_tr_fit, y_tr_fit)
            train_time = time.time() - t_train_start

            # Predict probabilities
            t_infer_start = time.time()
            val_df_fold = df.iloc[val_idx].copy()
            if "subsample_val_obs" in spec and spec["subsample_val_obs"]:
                obs_keys = val_df_fold[["trip_id", "point_idx"]].drop_duplicates()
                if len(obs_keys) > spec["subsample_val_obs"]:
                    sample_obs = obs_keys.sample(n=spec["subsample_val_obs"], random_state=RANDOM_SEED + fold)
                    sub_idx = val_df_fold.reset_index().merge(sample_obs, on=["trip_id", "point_idx"])["index"].values
                    # map sub_idx relative to fold
                    val_mask = np.isin(val_idx, sub_idx)
                    X_val_eval_use = X_val_eval[val_mask]
                    y_val_use = y_val[val_mask]
                    val_df_fold_use = df.iloc[val_idx[val_mask]].copy()
                else:
                    X_val_eval_use = X_val_eval
                    y_val_use = y_val
                    val_df_fold_use = val_df_fold
            else:
                X_val_eval_use = X_val_eval
                y_val_use = y_val
                val_df_fold_use = val_df_fold

            val_probs = model.predict_proba(X_val_eval_use)[:, 1]
            infer_time = time.time() - t_infer_start

            if len(val_probs) == len(val_idx):
                oof_preds[val_idx] = val_probs

            # Binary metrics
            val_preds_bin = (val_probs >= 0.5).astype(np.int8)
            macro_f1 = f1_score(y_val_use, val_preds_bin, average="macro", zero_division=0)
            pos_f1 = f1_score(y_val_use, val_preds_bin, pos_label=1, zero_division=0)
            precision = precision_score(y_val_use, val_preds_bin, pos_label=1, zero_division=0)
            recall = recall_score(y_val_use, val_preds_bin, pos_label=1, zero_division=0)
            loss = log_loss(y_val_use, np.clip(val_probs, 1e-7, 1 - 1e-7))

            # Observation-level Top-1 accuracy
            val_df_fold_use["pred_prob"] = val_probs
            top1_acc = compute_top1_accuracy(val_df_fold_use, prob_col="pred_prob")

            tn, fp, fn, tp = confusion_matrix(y_val_use, val_preds_bin).ravel()

            metric_row = {
                "model": name,
                "fold": fold,
                "n_train_rows": len(X_tr),
                "n_val_rows": len(X_val),
                "n_val_obs": val_df_fold.drop_duplicates(subset=["trip_id", "point_idx"]).shape[0],
                "macro_f1": round(macro_f1 * 100, 3),
                "positive_f1": round(pos_f1 * 100, 3),
                "precision": round(precision * 100, 3),
                "recall": round(recall * 100, 3),
                "top1_accuracy": round(top1_acc, 3),
                "log_loss": round(loss, 4),
                "train_time_sec": round(train_time, 2),
                "infer_time_sec": round(infer_time, 2),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }
            detailed_results.append(metric_row)
            fold_metrics.append(metric_row)
            print(f"  Fold {fold}: Macro F1 = {macro_f1*100:.2f}% | Top-1 Acc = {top1_acc:.2f}% | Loss = {loss:.4f} | Train = {train_time:.1f}s")

        # Summary across folds
        f_df = pd.DataFrame(fold_metrics)
        summary_results.append({
            "model": name,
            "family": spec["family"],
            "cv_macro_f1_mean": round(f_df["macro_f1"].mean(), 3),
            "cv_macro_f1_std": round(f_df["macro_f1"].std(), 3),
            "cv_top1_accuracy_mean": round(f_df["top1_accuracy"].mean(), 3),
            "cv_top1_accuracy_std": round(f_df["top1_accuracy"].std(), 3),
            "cv_log_loss_mean": round(f_df["log_loss"].mean(), 4),
            "cv_precision_mean": round(f_df["precision"].mean(), 3),
            "cv_recall_mean": round(f_df["recall"].mean(), 3),
            "avg_train_time_sec": round(f_df["train_time_sec"].mean(), 2),
            "avg_infer_time_sec": round(f_df["infer_time_sec"].mean(), 2),
            "hyperparameters": json.dumps(spec["params"]),
        })

    det_df = pd.DataFrame(detailed_results)
    sum_df = pd.DataFrame(summary_results).sort_values("cv_top1_accuracy_mean", ascending=False)

    det_df.to_csv(out_dir / "cv_results_detailed.csv", index=False)
    sum_df.to_csv(out_dir / "cv_results_summary.csv", index=False)

    print("\n" + "="*70)
    print("               CROSS-VALIDATION RESULTS SUMMARY (5-FOLD GROUPKFOLD)")
    print("="*70)
    print(sum_df[["model", "family", "cv_top1_accuracy_mean", "cv_macro_f1_mean", "cv_log_loss_mean", "avg_train_time_sec"]].to_string(index=False))
    print("="*70)
    print(f"Detailed results saved to: {out_dir / 'cv_results_detailed.csv'}")
    print(f"Summary results saved to: {out_dir / 'cv_results_summary.csv'}")
    return sum_df


if __name__ == "__main__":
    train_and_evaluate_cv()
