"""Evaluate classification models and final tuned model on held-out evaluator_test partition."""
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler

from features.config import FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_SEED
from models.train_models import get_model_definitions, compute_top1_accuracy

out_dir = Path("results/models")
out_dir.mkdir(parents=True, exist_ok=True)


def evaluate_on_evaluator_test():
    train_file = Path("results/features/train_features.csv.gz")
    eval_file = Path("results/features/evaluator_test_features.csv.gz")

    print(f"Loading feature datasets for evaluator_test evaluation...")
    train_df = pd.read_csv(train_file)
    eval_df = pd.read_csv(eval_file)

    print(f"Train dataset: {len(train_df):,} rows across {train_df['trip_id'].nunique()} trips.")
    print(f"Evaluator-test dataset: {len(eval_df):,} rows across {eval_df['trip_id'].nunique()} trips.")

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.int8)

    X_eval = eval_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_eval = eval_df[TARGET_COLUMN].to_numpy(dtype=np.int8)

    model_defs = get_model_definitions()
    results = []
    per_trip_results = []

    # 1. Evaluate baseline model definitions
    for name, spec in model_defs.items():
        print(f"\nEvaluating {name} ({spec['family']}) on evaluator_test...")
        t0 = time.time()

        if "subsample_train" in spec and len(X_train) > spec["subsample_train"]:
            np.random.seed(RANDOM_SEED)
            s_idx = np.random.choice(len(X_train), size=spec["subsample_train"], replace=False)
            X_tr_fit, y_tr_fit = X_train[s_idx], y_train[s_idx]
        else:
            X_tr_fit, y_tr_fit = X_train, y_train

        if spec["needs_scale"]:
            scaler = StandardScaler()
            X_tr_fit = scaler.fit_transform(X_tr_fit)
            X_te_eval = scaler.transform(X_eval)
        else:
            X_te_eval = X_eval

        model = spec["model"].__class__(**spec["model"].get_params())
        model.fit(X_tr_fit, y_tr_fit)
        train_time = time.time() - t0

        t_infer = time.time()
        test_probs = model.predict_proba(X_te_eval)[:, 1]
        infer_time = time.time() - t_infer

        preds_bin = (test_probs >= 0.5).astype(np.int8)
        macro_f1 = f1_score(y_eval, preds_bin, average="macro", zero_division=0)
        pos_f1 = f1_score(y_eval, preds_bin, pos_label=1, zero_division=0)
        prec = precision_score(y_eval, preds_bin, pos_label=1, zero_division=0)
        rec = recall_score(y_eval, preds_bin, pos_label=1, zero_division=0)
        loss = log_loss(y_eval, np.clip(test_probs, 1e-7, 1 - 1e-7))

        test_eval_df = eval_df.copy()
        test_eval_df["pred_prob"] = test_probs
        top1_acc = compute_top1_accuracy(test_eval_df, prob_col="pred_prob")
        tn, fp, fn, tp = confusion_matrix(y_eval, preds_bin).ravel()

        results.append({
            "model": name,
            "family": spec["family"],
            "eval_top1_accuracy": round(top1_acc, 3),
            "eval_macro_f1": round(macro_f1 * 100, 3),
            "eval_positive_f1": round(pos_f1 * 100, 3),
            "eval_precision": round(prec * 100, 3),
            "eval_recall": round(rec * 100, 3),
            "eval_log_loss": round(loss, 4),
            "train_time_sec": round(train_time, 2),
            "infer_time_sec": round(infer_time, 2),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
        print(f"  {name}: Top-1 Acc = {top1_acc:.2f}% | Macro F1 = {macro_f1*100:.2f}% | Loss = {loss:.4f} | Infer = {infer_time:.2f}s")

    # 2. Evaluate final production model (Tuned HistGradientBoosting)
    final_model_path = Path("artifacts/final_model.joblib")
    if final_model_path.exists():
        print(f"\nEvaluating Final Production Model ({final_model_path})...")
        final_model = joblib.load(final_model_path)
        t_infer = time.time()
        final_probs = final_model.predict_proba(X_eval)[:, 1]
        infer_time = time.time() - t_infer

        preds_bin = (final_probs >= 0.5).astype(np.int8)
        macro_f1 = f1_score(y_eval, preds_bin, average="macro", zero_division=0)
        pos_f1 = f1_score(y_eval, preds_bin, pos_label=1, zero_division=0)
        prec = precision_score(y_eval, preds_bin, pos_label=1, zero_division=0)
        rec = recall_score(y_eval, preds_bin, pos_label=1, zero_division=0)
        loss = log_loss(y_eval, np.clip(final_probs, 1e-7, 1 - 1e-7))

        test_eval_df = eval_df.copy()
        test_eval_df["pred_prob"] = final_probs
        top1_acc = compute_top1_accuracy(test_eval_df, prob_col="pred_prob")
        tn, fp, fn, tp = confusion_matrix(y_eval, preds_bin).ravel()

        results.append({
            "model": "Final Tuned Model (HistGradientBoosting)",
            "family": "Tuned Gradient Boosted Trees",
            "eval_top1_accuracy": round(top1_acc, 3),
            "eval_macro_f1": round(macro_f1 * 100, 3),
            "eval_positive_f1": round(pos_f1 * 100, 3),
            "eval_precision": round(prec * 100, 3),
            "eval_recall": round(rec * 100, 3),
            "eval_log_loss": round(loss, 4),
            "train_time_sec": 40.6,
            "infer_time_sec": round(infer_time, 2),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
        print(f"  Final Tuned Model: Top-1 Acc = {top1_acc:.2f}% | Macro F1 = {macro_f1*100:.2f}% | Loss = {loss:.4f} | Infer = {infer_time:.2f}s")

        # Per-trip breakdown for the final model
        for trip_id, t_group in test_eval_df.groupby("trip_id"):
            t_y = t_group[TARGET_COLUMN].to_numpy()
            t_probs = t_group["pred_prob"].to_numpy()
            t_preds = (t_probs >= 0.5).astype(np.int8)
            t_f1 = f1_score(t_y, t_preds, average="macro", zero_division=0)
            t_top1 = compute_top1_accuracy(t_group, prob_col="pred_prob")
            per_trip_results.append({
                "trip_id": trip_id,
                "n_candidates": len(t_group),
                "n_observations": t_group.drop_duplicates(subset=["trip_id", "point_idx"]).shape[0],
                "top1_accuracy": round(t_top1, 2),
                "macro_f1": round(t_f1 * 100, 2),
            })

    # Save results
    res_df = pd.DataFrame(results)
    trip_df = pd.DataFrame(per_trip_results)

    res_df.to_csv(out_dir / "evaluator_test_results.csv", index=False)
    trip_df.to_csv(out_dir / "evaluator_test_per_trip.csv", index=False)

    print("\n" + "=" * 70)
    print("             EVALUATOR-TEST CLASSIFIER EVALUATION RESULTS")
    print("=" * 70)
    print(res_df[["model", "family", "eval_top1_accuracy", "eval_macro_f1", "eval_log_loss", "infer_time_sec"]].to_string(index=False))
    print("=" * 70)
    print(f"Saved: {out_dir / 'evaluator_test_results.csv'}")
    print(f"Saved: {out_dir / 'evaluator_test_per_trip.csv'}")
    return res_df, trip_df


if __name__ == "__main__":
    evaluate_on_evaluator_test()
