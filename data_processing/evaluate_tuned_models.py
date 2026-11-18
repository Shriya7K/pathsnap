"""Evaluate Optuna-tuned models with the original 5-fold GroupKFold.

Run from the project root:
    python data_processing/evaluate_tuned_models.py

This uses the same bounded per-model training samples and helper functions
as train_models.py, and writes separate files under results/tuning/.
"""

from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from train_models import (
    RANDOM_STATE,
    TRAIN_ROW_LIMITS,
    get_trip_files,
    trip_id_from_file,
    load_training_sample,
    evaluate_trip,
    fit_model,
)

TUNING_DIR = Path("results/tuning")
N_SPLITS = 5


def build_tuned_model(model_name, params):
    if model_name == "Random Forest":
        return RandomForestClassifier(
            **params,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            **params,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def main():
    params_path = TUNING_DIR / "best_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing tuning results: {params_path}")

    with params_path.open("r", encoding="utf-8") as file:
        tuning_results = json.load(file)

    files = get_trip_files()
    trip_ids = [trip_id_from_file(path) for path in files]

    if len(files) < N_SPLITS:
        raise ValueError("Not enough training trips for 5-fold GroupKFold.")

    splitter = GroupKFold(n_splits=N_SPLITS)
    results = []

    print("=" * 65)
    print("PATHSNAP - TUNED MODEL 5-FOLD EVALUATION")
    print("=" * 65)
    print(f"Training trips: {len(files)}")
    print(f"Validation folds: {N_SPLITS}")
    print("Models: Random Forest, HistGradientBoosting")

    for model_name in ["Random Forest", "HistGradientBoosting"]:
        params = tuning_results[model_name]["best_params"]

        print("\n" + "=" * 65)
        print(f"MODEL: {model_name}")
        print("Parameters:", params)
        print("=" * 65)

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(trip_ids, groups=trip_ids),
            start=1,
        ):
            train_files = [files[i] for i in train_idx]
            val_files = [files[i] for i in val_idx]

            print(f"\nFOLD {fold}/{N_SPLITS}")
            row_limit = TRAIN_ROW_LIMITS[model_name]

            X_train, y_train = load_training_sample(
                train_files,
                row_limit,
                seed=RANDOM_STATE + fold,
            )

            model = build_tuned_model(model_name, params)
            model = fit_model(model, model_name, X_train, y_train)

            del X_train, y_train
            gc.collect()

            total_points = 0
            positive_points = 0
            correct_points = 0
            conditional_correct = 0
            auc_values = []

            for path in val_files:
                metrics = evaluate_trip(model, path)
                total_points += metrics["total_points"]
                positive_points += metrics["points_with_positive_candidate"]
                correct_points += metrics["correct_points"]
                conditional_correct += metrics["conditional_correct"]

                if not np.isnan(metrics["auc"]):
                    auc_values.append(metrics["auc"])
                del metrics

            top1 = correct_points / total_points if total_points else 0.0
            coverage = positive_points / total_points if total_points else 0.0
            conditional = (
                conditional_correct / positive_points if positive_points else 0.0
            )
            mean_auc = float(np.mean(auc_values)) if auc_values else np.nan

            results.append({
                "model": model_name,
                "fold": fold,
                "training_row_limit": row_limit,
                "validation_points": total_points,
                "candidate_coverage": coverage,
                "top1_accuracy": top1,
                "conditional_accuracy": conditional,
                "mean_trip_auc": mean_auc,
            })

            print(f"Top-1 accuracy: {top1:.4f}")
            print(f"Candidate coverage: {coverage:.4f}")
            print(f"Conditional accuracy: {conditional:.4f}")
            print(f"Mean trip ROC-AUC: {mean_auc:.4f}")

            del model
            gc.collect()

            pd.DataFrame(results).to_csv(
                TUNING_DIR / "tuned_5fold_results.csv",
                index=False,
            )

    results_df = pd.DataFrame(results)
    summary = (
        results_df.groupby("model")
        .agg(
            folds_completed=("fold", "nunique"),
            mean_top1_accuracy=("top1_accuracy", "mean"),
            std_top1_accuracy=("top1_accuracy", "std"),
            mean_candidate_coverage=("candidate_coverage", "mean"),
            mean_conditional_accuracy=("conditional_accuracy", "mean"),
            mean_auc=("mean_trip_auc", "mean"),
        )
        .reset_index()
        .sort_values("mean_top1_accuracy", ascending=False)
    )

    summary.to_csv(TUNING_DIR / "tuned_5fold_comparison.csv", index=False)

    print("\n" + "=" * 65)
    print("TUNED MODEL 5-FOLD COMPARISON")
    print("=" * 65)
    print(summary.to_string(index=False))
    print(f"\nResults saved in: {TUNING_DIR.resolve()}")


if __name__ == "__main__":
    main()
