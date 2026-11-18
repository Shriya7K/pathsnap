"""Optuna tuning for Random Forest and HistGradientBoosting.

Run from the project root:
    python data_processing/tune_models.py

Uses trip-level 3-fold GroupKFold during tuning. The original 5-fold
baseline results are not modified.
"""

from pathlib import Path
import gc
import json
import warnings

import numpy as np
import optuna
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from train_models import (
    DATA_DIR,
    RANDOM_STATE,
    TRAIN_ROW_LIMITS,
    get_trip_files,
    trip_id_from_file,
    load_training_sample,
    evaluate_trip,
    fit_model,
)


warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------
# Tuning configuration
# --------------------------------------------------

OUTPUT_DIR = Path("results/tuning")
N_SPLITS = 3
N_TRIALS = 20

# --------------------------------------------------
# Objective helpers
# --------------------------------------------------

def make_model(model_name, trial):
    """Create one model using parameters proposed by Optuna."""

    if model_name == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 180, step=20),
            "max_depth": trial.suggest_int("max_depth", 10, 30, step=2),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", 0.7, 1.0]
            ),
        }

        model = RandomForestClassifier(
            **params,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    elif model_name == "HistGradientBoosting":
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.03, 0.2, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 80, 200, step=20),
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", 10, 31
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 15, 50, step=5
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-4, 10.0, log=True
            ),
        }

        model = HistGradientBoostingClassifier(
            **params,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, params


def objective(trial, model_name, files, splitter):
    """Average trip-level-fold Top-1 accuracy over the tuning folds."""

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(files, groups=np.arange(len(files))),
        start=1,
    ):
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]

        row_limit = TRAIN_ROW_LIMITS[model_name]
        X_train, y_train = load_training_sample(
            train_files,
            row_limit,
            seed=RANDOM_STATE + fold,
        )

        model, _ = make_model(model_name, trial)
        model = fit_model(model, model_name, X_train, y_train)

        del X_train, y_train
        gc.collect()

        total_points = 0
        correct_points = 0

        for path in val_files:
            metrics = evaluate_trip(model, path)
            total_points += metrics["total_points"]
            correct_points += metrics["correct_points"]
            del metrics

        fold_accuracy = (
            correct_points / total_points if total_points else 0.0
        )
        fold_scores.append(fold_accuracy)

        del model
        gc.collect()

        # Report after each fold so Optuna can stop weak trials early.
        trial.report(float(np.mean(fold_scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


# --------------------------------------------------
# Run tuning
# --------------------------------------------------

def tune_one_model(model_name, files, splitter):
    print("\n" + "=" * 65)
    print(f"OPTUNA TUNING: {model_name}")
    print("=" * 65)
    print(f"Trials: {N_TRIALS}")
    print(f"Validation folds per trial: {N_SPLITS}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        ),
        study_name=model_name.replace(" ", "_").lower(),
    )

    study.optimize(
        lambda trial: objective(trial, model_name, files, splitter),
        n_trials=N_TRIALS,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    trials_path = OUTPUT_DIR / (
        "random_forest_trials.csv"
        if model_name == "Random Forest"
        else "hist_gradient_boosting_trials.csv"
    )
    study.trials_dataframe().to_csv(trials_path, index=False)

    best = {
        "model": model_name,
        "best_value_mean_3fold_top1_accuracy": study.best_value,
        "best_params": study.best_params,
        "completed_trials": sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        ),
        "pruned_trials": sum(
            trial.state == optuna.trial.TrialState.PRUNED
            for trial in study.trials
        ),
    }

    print(f"\nBest mean 3-fold Top-1 accuracy: {study.best_value:.4f}")
    print("Best parameters:")
    print(json.dumps(study.best_params, indent=4))

    return best


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = get_trip_files()
    trip_ids = [trip_id_from_file(path) for path in files]

    if len(files) < N_SPLITS:
        raise ValueError("Not enough training trips for GroupKFold.")

    # Use trip IDs as groups. Each trip appears in either train or
    # validation within a fold, never both.
    splitter = GroupKFold(n_splits=N_SPLITS)

    print("=" * 65)
    print("PATHSNAP - OPTUNA HYPERPARAMETER TUNING")
    print("=" * 65)
    print(f"Training trips: {len(files)}")
    print(f"Training feature directory: {DATA_DIR}")
    print(f"Validation: {N_SPLITS}-fold GroupKFold")
    print(f"Trials per model: {N_TRIALS}")
    print("Optimization metric: mean fold-level Top-1 accuracy")

    all_best = {}

    for model_name in ["Random Forest", "HistGradientBoosting"]:
        best = tune_one_model(model_name, files, splitter)
        all_best[model_name] = best

        # Save after each model, so the first model's result survives
        # if the second model is interrupted.
        with (OUTPUT_DIR / "best_params.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(all_best, file, indent=4)

    print("\n" + "=" * 65)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 65)
    print(f"Best-parameter file: {(OUTPUT_DIR / 'best_params.json').resolve()}")
    print(f"Trial results directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
