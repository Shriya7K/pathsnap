"""Bayesian hyperparameter optimisation using Optuna for PathSnap Person 2."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features.config import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    RANDOM_SEED,
)
from models.train_models import compute_top1_accuracy


out_dir = Path("results/tuning")
out_dir.mkdir(parents=True, exist_ok=True)


def run_optuna_tuning(
    train_file: Path = Path("results/features/train_features.csv.gz"),
    test_file: Path = Path("results/features/student_test_features.csv.gz"),
    n_trials: int = 15,
    n_splits: int = 3,  # 3-fold GroupKFold for rapid, robust exploration
):
    print("Loading training data for Optuna tuning...")
    df = pd.read_csv(train_file)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int8)
    groups = df["trip_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(X, y, groups=groups))

    def objective(trial: optuna.Trial) -> float:
        # Define search space for HistGradientBoostingClassifier
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 150, step=25),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 5.0, log=True),
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
        }

        fold_f1s = []
        for train_idx, val_idx in folds:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Use 200k representative subsample per trial for rapid Bayesian exploration
            if len(X_tr) > 200000:
                np.random.seed(RANDOM_SEED + trial.number)
                s_idx = np.random.choice(len(X_tr), size=200000, replace=False)
                X_tr_fit, y_tr_fit = X_tr[s_idx], y_tr[s_idx]
            else:
                X_tr_fit, y_tr_fit = X_tr, y_tr

            model = HistGradientBoostingClassifier(**params)
            model.fit(X_tr_fit, y_tr_fit)

            val_probs = model.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(np.int8)
            f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            fold_f1s.append(f1)

        return float(np.mean(fold_f1s) * 100.0)

    print(f"\nStarting Optuna Bayesian optimization ({n_trials} trials, {n_splits}-fold GroupKFold objective: Macro F1)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="hist_gradient_boosting_tuning")

    t_tune_start = time.time()
    study.optimize(objective, n_trials=n_trials)
    total_tune_time = time.time() - t_tune_start

    print(f"\nOptimization completed in {total_tune_time:.1f}s.")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Validation Macro F1: {study.best_value:.3f}%")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)

    best_config = {
        "model": "HistGradientBoostingClassifier",
        "best_trial_number": study.best_trial.number,
        "best_cv_macro_f1": round(study.best_value, 3),
        "best_params": study.best_params,
        "total_tuning_time_sec": round(total_tune_time, 2),
    }
    with open(out_dir / "best_hyperparameters.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # --- Plot Optimization History ---
    fig, ax = plt.subplots(figsize=(8, 4))
    trial_vals = [t.value for t in study.trials if t.value is not None]
    best_vals = np.maximum.accumulate(trial_vals)
    ax.scatter(range(1, len(trial_vals) + 1), trial_vals, color="#386cb0", alpha=0.7, label="Trial Macro F1")
    ax.plot(range(1, len(trial_vals) + 1), best_vals, color="#e41a1c", lw=2, label="Best-so-far")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation Macro F1 (%)")
    ax.set_title("Optuna Bayesian Hyperparameter Optimization History")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "optimization_history.png", dpi=150)
    plt.close(fig)

    # --- Plot Parameter Importances ---
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, 4))
        params_sorted = list(importances.keys())[::-1]
        vals_sorted = [importances[p] for p in params_sorted]
        ax.barh(params_sorted, vals_sorted, color="#7fc97f", edgecolor="black")
        ax.set_xlabel("Relative Importance")
        ax.set_title("Hyperparameter Importance (fANOVA)")
        fig.tight_layout()
        fig.savefig(out_dir / "param_importances.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Could not plot parameter importances: {e}")

    # --- Step 14: Compare Tuned vs Untuned Performance on 5-Fold CV and Student-Test ---
    print("\nEvaluating Tuned vs Untuned model across full dataset...")
    test_df = pd.read_csv(test_file)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.int8)

    # Untuned baseline
    untuned_params = {
        "max_iter": 100, "learning_rate": 0.08, "max_leaf_nodes": 31,
        "min_samples_leaf": 30, "class_weight": "balanced", "random_state": RANDOM_SEED
    }
    tuned_params = dict(study.best_params)
    tuned_params["class_weight"] = "balanced"
    tuned_params["random_state"] = RANDOM_SEED

    models_to_eval = [("Untuned Baseline", untuned_params), ("Tuned with Optuna", tuned_params)]
    comp_records = []

    # 5-fold CV comparison
    gkf5 = GroupKFold(n_splits=5)
    folds5 = list(gkf5.split(X, y, groups=groups))

    for tag, p in models_to_eval:
        t0 = time.time()
        cv_top1s, cv_f1s, cv_losses = [], [], []

        for tr_idx, val_idx in folds5:
            m = HistGradientBoostingClassifier(**p)
            m.fit(X[tr_idx], y[tr_idx])
            probs = m.predict_proba(X[val_idx])[:, 1]
            preds = (probs >= 0.5).astype(np.int8)

            val_df_fold = df.iloc[val_idx].copy()
            val_df_fold["pred_prob"] = probs
            cv_top1s.append(compute_top1_accuracy(val_df_fold, prob_col="pred_prob"))
            cv_f1s.append(f1_score(y[val_idx], preds, average="macro", zero_division=0) * 100.0)
            cv_losses.append(log_loss(y[val_idx], np.clip(probs, 1e-7, 1 - 1e-7)))

        train_time = time.time() - t0

        # Fit on full training set and evaluate on student-test
        m_full = HistGradientBoostingClassifier(**p)
        m_full.fit(X, y)
        t_inf0 = time.time()
        te_probs = m_full.predict_proba(X_test)[:, 1]
        te_infer_time = time.time() - t_inf0
        te_preds = (te_probs >= 0.5).astype(np.int8)

        test_eval_df = test_df.copy()
        test_eval_df["pred_prob"] = te_probs
        te_top1 = compute_top1_accuracy(test_eval_df, prob_col="pred_prob")
        te_f1 = f1_score(y_test, te_preds, average="macro", zero_division=0) * 100.0
        te_loss = log_loss(y_test, np.clip(te_probs, 1e-7, 1 - 1e-7))

        comp_records.append({
            "Configuration": tag,
            "CV Top-1 Mean (%)": round(np.mean(cv_top1s), 3),
            "CV Top-1 Std (%)": round(np.std(cv_top1s), 3),
            "CV Macro F1 (%)": round(np.mean(cv_f1s), 3),
            "CV Log Loss": round(np.mean(cv_losses), 4),
            "Test Top-1 Accuracy (%)": round(te_top1, 3),
            "Test Macro F1 (%)": round(te_f1, 3),
            "Test Log Loss": round(te_loss, 4),
            "CV Total Train Time (s)": round(train_time, 2),
            "Test Inference Time (s)": round(te_infer_time, 2),
            "Parameters": json.dumps(p),
        })

    comp_df = pd.DataFrame(comp_records)
    comp_df.to_csv(out_dir / "tuned_vs_untuned_comparison.csv", index=False)

    print("\n" + "="*70)
    print("           TUNED VS UNTUNED MODEL PERFORMANCE COMPARISON")
    print("="*70)
    print(comp_df[["Configuration", "CV Top-1 Mean (%)", "CV Macro F1 (%)", "Test Top-1 Accuracy (%)", "Test Macro F1 (%)"]].to_string(index=False))
    print("="*70)
    print(f"Saved tuned-vs-untuned comparison to: {out_dir / 'tuned_vs_untuned_comparison.csv'}")
    return study, best_config


if __name__ == "__main__":
    run_optuna_tuning()
