"""Fit final best-tuned model and serialize all inference artifacts."""
import json
import time
import importlib.metadata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features.config import (
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    RANDOM_SEED,
)

out_dir = Path("artifacts")
out_dir.mkdir(parents=True, exist_ok=True)


def fit_and_save_final_artifacts(
    train_file: Path = Path("results/features/train_features.csv.gz"),
    tuning_file: Path = Path("results/tuning/best_hyperparameters.json"),
):
    print("Loading full training dataset for final model fitting...")
    t0 = time.time()
    df = pd.read_csv(train_file)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int8)

    # Load tuned parameters if available
    if tuning_file.exists():
        with open(tuning_file) as f:
            tuning_data = json.load(f)
        params = dict(tuning_data.get("best_params", {}))
        print(f"Loaded tuned hyperparameters from {tuning_file}")
    else:
        params = {
            "max_iter": 100,
            "learning_rate": 0.08,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
        }
        print("Using default HistGradientBoosting parameters.")

    params["class_weight"] = "balanced"
    params["random_state"] = RANDOM_SEED

    print(f"Fitting final HistGradientBoostingClassifier on {len(df):,} candidate rows...")
    model = HistGradientBoostingClassifier(**params)
    model.fit(X, y)
    train_time = time.time() - t0
    print(f"Final model fitted in {train_time:.1f}s.")

    # 1. Save trained model
    model_path = out_dir / "final_model.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path} ({model_path.stat().st_size:,} bytes)")

    # 2. Save feature configuration & ordering
    config_path = out_dir / "feature_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "feature_columns": FEATURE_COLUMNS,
            "identifier_columns": IDENTIFIER_COLUMNS,
            "target_column": TARGET_COLUMN,
            "random_seed": RANDOM_SEED,
            "feature_metadata": FEATURE_METADATA,
        }, f, indent=2)
    print(f"Saved feature config to {config_path}")

    # 3. Save comprehensive metadata
    packages = ["scikit-learn", "numpy", "pandas", "optuna", "joblib"]
    pkg_versions = {}
    for pkg in packages:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pkg_versions[pkg] = "unknown"

    meta_path = out_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model_type": "HistGradientBoostingClassifier",
            "model_family": "Gradient Boosted Decision Trees",
            "hyperparameters": params,
            "training_samples": len(df),
            "training_trips": int(df["trip_id"].nunique()),
            "training_time_sec": round(train_time, 2),
            "random_seed": RANDOM_SEED,
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
            "package_versions": pkg_versions,
        }, f, indent=2)
    print(f"Saved model metadata to {meta_path}")

    return model_path


if __name__ == "__main__":
    fit_and_save_final_artifacts()
