
"""Train and save the final tuned Random Forest model."""

from pathlib import Path
import gc
import json

import joblib
from sklearn.ensemble import RandomForestClassifier

from train_models import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TRAIN_ROW_LIMITS,
    get_trip_files,
    load_training_sample,
)

OUTPUT_DIR = Path("models")
MODEL_PATH = OUTPUT_DIR / "final_model.joblib"
METADATA_PATH = OUTPUT_DIR / "final_model_metadata.json"

BEST_PARAMS = {
    "n_estimators": 180,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = get_trip_files()

    print("=" * 60)
    print("PATHSNAP - FINAL RANDOM FOREST TRAINING")
    print("=" * 60)
    print(f"Training trips: {len(files)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    row_limit = TRAIN_ROW_LIMITS["Random Forest"]

    X_train, y_train = load_training_sample(
        files,
        row_limit,
        seed=RANDOM_STATE,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Positive ratio: {y_train.mean():.2%}")

    model = RandomForestClassifier(
        **BEST_PARAMS,
        class_weight="balanced_subsample",
        n_jobs=2,
        random_state=RANDOM_STATE,
    )

    print("\nTraining final model...")
    model.fit(X_train, y_train)

    print("Training completed.")

    joblib.dump(model, MODEL_PATH, compress=3)

    metadata = {
        "model": "RandomForestClassifier",
        "features": FEATURE_COLUMNS,
        "hyperparameters": BEST_PARAMS,
        "training_trips": len(files),
        "training_rows": len(X_train),
        "positive_ratio": float(y_train.mean()),
        "random_state": RANDOM_STATE,
        "probability_method": "predict_proba",
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nFinal model saved:", MODEL_PATH)
    print("Metadata saved:", METADATA_PATH)

    del X_train, y_train, model
    gc.collect()

    print("\nFINAL TRAINING COMPLETE")


if __name__ == "__main__":
    main()
