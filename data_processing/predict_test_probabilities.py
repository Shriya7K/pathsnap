
"""Generate candidate probabilities for PathSnap student_test trips."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("models/final_model.joblib")
METADATA_PATH = Path("models/final_model_metadata.json")

INPUT_DIR = Path("results/test_features")
OUTPUT_DIR = Path("results/predictions")

CHUNK_SIZE = 5000


def main():
    print("=" * 60)
    print("PATHSNAP - TEST PROBABILITY PREDICTION")
    print("=" * 60)

    # Check required files.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load trained model and feature metadata.
    model = joblib.load(MODEL_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_columns = metadata["features"]

    if model.n_features_in_ != len(feature_columns):
        raise ValueError("Model and metadata feature counts do not match.")

    if 1 not in model.classes_:
        raise ValueError("Positive class (1) not found in model classes.")

    positive_class_index = list(model.classes_).index(1)

    input_files = sorted(INPUT_DIR.glob("student_test_*.csv.gz"))

    if len(input_files) != 10:
        raise ValueError(
            f"Expected 10 test feature files, found {len(input_files)}."
        )

    print("Model:", type(model).__name__)
    print("Features:", len(feature_columns))
    print("Test trips:", len(input_files))

    total_rows = 0
    processed = 0

    for input_path in input_files:
        print(f"\nProcessing {input_path.name}...")

        data = pd.read_csv(input_path)

        missing_columns = [
            col for col in feature_columns if col not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing features in {input_path.name}: {missing_columns}"
            )

        if data.empty:
            raise ValueError(f"Empty feature file: {input_path.name}")

        if data.duplicated(["point_idx", "arc_id"]).any():
            raise ValueError(
                f"Duplicate candidate rows in {input_path.name}"
            )

        X = data[feature_columns]

        if X.isna().any().any():
            raise ValueError(f"Missing feature values in {input_path.name}")

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(
                f"Non-finite feature values in {input_path.name}"
            )

        # Predict in chunks to control memory usage.
        probabilities = []

        for start in range(0, len(X), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(X))

            batch = X.iloc[start:end]
            batch_probabilities = model.predict_proba(batch)[
                :, positive_class_index
            ]

            probabilities.append(batch_probabilities)

        probabilities = np.concatenate(probabilities)

        if len(probabilities) != len(data):
            raise ValueError("Prediction row count does not match input.")

        if not np.isfinite(probabilities).all():
            raise ValueError("Predictions contain non-finite values.")

        if not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise ValueError("Predictions outside probability range [0, 1].")

        # Preserve all candidate metadata and append probability.
        data["probability"] = probabilities

        output_path = OUTPUT_DIR / input_path.name

        data.to_csv(
            output_path,
            index=False,
            compression="gzip",
        )

        total_rows += len(data)
        processed += 1

        print("Candidate rows:", f"{len(data):,}")
        print(
            "Probability range:",
            f"{probabilities.min():.6f} - {probabilities.max():.6f}",
        )
        print("Saved:", output_path)

    print("\n========== PREDICTION SUMMARY ==========")
    print("Trips processed:", processed)
    print("Total predictions:", f"{total_rows:,}")
    print("Output directory:", OUTPUT_DIR)
    print("Probability column: probability")
    print("\nTEST PROBABILITY PREDICTION COMPLETE")


if __name__ == "__main__":
    main()
