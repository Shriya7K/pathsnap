"""Classifier probability generation for Person 3 HMM & Viterbi Decoder."""
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from features.config import FEATURE_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMN

out_dir = Path("results/probabilities")
out_dir.mkdir(parents=True, exist_ok=True)


def generate_probabilities_for_partition(
    partition: str = "student_test",
    feature_file: Path = None,
    model_path: Path = Path("artifacts/final_model.joblib"),
    out_dir: Path = out_dir,
) -> Path:
    if feature_file is None:
        feature_file = Path(f"results/features/{partition}_features.csv.gz")

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run fit_final_model.py first.")
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature dataset not found at {feature_file}. Run build_features.py first.")

    print(f"Loading {partition} features from {feature_file}...")
    df = pd.read_csv(feature_file)
    print(f"Loaded {len(df):,} candidate rows. Loading model from {model_path}...")
    model = joblib.load(model_path)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    t0 = time.time()
    probs = model.predict_proba(X)[:, 1]
    infer_time = time.time() - t0
    print(f"Inference complete on {len(df):,} candidates in {infer_time:.2f}s ({len(df)/max(infer_time,0.01):,.0f} rows/s).")

    # Verify probability validity in [0, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Invalid predicted probability outside [0, 1]!"

    out_df = pd.DataFrame({
        "trip_id": df["trip_id"],
        "point_idx": df["point_idx"],
        "arc_id": df["arc_id"],
        "predicted_probability": probs.round(6),
        "predicted_label": (probs >= 0.5).astype(np.int8),
    })

    if "perpendicular_distance" in df.columns:
        out_df["perpendicular_distance"] = df["perpendicular_distance"]
    if "distance_rank" in df.columns:
        out_df["distance_rank"] = df["distance_rank"]

    # Include true_label if available for offline evaluation
    if TARGET_COLUMN in df.columns:
        out_df["true_label"] = df[TARGET_COLUMN].astype(np.int8)

    out_file = out_dir / f"person3_classifier_probabilities_{partition}.csv.gz"
    out_df.to_csv(out_file, index=False, compression="gzip")
    print(f"Saved Person 3 probability dataset to {out_file} ({out_file.stat().st_size:,} bytes)")
    return out_file


def predict_trip_probabilities(trip_id: str, partition: str = "student_test", model_path: Path = Path("artifacts/final_model.joblib")) -> pd.DataFrame:
    """Python API for Person 3: Returns candidate probabilities for a single trip.

    Returns:
    --------
    pd.DataFrame with columns:
        - trip_id (str)
        - point_idx (int)
        - arc_id (int)
        - predicted_probability (float) [0.0, 1.0]
        - predicted_label (int) {0, 1}
        - perpendicular_distance (float)
        - distance_rank (int)
    """
    model = joblib.load(model_path)
    from features.engineer import extract_trip_features
    df = extract_trip_features(partition=partition, trip_id=trip_id, valid_only=False)
    if len(df) == 0:
        return pd.DataFrame(columns=["trip_id", "point_idx", "arc_id", "predicted_probability", "predicted_label"])

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    probs = model.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "trip_id": df["trip_id"],
        "point_idx": df["point_idx"],
        "arc_id": df["arc_id"],
        "predicted_probability": probs.round(6),
        "predicted_label": (probs >= 0.5).astype(np.int8),
        "perpendicular_distance": df["perpendicular_distance"],
        "distance_rank": df["distance_rank"],
    })


if __name__ == "__main__":
    generate_probabilities_for_partition("student_test")
    generate_probabilities_for_partition("train")
