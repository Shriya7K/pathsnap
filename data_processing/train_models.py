"""Train and compare five ML models using trip-level GroupKFold.

Run from the project root:
    python data_processing/train_models.py

Outputs:
    results/model_training/fold_results.csv
    results/model_training/model_comparison.csv
"""

from pathlib import Path
import gc
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("results/training_features")
OUTPUT_DIR = Path("results/model_training")

N_SPLITS = 5
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "dist",
    "log_dist",
    "rank",
    "arc_length",
    "projection_ratio",
    "dist_to_start",
    "dist_to_end",
    "gps_speed",
    "gps_move_distance",
    "time_delta",
    "heading_diff",
    "heading_alignment",
    "heading_valid",
    "gps_heading_change",
]

# Conservative row limits to control RAM and runtime.
# Sampling always keeps all candidate rows belonging to a selected GPS point.
TRAIN_ROW_LIMITS = {
    "Logistic Regression": 60000,
    "KNN": 15000,
    "Random Forest": 50000,
    "HistGradientBoosting": 60000,
    "MLP": 30000,
}

PREDICTION_CHUNK_SIZE = 5000


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def get_trip_files():
    files = sorted(DATA_DIR.glob("train_*.csv.gz"))
    if not files:
        raise FileNotFoundError(
            f"No training files found in {DATA_DIR}. "
            "Run build_training_data.py first."
        )
    return files


def trip_id_from_file(path):
    return path.name.replace("train_", "").replace(".csv.gz", "")


def sample_complete_points(df, max_rows, rng):
    """Sample complete GPS-point candidate groups, not individual rows."""
    if len(df) <= max_rows:
        return df

    groups = df.groupby("point_idx", sort=False).indices
    point_ids = np.array(list(groups.keys()))
    rng.shuffle(point_ids)

    selected_indices = []
    row_count = 0

    for point_id in point_ids:
        indices = groups[point_id]
        if row_count + len(indices) > max_rows:
            continue

        selected_indices.extend(indices)
        row_count += len(indices)

        if row_count >= max_rows:
            break

    if not selected_indices:
        # Keep at least one complete point, even if it exceeds the limit.
        selected_indices = list(groups[point_ids[0]])

    return df.iloc[selected_indices].copy()


def check_required_columns(df, path, required):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )


def load_training_sample(train_files, row_limit, seed):
    """Load a bounded, trip-balanced sample from the training trips."""
    rng = np.random.default_rng(seed)
    sampled_parts = []
    per_trip_limit = max(1, row_limit // len(train_files))

    required = FEATURE_COLUMNS + ["point_idx", "arc_id", "label"]

    for path in train_files:
        df = pd.read_csv(path)
        check_required_columns(df, path, required)

        df = sample_complete_points(df, per_trip_limit, rng)
        sampled_parts.append(df[required].copy())
        del df

    data = pd.concat(sampled_parts, ignore_index=True)
    del sampled_parts

    X = data[FEATURE_COLUMNS].astype(np.float32)
    y = data["label"].astype(int)

    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Training features contain NaN or infinite values.")

    if y.nunique() < 2:
        raise ValueError(
            "Training sample contains only one class. "
            "Increase the corresponding TRAIN_ROW_LIMITS value."
        )

    print(
        f"Actual training rows: {len(data):,} | "
        f"Positive ratio: {y.mean():.2%}"
    )

    del data
    return X, y


# --------------------------------------------------
# Models
# --------------------------------------------------

def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=11,
                weights="distance",
                n_jobs=-1,
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.1,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),

        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=30,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# --------------------------------------------------
# Model fitting
# --------------------------------------------------

def fit_model(model, name, X, y):
    weights = compute_sample_weight(class_weight="balanced", y=y)

    if name == "HistGradientBoosting":
        model.fit(X, y, sample_weight=weights)
    elif name == "MLP":
        model.fit(X, y, model__sample_weight=weights)
    else:
        # Logistic Regression and Random Forest use their class_weight
        # settings. KNN has no native class_weight parameter.
        model.fit(X, y)

    return model


# --------------------------------------------------
# Validation
# --------------------------------------------------

def evaluate_trip(model, path):
    """
    Evaluate candidate ranking for one held-out trip.

    top1_accuracy:
        Correct top-ranked candidates / all GPS points represented
        in the candidate file.

    candidate_coverage:
        GPS points with at least one positive candidate / all
        represented GPS points.

    conditional_accuracy:
        Correct top-ranked candidates / points with a positive
        candidate available in the candidate set.

    ROC-AUC is calculated over candidate rows for this trip.
    """
    df = pd.read_csv(path)

    required = FEATURE_COLUMNS + ["point_idx", "dist", "label"]
    check_required_columns(df, path, required)

    if df.empty:
        raise ValueError(f"{path.name} contains no validation candidates.")

    positive_point_ids = set(
        df.loc[df["label"] == 1, "point_idx"].unique()
    )

    probabilities = np.empty(len(df), dtype=np.float32)

    # Predict in chunks so the full validation feature matrix is not
    # duplicated in memory.
    for start in range(0, len(df), PREDICTION_CHUNK_SIZE):
        end = min(start + PREDICTION_CHUNK_SIZE, len(df))
        X_chunk = df.iloc[start:end][FEATURE_COLUMNS].astype(np.float32)

        if X_chunk.isna().any().any() or not np.isfinite(
            X_chunk.to_numpy()
        ).all():
            raise ValueError(
                f"{path.name} contains NaN or infinite validation features."
            )

        probabilities[start:end] = model.predict_proba(X_chunk)[:, 1]
        del X_chunk

    df["probability"] = probabilities

    # Pick one candidate per GPS point: highest probability first,
    # then shortest GPS-to-candidate distance as a tie-breaker.
    ranked = df.sort_values(
        ["point_idx", "probability", "dist"],
        ascending=[True, False, True],
        kind="stable",
    )
    best = ranked.drop_duplicates(subset="point_idx", keep="first")

    total_points = len(best)
    points_with_positive = len(positive_point_ids)
    correct_points = int(best["label"].sum())

    conditional_correct = int(
        (
            (best["label"] == 1)
            & best["point_idx"].isin(positive_point_ids)
        ).sum()
    )

    auc = np.nan
    labels = df["label"].to_numpy(dtype=int)
    if len(np.unique(labels)) == 2:
        auc = roc_auc_score(labels, probabilities)

    metrics = {
        "total_points": total_points,
        "points_with_positive_candidate": points_with_positive,
        "correct_points": correct_points,
        "conditional_correct": conditional_correct,
        "auc": auc,
    }

    del df, ranked, best, probabilities, labels
    return metrics


# --------------------------------------------------
# Cross-validation
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = get_trip_files()
    trip_ids = [trip_id_from_file(path) for path in files]

    print("=" * 65)
    print("PATHSNAP - FIVE MODEL COMPARISON")
    print("=" * 65)
    print(f"Training trips: {len(files)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Cross-validation folds: {N_SPLITS}")
    print("Execution: FULL RUN (all five models, all five folds)")

    if len(files) < N_SPLITS:
        raise ValueError("Not enough trips for GroupKFold.")

    splitter = GroupKFold(n_splits=N_SPLITS)
    models = get_models()
    results = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(trip_ids, groups=trip_ids),
        start=1,
    ):
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]

        print("\n" + "=" * 65)
        print(f"FOLD {fold}/{N_SPLITS}")
        print("=" * 65)
        print(f"Training trips: {len(train_files)}")
        print(f"Validation trips: {len(val_files)}")

        for model_name, model_template in models.items():
            print("\n" + "-" * 50)
            print(f"Training {model_name}")
            print("-" * 50)

            row_limit = TRAIN_ROW_LIMITS[model_name]
            X_train, y_train = load_training_sample(
                train_files,
                row_limit,
                seed=RANDOM_STATE + fold,
            )

            model = clone(model_template)
            model = fit_model(model, model_name, X_train, y_train)

            del X_train, y_train
            gc.collect()

            total_points = 0
            points_with_positive = 0
            correct_points = 0
            conditional_correct = 0
            auc_values = []

            for path in val_files:
                metrics = evaluate_trip(model, path)

                total_points += metrics["total_points"]
                points_with_positive += (
                    metrics["points_with_positive_candidate"]
                )
                correct_points += metrics["correct_points"]
                conditional_correct += metrics["conditional_correct"]

                if not np.isnan(metrics["auc"]):
                    auc_values.append(metrics["auc"])

                del metrics

            top1_accuracy = (
                correct_points / total_points if total_points else 0.0
            )
            candidate_coverage = (
                points_with_positive / total_points if total_points else 0.0
            )
            conditional_accuracy = (
                conditional_correct / points_with_positive
                if points_with_positive else 0.0
            )
            mean_auc = (
                float(np.mean(auc_values)) if auc_values else np.nan
            )

            result = {
                "fold": fold,
                "model": model_name,
                "training_row_limit": row_limit,
                "validation_points": total_points,
                "points_with_positive_candidate": points_with_positive,
                "correct_points": correct_points,
                "candidate_coverage": candidate_coverage,
                "top1_accuracy": top1_accuracy,
                "conditional_accuracy": conditional_accuracy,
                "mean_trip_auc": mean_auc,
            }
            results.append(result)

            print(f"Validation points: {total_points:,}")
            print(f"Candidate coverage: {candidate_coverage:.4f}")
            print(f"Top-1 accuracy: {top1_accuracy:.4f}")
            print(f"Conditional accuracy: {conditional_accuracy:.4f}")
            print(f"Mean trip ROC-AUC: {mean_auc:.4f}")

            # Save progress after every model so completed runs are retained.
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "fold_results.csv",
                index=False,
            )

            del model
            gc.collect()

    # --------------------------------------------------
    # Final comparison
    # --------------------------------------------------

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No model results were produced.")

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
        .sort_values("mean_top1_accuracy", ascending=False)
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)
    print(summary.to_string(index=False))

    print("\nResults saved in:")
    print(OUTPUT_DIR.resolve())
    print("\nCROSS-VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
