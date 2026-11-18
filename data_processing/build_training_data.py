
from pathlib import Path

import pandas as pd

from candidate_gen import generate
from features import build_features, FEATURE_COLUMNS
from load_candidates import list_trips, load_trip

OUTPUT_DIR = Path("results/training_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_rows = 0
total_positive = 0
total_negative = 0
processed = 0

for partition, trip in list_trips():

    # Only training trips. Never train on student_test.
    if partition != "train":
        continue

    print(f"\nProcessing trip {trip}...")

    td = generate(Path("data") / partition / trip)
    X = build_features(td)

    # Load labels separately.
    labels = load_trip(partition, trip)[
        ["point_idx", "arc_id", "label", "label_valid"]
    ]

    # Join using GPS point and candidate arc identity.
    data = X.merge(
        labels,
        on=["point_idx", "arc_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(X):
        raise ValueError(f"Row mismatch in trip {trip}")

    # Keep only reliable labels.
    data = data[data["label_valid"] == 1].copy()

    # Preserve trip identity for GroupKFold.
    data["trip"] = trip

    # Save features and target, not unused ground-truth fields.
    output_columns = (
        ["trip", "point_idx", "arc_id"]
        + FEATURE_COLUMNS
        + ["label"]
    )

    data[output_columns].to_csv(
        OUTPUT_DIR / f"train_{trip}.csv.gz",
        index=False,
        compression="gzip",
    )

    positives = int(data["label"].sum())
    negatives = len(data) - positives

    total_rows += len(data)
    total_positive += positives
    total_negative += negatives
    processed += 1

    print(
        f"Rows: {len(data):,} | "
        f"Positive: {positives:,} | "
        f"Negative: {negatives:,} | SAVED"
    )

print("\n========== TRAINING DATA SUMMARY ==========")
print("Trips processed:", processed)
print("Valid candidate rows:", f"{total_rows:,}")
print("Positive candidates:", f"{total_positive:,}")
print("Negative candidates:", f"{total_negative:,}")

if total_rows:
    print(
        "Positive ratio:",
        f"{100 * total_positive / total_rows:.2f}%"
    )

print("\nTRAINING DATA GENERATION COMPLETE")
