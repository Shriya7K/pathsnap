
"""Generate label-free features for PathSnap student_test trips."""

from pathlib import Path
import numpy as np

from candidate_gen import generate
from features import build_features, FEATURE_COLUMNS
from load_candidates import list_trips

OUTPUT_DIR = Path("results/test_features")
DATA_DIR = Path("data")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover test trips from existing candidate files.
    all_trips = list_trips()
    test_trips = [
        (partition, trip)
        for partition, trip in all_trips
        if partition == "student" and trip.startswith("test_")
    ]

    if not test_trips:
        raise RuntimeError(
            "No student_test trips found. Check candidate filenames."
        )

    print("=" * 60)
    print("PATHSNAP - TEST FEATURE GENERATION")
    print("=" * 60)
    print("Test trips found:", len(test_trips))

    total_rows = 0
    processed = 0

    for partition, trip in test_trips:
        print(f"\nProcessing {partition}/{trip}...")

        # Convert candidate ID to the actual dataset directory.
        # Example: test_00000000 -> data/student_test/00000000
        trip_id = trip.replace("test_", "", 1)
        trip_dir = DATA_DIR / "student_test" / trip_id

        if not trip_dir.exists():
            raise FileNotFoundError(
                f"Trip directory not found: {trip_dir}"
            )

        # Generate candidates and reconstruct network geometry.
        # This does not access route labels.
        td = generate(trip_dir)

        # Build the same 14 features used during training.
        features = build_features(td)

        if features.empty:
            print("WARNING: No candidate rows. Skipping.")
            continue

        output = features[
            ["point_idx", "arc_id"] + FEATURE_COLUMNS
        ].copy()

        output.insert(0, "trip", trip)

        # Validate feature values.
        if output[FEATURE_COLUMNS].isna().any().any():
            raise ValueError(f"Missing feature values in {trip}")

        if not np.isfinite(
            output[FEATURE_COLUMNS].to_numpy()
        ).all():
            raise ValueError(f"Non-finite feature values in {trip}")

        if output.duplicated(["point_idx", "arc_id"]).any():
            raise ValueError(f"Duplicate candidate rows in {trip}")

        output_path = OUTPUT_DIR / f"student_test_{trip_id}.csv.gz"

        output.to_csv(
            output_path,
            index=False,
            compression="gzip",
        )

        total_rows += len(output)
        processed += 1

        print(f"GPS points: {len(td.xy):,}")
        print(f"Candidate rows: {len(output):,}")
        print(f"Features: {len(FEATURE_COLUMNS)}")
        print("Saved:", output_path)

    print("\n========== TEST FEATURE SUMMARY ==========")
    print("Trips processed:", processed)
    print("Total candidate rows:", f"{total_rows:,}")
    print("Feature columns:", len(FEATURE_COLUMNS))
    print("\nTEST FEATURE GENERATION COMPLETE")


if __name__ == "__main__":
    main()
