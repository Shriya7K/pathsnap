
from pathlib import Path

from candidate_gen import generate
from features import build_features, FEATURE_COLUMNS

root = Path("data/train")
total_points = 0
total_rows = 0

trip_dirs = sorted(p for p in root.iterdir() if p.is_dir())

for trip_dir in trip_dirs:
    td = generate(trip_dir)
    X = build_features(td)

    assert len(X) == len(td.cands), f"Row mismatch: {td.trip}"
    assert X[FEATURE_COLUMNS].notna().all().all(), f"Missing values: {td.trip}"

    total_points += len(td.xy)
    total_rows += len(X)

    print(f"{td.trip}: {len(td.xy)} points | {len(X)} feature rows | OK")

print("\nALL TRAINING TRIPS PASSED")
print("Total GPS points:", total_points)
print("Total feature rows:", total_rows)
print("Feature count:", len(FEATURE_COLUMNS))
