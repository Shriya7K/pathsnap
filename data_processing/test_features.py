
from candidate_gen import generate
from features import build_features, FEATURE_COLUMNS

td = generate("data/train/00000001")
X = build_features(td)

print("Trip:", td.trip)
print("Candidate rows:", len(td.cands))
print("Feature rows:", len(X))
print("Feature count:", len(FEATURE_COLUMNS))
print("\nFeatures:")
print(X.head())
print("\nMissing values:", X[FEATURE_COLUMNS].isna().sum().sum())

assert len(X) == len(td.cands)
assert X[FEATURE_COLUMNS].notna().all().all()
assert "label" not in X.columns
assert "label_valid" not in X.columns
assert "route_pos" not in X.columns
assert "true_arc" not in X.columns

print("\nFEATURE TEST PASSED")
