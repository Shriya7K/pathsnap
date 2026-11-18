
from candidate_gen import generate
from features import build_features, FEATURE_COLUMNS
from load_candidates import load_trip

PARTITION = "train"
TRIP = "00000001"

# Generate features without accessing ground truth.
td = generate(f"data/{PARTITION}/{TRIP}")
X = build_features(td)

# Load labels separately.
labels = load_trip(PARTITION, TRIP)[
    ["point_idx", "arc_id", "label", "label_valid"]
]

# Join labels to features using candidate identity.
data = X.merge(
    labels,
    on=["point_idx", "arc_id"],
    how="inner",
    validate="one_to_one",
)

# Keep only reliable training labels.
valid_data = data[data["label_valid"] == 1].copy()

print("Trip:", TRIP)
print("Feature rows:", len(X))
print("Label rows:", len(labels))
print("Joined rows:", len(data))
print("Valid rows:", len(valid_data))
print("Positive labels:", int(valid_data["label"].sum()))
print("Negative labels:", int((valid_data["label"] == 0).sum()))

assert len(X) == len(labels) == len(data)
assert set(FEATURE_COLUMNS).issubset(data.columns)
assert not set([
    "label", "label_valid", "route_pos", "true_arc", "d_true"
]).intersection(FEATURE_COLUMNS)

assert valid_data["label"].isin([0, 1]).all()

print("\nTRAINING LABEL JOIN PASSED")
