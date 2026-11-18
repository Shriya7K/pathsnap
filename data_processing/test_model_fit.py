
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

X = np.random.default_rng(42).normal(size=(500, 14))
y = np.array([0] * 450 + [1] * 50)

weights = compute_sample_weight(
    class_weight="balanced",
    y=y,
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(16,),
        max_iter=5,
        random_state=42,
    )),
])

model.fit(
    X,
    y,
    model__sample_weight=weights,
)

print("MLP pipeline fit successful.")
print("Predictions:", model.predict_proba(X[:5])[:, 1])
