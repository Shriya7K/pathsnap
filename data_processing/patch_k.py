from pathlib import Path

p = Path("data_processing/candidates.py")
s = p.read_text()

edits = [
    ("R_MAX, K_MAX = 150.0, 20", "R_MAX, K_MAX = 150.0, 50"),
    ("GRID = [(5, 30), (5, 50), (10, 50), (10, 100), (20, 100), (20, 150)]",
     "GRID = [(5, 50), (10, 100), (20, 100), (30, 100), (50, 100), (50, 150)]"),
]

for old, new in edits:
    assert s.count(old) == 1, f"Could not find exactly once: {old[:50]}"
    s = s.replace(old, new)

p.write_text(s)
print("Patched OK")