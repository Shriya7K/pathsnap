from pathlib import Path

old = 'c = c.sort_values(["point_idx", "dist"], kind="stable")'
new = 'c = c.sort_values(["point_idx", "dist", "arc_id"], kind="stable")'

for name in ["candidates.py", "candidate_gen.py"]:
    p = Path("data_processing") / name
    s = p.read_text()
    assert s.count(old) == 1, f"{name}: expected exactly one match"
    p.write_text(s.replace(old, new))
    print("patched", name)