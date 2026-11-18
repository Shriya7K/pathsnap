from pathlib import Path

p = Path("data_processing/candidates.py")
s = p.read_text()

edits = [
    ("    d_true = pt_seg_dist(xy, A[j], B[j])\n",
     "    d_true = pt_seg_dist(xy, A[j], B[j])\n    valid = d_true <= 50.0\n"),
    ("    c[\"label\"] = (c[\"arc_id\"].to_numpy() == true_arc[c[\"point_idx\"].to_numpy()]).astype(int)\n",
     "    c[\"label\"] = (c[\"arc_id\"].to_numpy() == true_arc[c[\"point_idx\"].to_numpy()]).astype(int)\n"
     "    c[\"label_valid\"] = valid[c[\"point_idx\"].to_numpy()].astype(int)\n"),
    ("\"avg_cands\": round(len(c) / len(xy), 2)}",
     "\"avg_cands\": round(len(c) / len(xy), 2), \"n_valid\": int(valid.sum())}"),
    ("int(((t_rank >= K) | (t_dist > R)).sum())",
     "int((((t_rank >= K) | (t_dist > R)) & valid).sum())"),
    ("\"d_true\": d_true}", "\"d_true\": d_true, \"label_valid\": valid}"),
    ("n = sm.n_points.sum()\n", "n = sm.n_points.sum()\nnv = sm.n_valid.sum()\n"),
    ("'].sum() / n * 100:.2f}", "'].sum() / nv * 100:.2f}"),
    ("sm[\"miss_K10_R100\"] / sm.n_points * 100",
     "sm[\"miss_K10_R100\"] / sm.n_valid.clip(lower=1) * 100"),
]

for old, new in edits:
    assert s.count(old) == 1, f"Could not find exactly once: {old[:50]}"
    s = s.replace(old, new)

p.write_text(s)
print("Patched OK")