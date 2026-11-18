import sys
import time
from pathlib import Path
import numpy as np
from candidate_gen import generate
from load_candidates import load_trip

root = Path(sys.argv[1])
trips = sys.argv[2:] or ["00000001", "00000072", "00000033", "00000000"]
all_ok = True

for trip in trips:
    track = next(root.rglob(f"{trip}.track"))
    partition = track.relative_to(root).parts[0]

    t0 = time.time()
    td = generate(track.parent)
    secs = time.time() - t0

    ref = load_trip(partition, trip)[["point_idx", "arc_id", "dist", "rank"]]
    a = td.cands.sort_values(["point_idx", "rank"]).reset_index(drop=True)
    b = ref.sort_values(["point_idx", "rank"]).reset_index(drop=True)

    same_rows = len(a) == len(b)
    same_dist = same_rows and np.allclose(a.dist.to_numpy(), b.dist.to_numpy())
    merged = a.merge(b, on=["point_idx", "arc_id"])
    same_arcs = same_rows and len(merged) == len(a)
    ok = same_rows and same_dist and same_arcs
    all_ok &= ok
    print(f"{partition}/{trip}: rows new={len(a)} old={len(b)} | "
          f"same distances: {same_dist} | same arcs: {same_arcs} | {secs:.1f}s | "
          f"{'OK' if ok else 'MISMATCH'}")

print("\nALL OK" if all_ok else "\nSOME MISMATCHES - paste this output to me")