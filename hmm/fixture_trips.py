"""SYNTHETIC FIXTURE: tiny grid trips in the real data layout, for tests and dry runs. NOT real data.

Builds, under <out>/:
    data/<partition>/<trip>/<trip>.track|.nodes|.arcs|.route      (tab-separated, like the real data)
    candidates/<partition>_<trip>_cands.csv.gz, ..._points.csv.gz  (P1's file layout; candidates come from
                                                                    P1's own candidate_gen.generate)
    probs.csv                                                      (a P2-style probability file)
Every trip is a random walk on a 6x6 two-way grid with noisy GPS. Nothing here is a result.

    python -m hmm.fixture_trips --out <dir>
    python -m hmm.run_all --probs <dir>/probs.csv --label "synthetic fixture" --data-root <dir>/data \
           --cands-dir <dir>/candidates --out-root <dir>/results --no-png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_scores import distance_softmax
from .graph import REPO

LON0, LAT0, STEP = 77.5, 13.0, 0.001        # grid spacing ~ 108-111 m
ROWS = COLS = 6


def grid():
    nodes = np.array([[LON0 + c * STEP, LAT0 + r * STEP] for r in range(ROWS) for c in range(COLS)])
    arcs = []
    for r in range(ROWS):
        for c in range(COLS):
            u = r * COLS + c
            if c + 1 < COLS:
                arcs += [(u, u + 1), (u + 1, u)]
            if r + 1 < ROWS:
                arcs += [(u, u + COLS), (u + COLS, u)]
    return nodes, np.array(arcs)


def random_walk(arcs, rng, n_arcs=12):
    """Arc ids of a walk that never immediately reverses."""
    out_of = {}
    for i, (u, v) in enumerate(arcs):
        out_of.setdefault(int(u), []).append(i)
    cur = int(rng.integers(len(arcs)))
    path = [cur]
    while len(path) < n_arcs:
        opts = [a for a in out_of[int(arcs[cur][1])] if not (arcs[a][1] == arcs[cur][0])]
        cur = int(rng.choice(opts))
        path.append(cur)
    return path


def gps_along(nodes, arcs, path, rng, spacing_m=14.0, noise_m=6.0):
    pts, true = [], []
    for a in path:
        p, q = nodes[arcs[a][0]], nodes[arcs[a][1]]
        n = max(int(round(110.0 / spacing_m)), 1)
        for k in range(n):
            pts.append(p + (q - p) * (k + 0.5) / n)
            true.append(a)
    pts = np.array(pts) + rng.normal(0, noise_m / 111_000.0, (len(pts), 2))
    return pts, np.array(true)


def write_trip(out, part, trip, nodes, arcs, path, pts, true, ts, cands_dir):
    """Write one trip's data files and P1-layout candidate files. Returns the label-free candidates."""
    sys.path.insert(0, str(REPO))
    from data_processing.candidate_gen import generate
    d = Path(out) / "data" / part / trip
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / f"{trip}.track", np.column_stack([pts[:, 0], pts[:, 1], ts]), fmt="%.9f", delimiter="\t")
    np.savetxt(d / f"{trip}.nodes", nodes, fmt="%.9f", delimiter="\t")
    np.savetxt(d / f"{trip}.arcs", arcs, fmt="%d", delimiter="\t")
    np.savetxt(d / f"{trip}.route", np.array(path), fmt="%d", delimiter="\t")
    td = generate(d)                                         # P1's label-free candidate generator
    c = td.cands.copy()
    tr = true[c["point_idx"].to_numpy()] if len(c) and len(true) else np.array([], dtype=int)
    c["label"] = (c["arc_id"].to_numpy() == tr).astype(int)
    c["label_valid"] = 1
    c["trip"] = trip
    cands_dir.mkdir(parents=True, exist_ok=True)
    c.to_csv(cands_dir / f"{part}_{trip}_cands.csv.gz", index=False)
    d_true = np.zeros(len(pts))
    hit = c[c["label"] == 1]
    d_true[hit["point_idx"].to_numpy()] = hit["dist"].to_numpy()
    route_pos = np.array([path.index(int(a)) for a in true]) if len(true) else np.array([], dtype=int)
    pd.DataFrame({"point_idx": np.arange(len(pts)), "x": td.xy[:, 0], "y": td.xy[:, 1], "ts": td.ts,
                  "route_pos": route_pos, "true_arc": true, "d_true": d_true,
                  "label_valid": 1}).to_csv(cands_dir / f"{part}_{trip}_points.csv.gz", index=False)
    return c[["point_idx", "arc_id", "dist", "rank"]]


def build_fixture(out, n_train=3, n_student=2, seed=0, empty_trip=False):
    """Create the fixture. Returns dict(data_root, cands_dir, probs (DataFrame), trips [(partition, id)])."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    cands_dir = out / "candidates"
    nodes, arcs = grid()
    trips = [("train", f"{i:08d}") for i in range(1, n_train + 1)] + \
            [("student_test", f"{i:08d}") for i in range(11, 11 + n_student)]
    parts = []
    for part, trip in trips:
        path = random_walk(arcs, rng)
        pts, true = gps_along(nodes, arcs, path, rng)
        ts = np.arange(1.0, len(pts) + 1)
        c = write_trip(out, part, trip, nodes, arcs, path, pts, true, ts, cands_dir)
        p = distance_softmax(c, 20.0)                        # P2-style scores: a distance softmax (synthetic)
        p.insert(0, "trip_id", trip)
        parts.append(p)
    if empty_trip:                                           # a trip far away from its network: no candidates
        trip = f"{max(int(t) for _, t in trips) + 1:08d}"
        path = random_walk(arcs, rng)
        pts, true = gps_along(nodes, arcs, path, rng)
        pts[:, 1] += 0.5                                     # ~55 km north of the network
        write_trip(out, "train", trip, nodes, arcs, path, pts, true, np.arange(1.0, len(pts) + 1), cands_dir)
        trips.append(("train", trip))
    probs = pd.concat(parts, ignore_index=True)
    return {"data_root": out / "data", "cands_dir": cands_dir, "probs": probs, "trips": trips}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print("*** SYNTHETIC FIXTURE - tiny grid trips, not real data, not results ***")
    fx = build_fixture(a.out, seed=a.seed)
    fx["probs"].to_csv(Path(a.out) / "probs.csv", index=False)
    print(f"trips: {fx['trips']}\ndata: {fx['data_root']}\ncandidates: {fx['cands_dir']}\nprobs: {Path(a.out) / 'probs.csv'}")


if __name__ == "__main__":
    main()
