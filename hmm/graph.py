"""Per-trip road graph for the HMM decoder (Person 3).

Every trip has its own road network, so we build one `TripNetwork` per trip.

What it gives the decoder
  * node coordinates and arc lengths in METRES (projected to the trip's UTM zone,
    never Euclidean distance on degrees)
  * directed arc adjacency: arc A -> arc B  iff  A.to_node == B.from_node
  * bounded Dijkstra: shortest road distance from a node to every node within a limit
    (cached, because the same candidate arcs recur at consecutive GPS points)
  * a shortest arc path between two arcs (used to fill gaps in the final route)

CLI:  python -m hmm.graph --partitions train student_test
      -> results/hmm_graph_stats.csv  (arcs, nodes, connected components per trip)
"""
import argparse
import heapq
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEV_PARTITIONS = ("train", "student_test")   # the only partitions development code may touch
LIMIT_STEP_M = 250.0                          # Dijkstra limits are rounded up to this (cache reuse)


# --------------------------------------------------------------------------- file loading
def norm_trip_id(x):
    """Trip ids are 8-digit zero-padded folder names; CSVs may have lost the padding."""
    return str(int(x)).zfill(8) if str(x).strip().isdigit() else str(x)


def find_partition(trip_id, data_root=DATA_ROOT):
    """Return the partition folder that contains this trip (searching dev partitions only)."""
    trip_id = norm_trip_id(trip_id)
    for part in DEV_PARTITIONS:
        if (Path(data_root) / part / trip_id).is_dir():
            return part
    raise FileNotFoundError(f"trip {trip_id} not found in {DEV_PARTITIONS} under {data_root}")


def load_trip(partition, trip_id, data_root=DATA_ROOT, with_route=True):
    """Read one trip. Returns gps DataFrame (lon, lat, t), nodes (N,2), arcs (M,2), route or None."""
    trip_id = norm_trip_id(trip_id)
    d = Path(data_root) / partition / trip_id

    def rd(ext, dtype=None):
        return pd.read_csv(d / (trip_id + ext), sep=r"\s+", header=None, dtype=dtype)

    trk = rd(".track")
    gps = pd.DataFrame({"lon": trk[0].astype(float), "lat": trk[1].astype(float),
                        "t": trk[2].astype(float)})
    nodes = rd(".nodes").to_numpy(dtype=float)
    arcs = rd(".arcs").to_numpy(dtype=np.int64)
    route = None
    if with_route and (d / (trip_id + ".route")).exists():
        route = rd(".route")[0].to_numpy(dtype=np.int64)
    return gps, nodes, arcs, route


def utm_epsg(lon, lat):
    """EPSG code of the UTM zone containing (lon, lat) (same rule as P1's candidates.py)."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


# --------------------------------------------------------------------------- the network
class TripNetwork:
    """Directed road graph of one trip, in metres."""

    def __init__(self, nodes, arcs, lon0, lat0, cache_key=None):
        """nodes (N,2) lon/lat, arcs (M,2) from/to node ids (row index = arc id).
        (lon0, lat0) picks the UTM zone (use the GPS track mean).
        cache_key (e.g. 'train_00000001') enables an on-disk cache of the projected network."""
        self.nodes_ll = np.asarray(nodes, dtype=float)
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.n_nodes, self.n_arcs = len(self.nodes_ll), len(self.arcs)
        self.epsg = utm_epsg(lon0, lat0)
        self._tr = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self.from_node = self.arcs[:, 0]
        self.to_node = self.arcs[:, 1]

        self.node_xy = self._load_or_project(cache_key)
        d = self.node_xy[self.to_node] - self.node_xy[self.from_node]
        self.arc_len = np.hypot(d[:, 0], d[:, 1])          # metres, straight segment node->node

        # CSR "outgoing arcs per node": arcs sorted by from_node; out_ptr[u]..out_ptr[u+1]
        order = np.argsort(self.from_node, kind="stable")
        self.out_ptr = np.searchsorted(self.from_node[order], np.arange(self.n_nodes + 1))
        self.out_arc = order
        # plain python lists are faster than numpy scalars inside the Dijkstra loop
        self._ptr = self.out_ptr.tolist()
        self._oto = self.to_node[order].tolist()
        self._olen = self.arc_len[order].tolist()
        self._oid = order.tolist()
        self._reach_cache = {}                              # node -> (limit, {node: dist})

    # ---- projection (with cache)
    def _load_or_project(self, cache_key):
        crc = zlib.crc32(self.nodes_ll.tobytes()) ^ zlib.crc32(self.arcs.tobytes())
        f = CACHE_DIR / f"{cache_key}.npz" if cache_key else None
        if f is not None and f.exists():
            try:
                z = np.load(f)
                if int(z["crc"]) == crc and int(z["epsg"]) == self.epsg:
                    return z["node_xy"]
            except Exception:
                pass                                         # corrupt cache -> rebuild
        x, y = self._tr.transform(self.nodes_ll[:, 0], self.nodes_ll[:, 1])
        xy = np.column_stack([np.asarray(x), np.asarray(y)])
        if f is not None:
            CACHE_DIR.mkdir(exist_ok=True)
            np.savez(f, node_xy=xy, crc=crc, epsg=self.epsg)
        return xy

    def project(self, lon, lat):
        """lon/lat arrays -> (n,2) metres in this trip's UTM zone."""
        x, y = self._tr.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.column_stack([np.asarray(x), np.asarray(y)])

    # ---- arc adjacency
    def successors(self, arc):
        """Arcs that can directly follow `arc` (A.to_node == B.from_node)."""
        u = int(self.to_node[arc])
        return self.out_arc[self.out_ptr[u]:self.out_ptr[u + 1]]

    def connected(self, a, b):
        """True if arc b can directly follow arc a (or they are the same arc)."""
        return a == b or self.to_node[a] == self.from_node[b]

    def project_on_arc(self, xy, arcs):
        """Offset (m from the arc's start) of each point's projection onto its arc segment.
        xy (n,2), arcs (n,) -> offsets (n,), clipped to [0, arc_len]."""
        a = self.node_xy[self.from_node[arcs]]
        ab = self.node_xy[self.to_node[arcs]] - a
        l2 = (ab ** 2).sum(1)
        t = ((xy - a) * ab).sum(1) / np.where(l2 == 0, 1.0, l2)
        return np.clip(t, 0.0, 1.0) * self.arc_len[arcs]

    # ---- bounded shortest paths
    def reach(self, src_node, limit):
        """Shortest road distance (m) from src_node to every node within `limit` metres.
        Returned dict may include nodes slightly beyond `limit` (cache reuse): callers
        must compare distances to their own limit. Cached per node."""
        lim = math.ceil(limit / LIMIT_STEP_M) * LIMIT_STEP_M
        hit = self._reach_cache.get(src_node)
        if hit is not None and hit[0] >= lim:
            return hit[1]
        dist = self._dijkstra(src_node, lim)
        self._reach_cache[src_node] = (lim, dist)
        return dist

    def _dijkstra(self, src, limit):
        ptr, oto, olen = self._ptr, self._oto, self._olen
        dist = {src: 0.0}
        heap = [(0.0, src)]
        pop, push = heapq.heappop, heapq.heappush
        while heap:
            d, u = pop(heap)
            if d > dist[u]:
                continue                                      # stale heap entry
            for e in range(ptr[u], ptr[u + 1]):
                nd = d + olen[e]
                if nd > limit:
                    continue
                v = oto[e]
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    push(heap, (nd, v))
        return dist

    def shortest_arc_path(self, arc_a, arc_b, limit):
        """Arcs strictly BETWEEN arc_a and arc_b on the shortest road path (arc_a.to_node ->
        arc_b.from_node), or None if no path within `limit` metres. [] if directly adjacent."""
        src, dst = int(self.to_node[arc_a]), int(self.from_node[arc_b])
        if src == dst:
            return []
        ptr, oto, olen, oid = self._ptr, self._oto, self._olen, self._oid
        dist, prev = {src: 0.0}, {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                path = []
                while u != src:                               # walk predecessors back
                    e = prev[u]
                    path.append(e)
                    u = int(self.from_node[e])
                return path[::-1]
            if d > dist[u]:
                continue
            for e in range(ptr[u], ptr[u + 1]):
                nd = d + olen[e]
                v = oto[e]
                if nd <= limit and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = oid[e]
                    heapq.heappush(heap, (nd, v))
        return None

    # ---- statistics
    def stats(self):
        """Graph statistics for results/hmm_graph_stats.csv."""
        n, m = self.n_nodes, self.n_arcs
        g = coo_matrix((np.ones(m), (self.from_node, self.to_node)), shape=(n, n)).tocsr()
        nw, lw = connected_components(g, directed=True, connection="weak")
        ns, ls = connected_components(g, directed=True, connection="strong")
        arc_comp_w = np.bincount(lw[self.from_node], minlength=nw)
        arc_comp_s = np.bincount(ls[self.from_node][ls[self.from_node] == ls[self.to_node]],
                                 minlength=ns)
        return {
            "n_nodes": n, "n_arcs": m,
            "arc_len_mean_m": round(float(self.arc_len.mean()), 2),
            "arc_len_max_m": round(float(self.arc_len.max()), 2),
            "n_zero_len_arcs": int((self.arc_len == 0).sum()),
            "mean_out_degree": round(m / n, 3),
            "weak_components": int(nw),
            "largest_weak_pct_arcs": round(100 * arc_comp_w.max() / m, 2),
            "strong_components": int(ns),
            "largest_strong_pct_arcs": round(100 * arc_comp_s.max() / m, 2),
        }


def build_network(gps, nodes, arcs, cache_key=None):
    """Convenience: pick the UTM zone from the GPS track mean."""
    return TripNetwork(nodes, arcs, float(np.mean(gps["lon"])), float(np.mean(gps["lat"])), cache_key)


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Graph statistics per trip -> results/hmm_graph_stats.csv")
    ap.add_argument("--partitions", nargs="+", default=list(DEV_PARTITIONS))
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default=str(REPO / "results" / "hmm_graph_stats.csv"))
    a = ap.parse_args()
    if "evaluator_test" in a.partitions:
        raise SystemExit("evaluator_test must not be used in development code.")
    rows = []
    for part in a.partitions:
        for d in sorted((Path(a.data_root) / part).iterdir()):
            if not d.is_dir():
                continue
            gps, nodes, arcs, _ = load_trip(part, d.name, a.data_root, with_route=False)
            net = build_network(gps, nodes, arcs, cache_key=f"{part}_{d.name}")
            row = {"partition": part, "trip_id": d.name, "epsg": net.epsg, **net.stats()}
            rows.append(row)
            print(f"{part}/{d.name}: arcs={row['n_arcs']} weak={row['weak_components']} "
                  f"strong={row['strong_components']} largest_strong={row['largest_strong_pct_arcs']}%")
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nSaved {a.out} ({len(df)} trips)")


if __name__ == "__main__":
    main()
