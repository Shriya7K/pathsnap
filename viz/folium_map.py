"""Interactive Folium map of one trip (Person 3).

Layers: raw GPS (blue) | road network near the track (grey) | reference route (green, dev
partitions only) | classifier-only route (orange) | HMM route (red).

Python:  make_map(trip_id, predicted_route, out_path, partition, classifier_route=None)
CLI:     python -m viz.folium_map --probs <probs.csv.gz> --partition student_test --n 5 [--png]
PNG:     export_png("results/maps/x.html", "results/maps/x.png")   (needs Edge/Chrome installed)
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import folium
import numpy as np

from hmm.evaluate import SYNTH_BANNER, SYNTH_DIR, classifier_only, is_synthetic
from hmm.graph import DATA_ROOT, DEV_PARTITIONS, REPO, build_network, load_trip, norm_trip_id
from hmm.probs_io import CANDS_DIR, add_args, load_from_args
from hmm.viterbi import decode_detailed, dedupe, load_config

MAPS_DIR = REPO / "results" / "maps"
COLORS = {"gps": "#1f77b4", "net": "#9a9a9a", "ref": "#2ca02c", "cls": "#ff7f0e", "hmm": "#d62728"}


def _segments(route, nodes, arcs):
    """Arc ids -> list of [(lat,lon),(lat,lon)] segments (kept separate so breaks stay visible)."""
    r = np.asarray(route, dtype=int)
    r = r[(r >= 0) & (r < len(arcs))]
    a, b = nodes[arcs[r, 0]], nodes[arcs[r, 1]]
    return [[(p[1], p[0]), (q[1], q[0])] for p, q in zip(a, b)]


def _network_near_track(gps, nodes, arcs, buffer_m, cap):
    """Indices of arcs inside the track's bounding box + buffer (bounding box in degrees is
    only used to decide what to DRAW; no distances are computed from it). Undirected duplicates
    are dropped and the buffer shrinks until at most `cap` segments remain."""
    lat0 = gps["lat"].mean()
    for buf in (buffer_m, buffer_m / 2, buffer_m / 4, 0.0):
        dlat = buf / 111_320.0
        dlon = buf / (111_320.0 * max(np.cos(np.radians(lat0)), 0.1))
        lo, hi = gps[["lon", "lat"]].min().to_numpy() - [dlon, dlat], gps[["lon", "lat"]].max().to_numpy() + [dlon, dlat]
        p, q = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
        inside = ((p >= lo) & (p <= hi)).all(1) | ((q >= lo) & (q <= hi)).all(1)
        idx = np.nonzero(inside)[0]
        keep = arcs[idx].min(1) * (len(nodes) + 1) + arcs[idx].max(1)      # same street, both directions
        idx = idx[np.unique(keep, return_index=True)[1]]
        if len(idx) <= cap:
            return idx
    return idx[:cap]


def make_map(trip_id, predicted_route, out_path, partition, classifier_route=None,
             data_root=DATA_ROOT, show_reference=True, banner=None, banner_color="#d9534f",
             max_points=3000, buffer_m=400.0, max_network_segments=25000):
    """Write an interactive map to out_path and return it.
    predicted_route: list of arc ids (HMM route). classifier_route: optional list (orange layer).
    The reference route is drawn only for dev partitions (train, student_test)."""
    trip_id = norm_trip_id(trip_id)
    draw_ref = show_reference and partition in DEV_PARTITIONS          # never for evaluator trips
    gps, nodes, arcs, ref = load_trip(partition, trip_id, data_root, with_route=draw_ref)

    m = folium.Map(location=[gps["lat"].mean(), gps["lon"].mean()], zoom_start=14, tiles="OpenStreetMap",
                   control_scale=True)

    net_idx = _network_near_track(gps, nodes, arcs, buffer_m, max_network_segments)
    layers = [("Road network (near track)", COLORS["net"], _segments(net_idx, nodes, arcs), 1, 0.6, True)]
    if draw_ref and ref is not None:
        layers.append(("Reference route", COLORS["ref"], _segments(ref, nodes, arcs), 7, 0.55, True))
    if classifier_route is not None:
        layers.append(("Classifier-only route", COLORS["cls"], _segments(classifier_route, nodes, arcs), 4, 0.85, True))
    layers.append(("HMM route", COLORS["hmm"], _segments(predicted_route, nodes, arcs), 3, 0.95, True))
    # draw order = stacking order: network, then GPS points, then route layers on top
    gg = folium.FeatureGroup(name="GPS points", show=True)
    step = max(1, int(np.ceil(len(gps) / max_points)))                 # thin long tracks to keep file small
    for lon, lat, t in gps[["lon", "lat", "t"]].to_numpy()[::step]:
        folium.CircleMarker((lat, lon), radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=1, tooltip=f"t={t:.0f}s").add_to(gg)

    for n_layer, (name, color, segs, weight, opacity, show) in enumerate(layers):
        if n_layer == 1:
            gg.add_to(m)
        fg = folium.FeatureGroup(name=f"{name} ({len(segs)} arcs)", show=show)
        if segs:
            folium.PolyLine(segs, color=color, weight=weight, opacity=opacity).add_to(fg)
        fg.add_to(m)

    entries = [("gps", "&#9679;", f"GPS points (every {step})" if step > 1 else "GPS points"),
               ("net", "&#9644;", "Road network near track")]
    if draw_ref and ref is not None:
        entries.append(("ref", "&#9644;", "Reference route"))
    if classifier_route is not None:
        entries.append(("cls", "&#9644;", "Classifier-only route"))
    entries.append(("hmm", "&#9644;", "HMM route"))
    rows = "".join(f'<div><span style="color:{COLORS[k]};font-size:16px">{sym}</span> {txt}</div>'
                   for k, sym, txt in entries)
    title = f"{partition}/{trip_id}"
    html = (f'<div style="position:fixed;bottom:28px;left:12px;z-index:9999;background:white;padding:8px 12px;'
            f'border:1px solid #888;border-radius:4px;font:13px sans-serif"><b>{title}</b>{rows}</div>')
    if banner:
        html += (f'<div style="position:fixed;top:10px;left:60px;right:60px;z-index:9999;background:#fff3cd;'
                 f'border:2px solid {banner_color};padding:6px 10px;font:bold 13px sans-serif;text-align:center">{banner}</div>')
    m.get_root().html.add_child(folium.Element(html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[gps["lat"].min(), gps["lon"].min()], [gps["lat"].max(), gps["lon"].max()]])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return m


def _find_browser():
    for name in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        p = shutil.which(name)
        if p:
            return p
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
        if Path(p).exists():
            return p
    return None


def export_png(html_path, png_path, size=(1400, 900), wait_ms=12000):
    """Screenshot an HTML map with headless Edge/Chrome. Returns True on success; otherwise
    prints how to screenshot manually. (Map tiles need internet access while rendering.)"""
    html_path, png_path = Path(html_path).resolve(), Path(png_path).resolve()
    browser = _find_browser()
    if browser:
        cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={size[0]},{size[1]}",
               f"--virtual-time-budget={wait_ms}", f"--screenshot={png_path}", html_path.as_uri()]
        try:
            subprocess.run(cmd, timeout=120, capture_output=True)
            if png_path.exists() and png_path.stat().st_size > 0:
                return True
        except Exception as e:
            print("headless screenshot failed:", e)
    print(f"Could not auto-screenshot. Manually: open {html_path} in a browser, wait for the map tiles "
          f"to load, then press Win+Shift+S (Windows) / Cmd+Shift+4 (Mac) and save as {png_path}.")
    return False


def render_maps(by_trip, trips, out_dir, cfg, label=None, synthetic=False, png=True,
                data_root=DATA_ROOT, n=5, png_fn=None):
    """Render maps for the first n of `trips` [(partition, trip_id)] and return [(html, png_or_None, trip)].
    by_trip: {trip_id: probs DataFrame}. Trips with no candidates are skipped (nothing to draw)."""
    png_fn = png_fn or export_png
    done = []
    for part, trip in trips:
        if len(done) >= n:
            break
        pt = by_trip.get(trip)
        if pt is None or len(pt) == 0:
            continue
        gps, nodes, arcs, _ = load_trip(part, trip, data_root, with_route=False)
        net = build_network(gps, nodes, arcs, cache_key=f"{part}_{trip}")
        res = decode_detailed(gps, nodes, arcs, pt, cfg, network=net)
        cls = dedupe(x for x in classifier_only(pt, len(gps)) if x is not None)
        slug = "".join(c if c.isalnum() else "_" for c in label.lower()) if label else ""
        suffix = "_SYNTHETIC" if synthetic else (f"_{slug}" if label else "")
        out = Path(out_dir) / f"{part}_{trip}{suffix}.html"
        make_map(trip, res.route, out, part, classifier_route=cls, data_root=data_root,
                 banner=SYNTH_BANNER if synthetic else (f"Score source: {label}" if label else None),
                 banner_color="#d9534f" if synthetic else "#1f77b4")
        print(f"Saved {out}  (HMM route {len(res.route)} arcs, classifier-only {len(cls)} arcs)")
        ok = png_fn(out, out.with_suffix(".png")) if png else None
        done.append((out, out.with_suffix(".png") if ok else None, trip))
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render maps: classifier-only vs HMM route for N trips")
    ap.add_argument("--probs", required=True, help="csv / csv.gz / parquet: trip_id, point_idx, arc_id, prob")
    ap.add_argument("--partition", default="student_test", choices=list(DEV_PARTITIONS))
    ap.add_argument("--trips", nargs="*")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out-dir", default=None, help="default results/maps (synthetic: results/hmm_synthetic/maps)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--label", default=None, help='score source shown on the map + in the file name')
    ap.add_argument("--png", action="store_true", help="also export a PNG screenshot per map")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--cands-dir", default=str(CANDS_DIR))
    add_args(ap)
    a = ap.parse_args(argv)

    synth = is_synthetic(a.probs)
    out_dir = Path(a.out_dir) if a.out_dir else (SYNTH_DIR / "maps" if synth else MAPS_DIR)
    if synth:
        print(SYNTH_BANNER)
    lp = load_from_args(a, [a.partition], a.data_root, a.cands_dir)
    trips = lp.trips
    if a.trips:
        want = {norm_trip_id(t) for t in a.trips}
        trips = [t for t in trips if t[1] in want]
    render_maps(lp.by_trip, trips, out_dir, load_config(a.config), a.label, synth, a.png, a.data_root, a.n)


if __name__ == "__main__":
    main()
