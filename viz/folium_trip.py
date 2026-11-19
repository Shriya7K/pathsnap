import sys
from pathlib import Path
import pandas as pd
import folium

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
partition = sys.argv[2] if len(sys.argv) > 2 else "train"
trip = sys.argv[3] if len(sys.argv) > 3 else "00000001"
d = root / partition / trip


def read(ext):
    return pd.read_csv(d / (trip + ext), sep="\t", header=None)


trk, nodes, arcs, route = read(".track"), read(".nodes"), read(".arcs"), read(".route")
nlon, nlat = nodes[0].to_numpy(), nodes[1].to_numpy()
a_from, a_to = arcs[0].to_numpy(), arcs[1].to_numpy()
rt = route[0].to_numpy()

m = folium.Map(location=[trk[1].mean(), trk[0].mean()], zoom_start=14, tiles=None)

# 1. Esri World Street Map - Clean, high-detail street tiles (Zero watermarks, no API key needed)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri &mdash; Street Map",
    name="Esri Street Map (Clean, No Watermark)",
    max_zoom=19,
    show=True,
).add_to(m)

# 2. Esri Satellite Imagery (Zero watermarks, no API key needed)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri &mdash; Satellite Imagery",
    name="Esri Satellite (Aerial View)",
    max_zoom=19,
    show=False,
).add_to(m)

# 3. Pure White Canvas (Zero basemap clutter, maximum contrast for road network)
folium.TileLayer(
    tiles="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
    attr="Clean Vector Canvas",
    name="Clean Canvas (No Basemap)",
    max_zoom=19,
    show=False,
).add_to(m)

# Road network: candidate arcs near the track
pad = 0.005
ok = ((nlon[a_from] > trk[0].min() - pad) & (nlon[a_from] < trk[0].max() + pad) &
      (nlat[a_from] > trk[1].min() - pad) & (nlat[a_from] < trk[1].max() + pad))
idx = ok.nonzero()[0][:15000]
net = folium.FeatureGroup(name="Road network (STRtree candidates)", show=True)
segs = [[(nlat[a_from[i]], nlon[a_from[i]]), (nlat[a_to[i]], nlon[a_to[i]])] for i in idx]
folium.PolyLine(segs, color="#555555", weight=2.0, opacity=0.75).add_to(net)
net.add_to(m)

# Reference route (ground truth sequence)
rg = folium.FeatureGroup(name="Reference route (Ground Truth)", show=True)
rsegs = [[(nlat[a_from[i]], nlon[a_from[i]]), (nlat[a_to[i]], nlon[a_to[i]])] for i in rt]
folium.PolyLine(rsegs, color="#00aa00", weight=5, opacity=0.9).add_to(rg)
rg.add_to(m)

# GPS points (subsampled 1/5 for responsiveness)
gg = folium.FeatureGroup(name="GPS observations (subsampled 1/5)", show=True)
for lon, lat in trk[[0, 1]].to_numpy()[::5]:
    folium.CircleMarker((lat, lon), radius=3, color="#e60000", fill=True, fill_color="#ff3333", fill_opacity=0.9).add_to(gg)
gg.add_to(m)

legend = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
padding: 10px 14px; border: 2px solid #555; border-radius: 6px; font-size: 13px; box-shadow: 2px 2px 8px rgba(0,0,0,0.3);">
<b>PathSnap Ground Truth Alignment</b><br>
<span style="color:#555555; font-weight:bold;">&#9644;&#9644;</span> Road network (candidates)<br>
<span style="color:#00aa00; font-weight:bold;">&#9644;&#9644;</span> Reference route (Ground Truth)<br>
<span style="color:#e60000; font-weight:bold;">&#9679;</span> GPS observations
</div>"""
m.get_root().html.add_child(folium.Element(legend))
folium.LayerControl(collapsed=False).add_to(m)

Path("results/eda").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(parents=True, exist_ok=True)

out_eda = f"results/eda/map_{partition}_{trip}.html"
out_res = f"results/map_{partition}_{trip}.html"
m.save(out_eda)
m.save(out_res)
print(f"Saved {out_eda} and {out_res} | points: {len(trk)} | route arcs: {len(rt)} | network arcs drawn: {len(idx)}")
