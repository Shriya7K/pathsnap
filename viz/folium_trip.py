import sys
from pathlib import Path
import pandas as pd
import folium

root = Path(sys.argv[1])
partition = sys.argv[2]
trip = sys.argv[3]
d = root / partition / trip


def read(ext):
    return pd.read_csv(d / (trip + ext), sep="\t", header=None)


trk, nodes, arcs, route = read(".track"), read(".nodes"), read(".arcs"), read(".route")
nlon, nlat = nodes[0].to_numpy(), nodes[1].to_numpy()
a_from, a_to = arcs[0].to_numpy(), arcs[1].to_numpy()
rt = route[0].to_numpy()

m = folium.Map(location=[trk[1].mean(), trk[0].mean()], zoom_start=14, tiles=None)

# network: only arcs near the track, capped so the map stays light
pad = 0.003
ok = ((nlon[a_from] > trk[0].min() - pad) & (nlon[a_from] < trk[0].max() + pad) &
      (nlat[a_from] > trk[1].min() - pad) & (nlat[a_from] < trk[1].max() + pad))
idx = np_idx = ok.nonzero()[0][:15000]
net = folium.FeatureGroup(name="Road network", show=True)
segs = [[(nlat[a_from[i]], nlon[a_from[i]]), (nlat[a_to[i]], nlon[a_to[i]])] for i in idx]
folium.PolyLine(segs, color="#888888", weight=1, opacity=0.6).add_to(net)
net.add_to(m)

# reference route
rg = folium.FeatureGroup(name="Reference route", show=True)
rsegs = [[(nlat[a_from[i]], nlon[a_from[i]]), (nlat[a_to[i]], nlon[a_to[i]])] for i in rt]
folium.PolyLine(rsegs, color="green", weight=5, opacity=0.8).add_to(rg)
rg.add_to(m)

# GPS points (every 5th to keep it light)
gg = folium.FeatureGroup(name="GPS points", show=True)
for lon, lat in trk[[0, 1]].to_numpy()[::5]:
    folium.CircleMarker((lat, lon), radius=2, color="red", fill=True, fill_opacity=0.8).add_to(gg)
gg.add_to(m)

legend = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
padding: 8px 12px; border: 1px solid #999; font-size: 13px;">
<b>Legend</b><br>
<span style="color:#888888;">&#9644;</span> Road network<br>
<span style="color:green;">&#9644;</span> Reference route<br>
<span style="color:red;">&#9679;</span> GPS points (every 5th)
</div>"""
m.get_root().html.add_child(folium.Element(legend))
folium.LayerControl().add_to(m)

Path("results").mkdir(exist_ok=True)
out = f"results/map_{partition}_{trip}.html"
m.save(out)
print("Saved", out, "| points:", len(trk), "| route arcs:", len(rt), "| network arcs drawn:", len(idx))