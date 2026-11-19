# PathSnap

Map matching for noisy GPS traces. Given a vehicle's GPS points and a road network, PathSnap predicts which road segment (arc) the vehicle was on, and reconstructs a connected route.

## Pipeline

1. Candidate generation: nearest arcs within a radius per GPS point
2. Feature engineering: distance, heading, speed, arc context
3. ML classifier: scores each (point, candidate) pair
4. HMM + Viterbi: decodes scores into a connected route
5. Serving: FastAPI backend + UI, with Folium map visualisation

## Team

* Shriya K
* Shriranjini
* Sujal T H
* Sujan

## Setup, usage, deployment



\## Data interface (Person 1 -> others)



Run once after cloning (put the dataset in `data/`):

&#x20;   python data\_processing/candidates.py data



Load with `data\_processing/load\_candidates.py` (`load\_trip(partition, trip)`).

Defaults: K=30 nearest arcs, within 100 m. Candidate miss rate 0.02% on valid labels.



Columns: point\_idx, arc\_id, dist (m), rank (0 = nearest), label (1 = reference arc),

label\_valid, x, y (UTM metres), ts.



Rules:

\- Train only on rows with label\_valid == 1. About 10% of points are off the

&#x20; reference route (e.g. long stationary periods at the end of trips 54 and 33) and have no valid label.

\- Never use label, route\_pos, true\_arc or d\_true as features.

\- Split by trip (GroupKFold), never by row.

\- Arc ids are 0-based and local to each trip. Never merge networks across trips.

\- Time gaps can be huge (trips 90 and 78). Treat a large gap as a break in the HMM.

