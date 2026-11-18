# PathSnap 
 
Map matching for noisy GPS traces. Given a vehicle's GPS points and a road network, PathSnap predicts which road segment (arc) the vehicle was on, and reconstructs a connected route. 
 
## Pipeline 
1. Candidate generation: nearest arcs within a radius per GPS point 
2. Feature engineering: distance, heading, speed, arc context 
3. ML classifier: scores each (point, candidate) pair 
4. HMM + Viterbi: decodes scores into a connected route 
5. Serving: FastAPI backend + UI, with Folium map visualisation 
 
## Team 
- Name 1 - role 
- Name 2 - role 
 
## Setup, usage, deployment 
(to be filled in as we build) 
