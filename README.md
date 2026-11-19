# PathSnap

Map matching for noisy GPS traces. Given a vehicle's GPS points and a road network, PathSnap predicts which road segment (arc) the vehicle was on, and reconstructs a connected route.

## Pipeline

1. **Candidate Generation & Spatial Preprocessing (Person 1)**: Metric UTM projection, STRtree indexing, label alignment, candidate generation & miss-rate evaluation.
2. **Feature Engineering (Person 2)**: Distance, heading, speed, network context, and pairwise features.
3. **ML Classifier & Scoring (Person 3)**: Supervised scoring of (point, candidate arc) pairs.
4. **HMM + Viterbi Global Decoding (Person 4)**: Transition probabilities, topological routing, and route reconstruction.
5. **Serving & Visualization**: FastAPI backend + UI, with interactive Folium maps.

---

## Team

* Shriya K
* Shriranjini
* Sujal T H
* Sujan

---

## Person 1: Data Processing & Candidate Generation

### 1. Overview & Responsibilities
Person 1 is responsible for:
- Dataset inventory and exploratory data analysis (EDA).
- Geographic preprocessing and projection (per-trip metric UTM zones).
- Spatial indexing using Shapely `STRtree`.
- Candidate generation across multiple configurations ($K \in [5, 50]$, radius $\in [50, 150]\,\text{m}$).
- Dynamic programming ground-truth label alignment and off-route stationarity filtering.
- Candidate miss-rate evaluation and trade-off analysis.
- Clean, documented data handoff for Person 2.

---

### 2. Dataset Inventory & Quality Summary
- **Partitions**: 80 training trips (`data/train/*`), 10 student-test trips (`data/student_test/*`).
- **Files per trip** (tab-separated, headerless):
  - `.track`: `longitude`, `latitude`, `timestamp`
  - `.nodes`: `longitude`, `latitude` (0-indexed node IDs by row)
  - `.arcs`: `from_node`, `to_node` (0-indexed arc IDs by row)
  - `.route`: Sequence of reference arc IDs representing the traversed route
- **Scale**:
  - Total GPS points: 227,577 (208,417 train, 19,160 student-test)
  - Total files: 360 files (95.58 MB)
  - Missing coordinates / NaNs: 0
  - Coordinate order: Col 0 = Longitude ($\in [-180, 180]$), Col 1 = Latitude ($\in [-90, 90]$)
  - Node ID and Arc ID indexing: 0-based across all 90 trips.
  - Route continuity: 99.94% consecutive connectivity across reference routes.

---

### 3. Geographic Preprocessing & Projections
To prevent distortion and avoid computing physical distances in angular degrees, each trip is projected into its corresponding local metric Universal Transverse Mercator (UTM) zone:
$$\text{UTM Zone} = \left\lfloor \frac{\text{mean\_lon} + 180}{6} \right\rfloor + 1$$
$$\text{EPSG} = (32600 \text{ if lat} \ge 0 \text{ else } 32700) + \text{zone}$$
Road network nodes and GPS observations for a trip are transformed into identical local UTM metric coordinates (meters Easting/Northing) using `pyproj.Transformer`. Trip networks are strictly isolated and never combined.

---

### 4. Ground-Truth Label Alignment & The >50m Stationarity Issue
#### Alignment Algorithm
The `.route` file provides an ordered sequence of traversed road arcs, not 1-to-1 GPS point labels. A dynamic programming (DP) algorithm aligns GPS points $i = 0 \dots N-1$ to route arc indices $j = 0 \dots M-1$:
$$\min_{\{j_i\}} \sum_{i=0}^{N-1} d_{\perp}(\mathbf{p}_i, \text{arc}_{j_i}) \quad \text{subject to } j_0 \le j_1 \le \dots \le j_{N-1}$$
This enforces monotonic route progression and prevents impossible backwards teleportation.

#### Investigation of the >50m Anomaly
Across the entire dataset, 23,295 observations (10.24%) have perpendicular distance $d_{\text{true}} > 50\,\text{m}$ from their aligned route arc:
- Over 96.9% of these points (22,574 points) occur in just 5 trips: `train/00000054` (17,274 pts), `train/00000033` (3,449 pts), `train/00000002` (868 pts), `train/00000084` (593 pts), and `train/00000066` (390 pts).
- **Root Cause**: GPS sensors continued recording for thousands of seconds after the vehicle parked at depots/parking lots beyond the end of the mapped reference route, exhibiting stationary GPS drift.
- **Handling**: These points are identified and flagged with `label_valid = 0`. Downstream models must filter `label_valid == 1` during training to prevent training on spurious off-route noise.

---

### 5. Candidate Generation & Spatial Indexing
- Arcs within the track's bounding box ($+ R_{\max}$ buffer) are indexed in a Shapely `STRtree`.
- For each GPS point, candidate arcs within radius $R$ are retrieved via `dwithin` queries.
- Orthogonal distances from GPS points to line segments are computed.
- Ties are broken deterministically by sorting on `(point_idx, dist, arc_id)`.
- Ranks are assigned from `0` (nearest) to `K-1`.

---

### 6. Candidate Miss-Rate Evaluation
Candidate miss rate measures the fraction of valid observations whose true reference arc failed to be captured within the candidate set:
$$\text{Candidate Miss Rate} = \frac{\text{Number of valid points whose true arc is absent}}{\text{Total valid labelled observations}}$$
$$\text{Recall} = 1.0 - \text{Miss Rate}$$

#### Experimental Results across Configurations (204,282 Valid Observations):
| Configuration | Radius (m) | Misses | Candidate Miss Rate | Candidate Recall | Avg Candidates / Point |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K = 5** | 50 m | 4,038 | 1.977 % | 98.023 % | 4.8 |
| **K = 10** | 100 m | 1,234 | 0.604 % | 99.396 % | 9.7 |
| **K = 20** | 100 m | 544 | 0.266 % | 99.734 % | 18.8 |
| **K = 30 (Default)** | 100 m | 43 | **0.021 %** | **99.979 %** | 26.5 |
| **K = 50** | 100 m | 1 | 0.000 % | 100.000 % | 35.1 |
| **K = 50** | 150 m | 1 | 0.000 % | 100.000 % | 38.2 |

**Recommendation for Person 2**: **$K=30$, $R=100\,\text{m}$** provides the optimal trade-off: an exceptional **99.98% recall** (only 43 misses out of 204,282 observations) while keeping candidate set size manageable ($26.5$ candidates per observation) for fast downstream feature engineering.

---

### 7. Data Interface & Handoff for Person 2

#### Storage & Format
Precomputed candidate sets are stored in `results/candidates/` as compressed CSV files (`.csv.gz`):
- `results/candidates/{partition}_{trip_id}_cands.csv.gz`: Candidate arc pairs for each GPS point.
- `results/candidates/{partition}_{trip_id}_points.csv.gz`: Point-level metadata and alignment diagnostics.

#### Schema Definitions
| Column Name | Data Type | Description |
| :--- | :---: | :--- |
| `trip_id` | `str` | Unique identifier of the trip (e.g. `'00000001'`) |
| `point_idx` | `int32` | 0-indexed sequential observation index within trip |
| `arc_id` | `int32` | 0-indexed local road arc identifier within the trip network |
| `label` | `int8` | Binary target: `1` if candidate is reference arc, `0` otherwise |
| `perpendicular_distance` | `float32` | Orthogonal distance in meters from GPS point to candidate arc |
| `distance_rank` | `int16` | Rank of candidate ordered by distance (`0` = closest arc) |
| `label_valid` | `int8` | `1` if ground truth is valid ($d_{\text{true}} \le 50\,\text{m}$), `0` if off-route |
| `x` | `float64` | GPS observation UTM Easting coordinate (meters) |
| `y` | `float64` | GPS observation UTM Northing coordinate (meters) |
| `ts` | `float64` | Timestamp of the GPS observation (seconds) |
| `dist` / `rank` | `float` / `int` | Backwards-compatibility aliases for `perpendicular_distance` and `distance_rank` |

#### How Person 2 Loads Data
Use the helper in `data_processing/load_candidates.py`:

```

---

## Person 2: Feature Engineering, Model Training & Bayesian Optimisation

### 1. Overview
Person 2 is responsible for:
- 19 modular features across geometric, kinematic, and network topology domains.
- Training 5 classification models across distinct families (Logistic Regression, KNN, Random Forest, HistGradientBoosting, MLP).
- Strict 5-fold `GroupKFold` cross-validation grouped by `trip_id` (zero spatial/temporal leakage).
- Observation-level Top-1 accuracy calculation grouping by `(trip_id, point_idx)`.
- Optuna Bayesian hyperparameter optimization (TPESampler) on HistGradientBoosting.
- Serialized production model artifacts (`artifacts/final_model.joblib`).

### 2. Experimental Results Summary

#### 5-Fold GroupKFold Cross-Validation (80 Training Trips, 2,013,074 Candidates)
| Model | Family | Mean CV Top-1 Acc | Mean CV Macro F1 | Mean Log Loss | Avg Train Time / Fold |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear | **79.32%** | 78.21% | 0.2193 | 1.86s |
| **Random Forest** | Bagging Ensemble | **78.41%** | **84.96%** | 0.1585 | 78.46s |
| **HistGradientBoosting** | Gradient Boosted Trees | **77.53%** | **83.65%** | 0.1733 | **9.39s (Fastest)** |
| **Multi-Layer Perceptron** | Neural Network | **76.85%** | **86.51%** | **0.1106** | 382.87s |
| **K-Nearest Neighbours** | Metric / Instance | **72.55%** | 80.30% | 0.1860 | 0.15s |

#### Held-Out Student-Test Evaluation (10 Trips, 220,026 Candidates)
- **Random Forest**: 81.92% Top-1 Accuracy | 85.52% Macro F1 | Loss: 0.1426
- **HistGradientBoosting**: 81.65% Top-1 Accuracy | 84.56% Macro F1 | Loss: 0.1535 (After Optuna: **82.15% Top-1**, **85.47% Macro F1**)
- **Logistic Regression**: 81.03% Top-1 Accuracy | 78.57% Macro F1 | Loss: 0.2045
- **Multi-Layer Perceptron**: 82.68% Top-1 Accuracy | 90.04% Macro F1 | Loss: 0.0792

---

## Person 3: Hidden Markov Model & Viterbi Sequence Decoding

### 1. Algorithm Overview
Implements the Newson & Krumm (2009) global sequence map-matching algorithm:
- **Emission Probabilities**: Person 2's trained ML classifier provides $P_{\text{ML}}(\text{arc}_i \mid \text{GPS}_t)$, fused with orthogonal Gaussian distance decay:
  $$\ln P_{\text{emiss}}(j) = \ln P_{\text{ML}}(j) - \frac{d_{\perp}^2}{2\sigma_z^2}$$
- **Transition Probabilities**: Road network topological shortest path routing via single-source Dijkstra:
  $$P_{\text{trans}}(i \to j) = \frac{1}{\beta} \exp\left( - \frac{|d_{\text{net}}(u, v) - d_{\text{gps}}|}{\beta} \right)$$
- **Viterbi Trellis**: Solves the dynamic programming trellis in log-space, elevating raw point-level accuracy (~82%) to **97.64% sequence route accuracy**!

---

## Person 4: Backend API, Distributed Multi-Node UI & Technical Report

### 1. Two-Laptop Distributed Architecture (As Mandated by Event Guidelines)
- **Host Server (Laptop 1)**: Runs the FastAPI backend service (`api/app.py`) on port `8000`. Houses candidate spatial extraction, machine learning inference, and the HMM Viterbi trellis engine. Serves the **Central Server Monitor** dashboard (`http://localhost:8000/server`), displaying connected client terminals, real-time inbound request feeds, and auto-resetting the mirrored map on client disconnect.
- **Client Terminals (Laptop 2 & Other Laptops)**: Accesses `http://<SERVER_IP>:8000/client` across campus Wi-Fi. Features a clean field submission portal with Student Folder benchmark trips (`student_test` and `train`), drag-and-drop `.track` file upload, and automatic 4-second client heartbeat telemetry.

### 2. REST API Endpoints
- `GET /`: Serves Server Monitor on localhost or Client Portal for external network clients.
- `GET /server`: Serves the dedicated Server Monitor Dashboard (Laptop 1).
- `GET /client`: Serves the dedicated Client Submission Portal (Laptop 2).
- `GET /api/health`: Health status and pipeline model verification.
- `GET /api/trips`: Lists trips prioritizing `student_test` and `train` from the Student Folder.
- `GET /api/trip/{partition}/{trip_id}`: Retrieves raw GPS points for a benchmark trip.
- `POST /api/match`: Full pipeline HMM Viterbi matching on GPS trajectory points.
- `POST /api/upload-and-match`: Uploads a custom `.track` or `.csv` file and snaps it to the network.
- `GET /api/server-stats`: Server telemetry endpoint for active clients and request history.
- `GET /api/client-heartbeat`: 4-second heartbeat signal from active client browsers.
- `POST /api/client-disconnect`: Window unload beacon to trigger clean slate resets.

### 3. Execution Commands
```powershell
# Step 1: Start Server on Laptop 1:
.\run.bat
# (or: python -u run_server.py)

# Step 2: Open Client Portal on Laptop 2 (Client):
# Open Chrome / Edge and navigate to: http://<SERVER_IP>:8000/client

# Step 3: Stream Live Server Logs in CMD (Laptop 1):
.\view_logs.bat

# Step 4: Re-compile 6-Page Technical Report PDF (Deliverable 2):
python build_competition_report_pdf.py
```

### 4. Competition Deliverables
- **Deliverable 1 (GitHub Repository)**: Clean, modular source code with `data_processing/`, `features/`, `models/`, `tuning/`, `hmm/`, `api/`, `ui/`, `viz/`, `artifacts/`, and `results/`.
- **Deliverable 2 (Technical Report PDF)**: [PATHSNAP_TECHNICAL_REPORT.pdf](PATHSNAP_TECHNICAL_REPORT.pdf) formatted strictly following official sections (2.2, 3, 4, 5, 6, 7, 8).
python
from data_processing.load_candidates import load_trip, load_all

# Load single trip (with default K=30, R=100m, valid_only=False)
df_trip = load_trip("train", "00000001", k=30, r=100.0)

# Load entire training set for feature engineering and ML training (only valid labels)
train_df = load_all(partition="train", k=30, r=100.0, valid_only=True)
```

---

## Person 2: Feature Engineering, Model Training & Bayesian Optimisation

### 1. Overview
Person 2 is responsible for:
- 19 modular features across geometric, kinematic, and network topology domains.
- Training 5 classification models across distinct families (Logistic Regression, KNN, Random Forest, HistGradientBoosting, MLP).
- Strict 5-fold `GroupKFold` cross-validation grouped by `trip_id` (zero spatial/temporal leakage).
- Observation-level Top-1 accuracy calculation grouping by `(trip_id, point_idx)`.
- Optuna Bayesian hyperparameter optimization (TPESampler) on HistGradientBoosting.
- Serialized production model artifacts (`artifacts/final_model.joblib`).

### 2. Experimental Results Summary

#### 5-Fold GroupKFold Cross-Validation (80 Training Trips, 2,013,074 Candidates)
| Model | Family | Mean CV Top-1 Acc | Mean CV Macro F1 | Mean Log Loss | Avg Train Time / Fold |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear | **79.32%** | 78.21% | 0.2193 | 1.86s |
| **Random Forest** | Bagging Ensemble | **78.41%** | **84.96%** | 0.1585 | 78.46s |
| **HistGradientBoosting** | Gradient Boosted Trees | **77.53%** | **83.65%** | 0.1733 | **9.39s (Fastest)** |
| **Multi-Layer Perceptron** | Neural Network | **76.85%** | **86.51%** | **0.1106** | 382.87s |
| **K-Nearest Neighbours** | Metric / Instance | **72.55%** | 80.30% | 0.1860 | 0.15s |

#### Held-Out Student-Test Evaluation (10 Trips, 220,026 Candidates)
- **Random Forest**: 81.92% Top-1 Accuracy | 85.52% Macro F1 | Loss: 0.1426
- **HistGradientBoosting**: 81.65% Top-1 Accuracy | 84.56% Macro F1 | Loss: 0.1535 (After Optuna: **82.15% Top-1**, **85.47% Macro F1**)
- **Logistic Regression**: 81.03% Top-1 Accuracy | 78.57% Macro F1 | Loss: 0.2045
- **Multi-Layer Perceptron**: 82.68% Top-1 Accuracy | 90.04% Macro F1 | Loss: 0.0792

---

## Person 3: Hidden Markov Model & Viterbi Sequence Decoding

### 1. Algorithm Overview
Implements the Newson & Krumm (2009) global sequence map-matching algorithm:
- **Emission Probabilities**: Person 2's trained ML classifier provides $P_{\text{ML}}(\text{arc}_i \mid \text{GPS}_t)$, fused with orthogonal Gaussian distance decay:
  $$\ln P_{\text{emiss}}(j) = \ln P_{\text{ML}}(j) - \frac{d_{\perp}^2}{2\sigma_z^2}$$
- **Transition Probabilities**: Road network topological shortest path routing via single-source Dijkstra:
  $$P_{\text{trans}}(i \to j) = \frac{1}{\beta} \exp\left( - \frac{|d_{\text{net}}(u, v) - d_{\text{gps}}|}{\beta} \right)$$
- **Viterbi Trellis**: Solves the dynamic programming trellis in log-space, elevating raw point-level accuracy (~82%) to **97.64% sequence route accuracy**!

---

## Person 4: Backend API, Distributed Multi-Node UI & Technical Report

### 1. Two-Laptop Distributed Architecture (As Mandated by Event Guidelines)
- **Host Server (Laptop 1)**: Runs the FastAPI backend service (`api/app.py`) on port `8000`. Houses candidate spatial extraction, machine learning inference, and the HMM Viterbi trellis engine. Serves the **Central Server Monitor** dashboard (`http://localhost:8000/server`), displaying connected client terminals, real-time inbound request feeds, and auto-resetting the mirrored map on client disconnect.
- **Client Terminals (Laptop 2 & Other Laptops)**: Accesses `http://<SERVER_IP>:8000/client` across campus Wi-Fi. Features a clean field submission portal with Student Folder benchmark trips (`student_test` and `train`), drag-and-drop `.track` file upload, and automatic 4-second client heartbeat telemetry.

### 2. REST API Endpoints
- `GET /`: Serves Server Monitor on localhost or Client Portal for external network clients.
- `GET /server`: Serves the dedicated Server Monitor Dashboard (Laptop 1).
- `GET /client`: Serves the dedicated Client Submission Portal (Laptop 2).
- `GET /api/health`: Health status and pipeline model verification.
- `GET /api/trips`: Lists trips prioritizing `student_test` and `train` from the Student Folder.
- `GET /api/trip/{partition}/{trip_id}`: Retrieves raw GPS points for a benchmark trip.
- `POST /api/match`: Full pipeline HMM Viterbi matching on GPS trajectory points.
- `POST /api/upload-and-match`: Uploads a custom `.track` or `.csv` file and snaps it to the network.
- `GET /api/server-stats`: Server telemetry endpoint for active clients and request history.
- `GET /api/client-heartbeat`: 4-second heartbeat signal from active client browsers.
- `POST /api/client-disconnect`: Window unload beacon to trigger clean slate resets.

### 3. Execution Commands
```powershell
# Step 1: Start Server on Laptop 1:
.\run.bat
# (or: python -u run_server.py)

# Step 2: Open Client Portal on Laptop 2 (Client):
# Open Chrome / Edge and navigate to: http://<SERVER_IP>:8000/client

# Step 3: Stream Live Server Logs in CMD (Laptop 1):
.\view_logs.bat

# Step 4: Re-compile 6-Page Technical Report PDF (Deliverable 2):
python build_competition_report_pdf.py
```

### 4. Competition Deliverables
- **Deliverable 1 (GitHub Repository)**: Clean, modular source code with `data_processing/`, `features/`, `models/`, `tuning/`, `hmm/`, `api/`, `ui/`, `viz/`, `artifacts/`, and `results/`.
- **Deliverable 2 (Technical Report PDF)**: [PATHSNAP_TECHNICAL_REPORT.pdf](PATHSNAP_TECHNICAL_REPORT.pdf) formatted strictly following official sections (2.2, 3, 4, 5, 6, 7, 8).


#### Crucial Downstream Modeling Rules:
1. **Group-Aware Splitting**: Always split train/validation sets using `GroupKFold(groups=train_df['trip_id'])`. Never split randomly by row.
2. **Feature Leakage Prevention**: Never use `label`, `label_valid`, `true_arc`, `d_true`, or `route_pos` as input features.
3. **Network Isolation**: Arc IDs are local to each trip. Never combine road networks across different trips.
4. **Temporal Gaps**: Trips `00000090` and `00000078` contain large time gaps (>50,000s). Treat large gaps ($\Delta t > 60\,\text{s}$) as breaks in sequential models / HMM transitions.

---

### 8. Reproducibility & Execution Commands

To execute the entire Person 1 pipeline from scratch:
```

---

## Person 2: Feature Engineering, Model Training & Bayesian Optimisation

### 1. Overview
Person 2 is responsible for:
- 19 modular features across geometric, kinematic, and network topology domains.
- Training 5 classification models across distinct families (Logistic Regression, KNN, Random Forest, HistGradientBoosting, MLP).
- Strict 5-fold `GroupKFold` cross-validation grouped by `trip_id` (zero spatial/temporal leakage).
- Observation-level Top-1 accuracy calculation grouping by `(trip_id, point_idx)`.
- Optuna Bayesian hyperparameter optimization (TPESampler) on HistGradientBoosting.
- Serialized production model artifacts (`artifacts/final_model.joblib`).

### 2. Experimental Results Summary

#### 5-Fold GroupKFold Cross-Validation (80 Training Trips, 2,013,074 Candidates)
| Model | Family | Mean CV Top-1 Acc | Mean CV Macro F1 | Mean Log Loss | Avg Train Time / Fold |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear | **79.32%** | 78.21% | 0.2193 | 1.86s |
| **Random Forest** | Bagging Ensemble | **78.41%** | **84.96%** | 0.1585 | 78.46s |
| **HistGradientBoosting** | Gradient Boosted Trees | **77.53%** | **83.65%** | 0.1733 | **9.39s (Fastest)** |
| **Multi-Layer Perceptron** | Neural Network | **76.85%** | **86.51%** | **0.1106** | 382.87s |
| **K-Nearest Neighbours** | Metric / Instance | **72.55%** | 80.30% | 0.1860 | 0.15s |

#### Held-Out Student-Test Evaluation (10 Trips, 220,026 Candidates)
- **Random Forest**: 81.92% Top-1 Accuracy | 85.52% Macro F1 | Loss: 0.1426
- **HistGradientBoosting**: 81.65% Top-1 Accuracy | 84.56% Macro F1 | Loss: 0.1535 (After Optuna: **82.15% Top-1**, **85.47% Macro F1**)
- **Logistic Regression**: 81.03% Top-1 Accuracy | 78.57% Macro F1 | Loss: 0.2045
- **Multi-Layer Perceptron**: 82.68% Top-1 Accuracy | 90.04% Macro F1 | Loss: 0.0792

---

## Person 3: Hidden Markov Model & Viterbi Sequence Decoding

### 1. Algorithm Overview
Implements the Newson & Krumm (2009) global sequence map-matching algorithm:
- **Emission Probabilities**: Person 2's trained ML classifier provides $P_{\text{ML}}(\text{arc}_i \mid \text{GPS}_t)$, fused with orthogonal Gaussian distance decay:
  $$\ln P_{\text{emiss}}(j) = \ln P_{\text{ML}}(j) - \frac{d_{\perp}^2}{2\sigma_z^2}$$
- **Transition Probabilities**: Road network topological shortest path routing via single-source Dijkstra:
  $$P_{\text{trans}}(i \to j) = \frac{1}{\beta} \exp\left( - \frac{|d_{\text{net}}(u, v) - d_{\text{gps}}|}{\beta} \right)$$
- **Viterbi Trellis**: Solves the dynamic programming trellis in log-space, elevating raw point-level accuracy (~82%) to **97.64% sequence route accuracy**!

---

## Person 4: Backend API, Distributed Multi-Node UI & Technical Report

### 1. Two-Laptop Distributed Architecture (As Mandated by Event Guidelines)
- **Host Server (Laptop 1)**: Runs the FastAPI backend service (`api/app.py`) on port `8000`. Houses candidate spatial extraction, machine learning inference, and the HMM Viterbi trellis engine. Serves the **Central Server Monitor** dashboard (`http://localhost:8000/server`), displaying connected client terminals, real-time inbound request feeds, and auto-resetting the mirrored map on client disconnect.
- **Client Terminals (Laptop 2 & Other Laptops)**: Accesses `http://<SERVER_IP>:8000/client` across campus Wi-Fi. Features a clean field submission portal with Student Folder benchmark trips (`student_test` and `train`), drag-and-drop `.track` file upload, and automatic 4-second client heartbeat telemetry.

### 2. REST API Endpoints
- `GET /`: Serves Server Monitor on localhost or Client Portal for external network clients.
- `GET /server`: Serves the dedicated Server Monitor Dashboard (Laptop 1).
- `GET /client`: Serves the dedicated Client Submission Portal (Laptop 2).
- `GET /api/health`: Health status and pipeline model verification.
- `GET /api/trips`: Lists trips prioritizing `student_test` and `train` from the Student Folder.
- `GET /api/trip/{partition}/{trip_id}`: Retrieves raw GPS points for a benchmark trip.
- `POST /api/match`: Full pipeline HMM Viterbi matching on GPS trajectory points.
- `POST /api/upload-and-match`: Uploads a custom `.track` or `.csv` file and snaps it to the network.
- `GET /api/server-stats`: Server telemetry endpoint for active clients and request history.
- `GET /api/client-heartbeat`: 4-second heartbeat signal from active client browsers.
- `POST /api/client-disconnect`: Window unload beacon to trigger clean slate resets.

### 3. Execution Commands
```powershell
# Step 1: Start Server on Laptop 1:
.\run.bat
# (or: python -u run_server.py)

# Step 2: Open Client Portal on Laptop 2 (Client):
# Open Chrome / Edge and navigate to: http://<SERVER_IP>:8000/client

# Step 3: Stream Live Server Logs in CMD (Laptop 1):
.\view_logs.bat

# Step 4: Re-compile 6-Page Technical Report PDF (Deliverable 2):
python build_competition_report_pdf.py
```

### 4. Competition Deliverables
- **Deliverable 1 (GitHub Repository)**: Clean, modular source code with `data_processing/`, `features/`, `models/`, `tuning/`, `hmm/`, `api/`, `ui/`, `viz/`, `artifacts/`, and `results/`.
- **Deliverable 2 (Technical Report PDF)**: [PATHSNAP_TECHNICAL_REPORT.pdf](PATHSNAP_TECHNICAL_REPORT.pdf) formatted strictly following official sections (2.2, 3, 4, 5, 6, 7, 8).
powershell
python run_person1_pipeline.py
```

---

## Person 2: Feature Engineering, Model Training & Bayesian Optimisation

### 1. Overview
Person 2 is responsible for:
- 19 modular features across geometric, kinematic, and network topology domains.
- Training 5 classification models across distinct families (Logistic Regression, KNN, Random Forest, HistGradientBoosting, MLP).
- Strict 5-fold `GroupKFold` cross-validation grouped by `trip_id` (zero spatial/temporal leakage).
- Observation-level Top-1 accuracy calculation grouping by `(trip_id, point_idx)`.
- Optuna Bayesian hyperparameter optimization (TPESampler) on HistGradientBoosting.
- Serialized production model artifacts (`artifacts/final_model.joblib`).

### 2. Experimental Results Summary

#### 5-Fold GroupKFold Cross-Validation (80 Training Trips, 2,013,074 Candidates)
| Model | Family | Mean CV Top-1 Acc | Mean CV Macro F1 | Mean Log Loss | Avg Train Time / Fold |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear | **79.32%** | 78.21% | 0.2193 | 1.86s |
| **Random Forest** | Bagging Ensemble | **78.41%** | **84.96%** | 0.1585 | 78.46s |
| **HistGradientBoosting** | Gradient Boosted Trees | **77.53%** | **83.65%** | 0.1733 | **9.39s (Fastest)** |
| **Multi-Layer Perceptron** | Neural Network | **76.85%** | **86.51%** | **0.1106** | 382.87s |
| **K-Nearest Neighbours** | Metric / Instance | **72.55%** | 80.30% | 0.1860 | 0.15s |

#### Held-Out Student-Test Evaluation (10 Trips, 220,026 Candidates)
- **Random Forest**: 81.92% Top-1 Accuracy | 85.52% Macro F1 | Loss: 0.1426
- **HistGradientBoosting**: 81.65% Top-1 Accuracy | 84.56% Macro F1 | Loss: 0.1535 (After Optuna: **82.15% Top-1**, **85.47% Macro F1**)
- **Logistic Regression**: 81.03% Top-1 Accuracy | 78.57% Macro F1 | Loss: 0.2045
- **Multi-Layer Perceptron**: 82.68% Top-1 Accuracy | 90.04% Macro F1 | Loss: 0.0792

---

## Person 3: Hidden Markov Model & Viterbi Sequence Decoding

### 1. Algorithm Overview
Implements the Newson & Krumm (2009) global sequence map-matching algorithm:
- **Emission Probabilities**: Person 2's trained ML classifier provides $P_{\text{ML}}(\text{arc}_i \mid \text{GPS}_t)$, fused with orthogonal Gaussian distance decay:
  $$\ln P_{\text{emiss}}(j) = \ln P_{\text{ML}}(j) - \frac{d_{\perp}^2}{2\sigma_z^2}$$
- **Transition Probabilities**: Road network topological shortest path routing via single-source Dijkstra:
  $$P_{\text{trans}}(i \to j) = \frac{1}{\beta} \exp\left( - \frac{|d_{\text{net}}(u, v) - d_{\text{gps}}|}{\beta} \right)$$
- **Viterbi Trellis**: Solves the dynamic programming trellis in log-space, elevating raw point-level accuracy (~82%) to **97.64% sequence route accuracy**!

---

## Person 4: Backend API, Distributed Multi-Node UI & Technical Report

### 1. Two-Laptop Distributed Architecture (As Mandated by Event Guidelines)
- **Host Server (Laptop 1)**: Runs the FastAPI backend service (`api/app.py`) on port `8000`. Houses candidate spatial extraction, machine learning inference, and the HMM Viterbi trellis engine. Serves the **Central Server Monitor** dashboard (`http://localhost:8000/server`), displaying connected client terminals, real-time inbound request feeds, and auto-resetting the mirrored map on client disconnect.
- **Client Terminals (Laptop 2 & Other Laptops)**: Accesses `http://<SERVER_IP>:8000/client` across campus Wi-Fi. Features a clean field submission portal with Student Folder benchmark trips (`student_test` and `train`), drag-and-drop `.track` file upload, and automatic 4-second client heartbeat telemetry.

### 2. REST API Endpoints
- `GET /`: Serves Server Monitor on localhost or Client Portal for external network clients.
- `GET /server`: Serves the dedicated Server Monitor Dashboard (Laptop 1).
- `GET /client`: Serves the dedicated Client Submission Portal (Laptop 2).
- `GET /api/health`: Health status and pipeline model verification.
- `GET /api/trips`: Lists trips prioritizing `student_test` and `train` from the Student Folder.
- `GET /api/trip/{partition}/{trip_id}`: Retrieves raw GPS points for a benchmark trip.
- `POST /api/match`: Full pipeline HMM Viterbi matching on GPS trajectory points.
- `POST /api/upload-and-match`: Uploads a custom `.track` or `.csv` file and snaps it to the network.
- `GET /api/server-stats`: Server telemetry endpoint for active clients and request history.
- `GET /api/client-heartbeat`: 4-second heartbeat signal from active client browsers.
- `POST /api/client-disconnect`: Window unload beacon to trigger clean slate resets.

### 3. Execution Commands
```powershell
# Step 1: Start Server on Laptop 1:
.\run.bat
# (or: python -u run_server.py)

# Step 2: Open Client Portal on Laptop 2 (Client):
# Open Chrome / Edge and navigate to: http://<SERVER_IP>:8000/client

# Step 3: Stream Live Server Logs in CMD (Laptop 1):
.\view_logs.bat

# Step 4: Re-compile 6-Page Technical Report PDF (Deliverable 2):
python build_competition_report_pdf.py
```

### 4. Competition Deliverables
- **Deliverable 1 (GitHub Repository)**: Clean, modular source code with `data_processing/`, `features/`, `models/`, `tuning/`, `hmm/`, `api/`, `ui/`, `viz/`, `artifacts/`, and `results/`.
- **Deliverable 2 (Technical Report PDF)**: [PATHSNAP_TECHNICAL_REPORT.pdf](PATHSNAP_TECHNICAL_REPORT.pdf) formatted strictly following official sections (2.2, 3, 4, 5, 6, 7, 8).


Or individual components:
```

---

## Person 2: Feature Engineering, Model Training & Bayesian Optimisation

### 1. Overview
Person 2 is responsible for:
- 19 modular features across geometric, kinematic, and network topology domains.
- Training 5 classification models across distinct families (Logistic Regression, KNN, Random Forest, HistGradientBoosting, MLP).
- Strict 5-fold `GroupKFold` cross-validation grouped by `trip_id` (zero spatial/temporal leakage).
- Observation-level Top-1 accuracy calculation grouping by `(trip_id, point_idx)`.
- Optuna Bayesian hyperparameter optimization (TPESampler) on HistGradientBoosting.
- Serialized production model artifacts (`artifacts/final_model.joblib`).

### 2. Experimental Results Summary

#### 5-Fold GroupKFold Cross-Validation (80 Training Trips, 2,013,074 Candidates)
| Model | Family | Mean CV Top-1 Acc | Mean CV Macro F1 | Mean Log Loss | Avg Train Time / Fold |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear | **79.32%** | 78.21% | 0.2193 | 1.86s |
| **Random Forest** | Bagging Ensemble | **78.41%** | **84.96%** | 0.1585 | 78.46s |
| **HistGradientBoosting** | Gradient Boosted Trees | **77.53%** | **83.65%** | 0.1733 | **9.39s (Fastest)** |
| **Multi-Layer Perceptron** | Neural Network | **76.85%** | **86.51%** | **0.1106** | 382.87s |
| **K-Nearest Neighbours** | Metric / Instance | **72.55%** | 80.30% | 0.1860 | 0.15s |

#### Held-Out Student-Test Evaluation (10 Trips, 220,026 Candidates)
- **Random Forest**: 81.92% Top-1 Accuracy | 85.52% Macro F1 | Loss: 0.1426
- **HistGradientBoosting**: 81.65% Top-1 Accuracy | 84.56% Macro F1 | Loss: 0.1535 (After Optuna: **82.15% Top-1**, **85.47% Macro F1**)
- **Logistic Regression**: 81.03% Top-1 Accuracy | 78.57% Macro F1 | Loss: 0.2045
- **Multi-Layer Perceptron**: 82.68% Top-1 Accuracy | 90.04% Macro F1 | Loss: 0.0792

---

## Person 3: Hidden Markov Model & Viterbi Sequence Decoding

### 1. Algorithm Overview
Implements the Newson & Krumm (2009) global sequence map-matching algorithm:
- **Emission Probabilities**: Person 2's trained ML classifier provides $P_{\text{ML}}(\text{arc}_i \mid \text{GPS}_t)$, fused with orthogonal Gaussian distance decay:
  $$\ln P_{\text{emiss}}(j) = \ln P_{\text{ML}}(j) - \frac{d_{\perp}^2}{2\sigma_z^2}$$
- **Transition Probabilities**: Road network topological shortest path routing via single-source Dijkstra:
  $$P_{\text{trans}}(i \to j) = \frac{1}{\beta} \exp\left( - \frac{|d_{\text{net}}(u, v) - d_{\text{gps}}|}{\beta} \right)$$
- **Viterbi Trellis**: Solves the dynamic programming trellis in log-space, elevating raw point-level accuracy (~82%) to **97.64% sequence route accuracy**!

---

## Person 4: Backend API, Distributed Multi-Node UI & Technical Report

### 1. Two-Laptop Distributed Architecture (As Mandated by Event Guidelines)
- **Host Server (Laptop 1)**: Runs the FastAPI backend service (`api/app.py`) on port `8000`. Houses candidate spatial extraction, machine learning inference, and the HMM Viterbi trellis engine. Serves the **Central Server Monitor** dashboard (`http://localhost:8000/server`), displaying connected client terminals, real-time inbound request feeds, and auto-resetting the mirrored map on client disconnect.
- **Client Terminals (Laptop 2 & Other Laptops)**: Accesses `http://<SERVER_IP>:8000/client` across campus Wi-Fi. Features a clean field submission portal with Student Folder benchmark trips (`student_test` and `train`), drag-and-drop `.track` file upload, and automatic 4-second client heartbeat telemetry.

### 2. REST API Endpoints
- `GET /`: Serves Server Monitor on localhost or Client Portal for external network clients.
- `GET /server`: Serves the dedicated Server Monitor Dashboard (Laptop 1).
- `GET /client`: Serves the dedicated Client Submission Portal (Laptop 2).
- `GET /api/health`: Health status and pipeline model verification.
- `GET /api/trips`: Lists trips prioritizing `student_test` and `train` from the Student Folder.
- `GET /api/trip/{partition}/{trip_id}`: Retrieves raw GPS points for a benchmark trip.
- `POST /api/match`: Full pipeline HMM Viterbi matching on GPS trajectory points.
- `POST /api/upload-and-match`: Uploads a custom `.track` or `.csv` file and snaps it to the network.
- `GET /api/server-stats`: Server telemetry endpoint for active clients and request history.
- `GET /api/client-heartbeat`: 4-second heartbeat signal from active client browsers.
- `POST /api/client-disconnect`: Window unload beacon to trigger clean slate resets.

### 3. Execution Commands
```powershell
# Step 1: Start Server on Laptop 1:
.\run.bat
# (or: python -u run_server.py)

# Step 2: Open Client Portal on Laptop 2 (Client):
# Open Chrome / Edge and navigate to: http://<SERVER_IP>:8000/client

# Step 3: Stream Live Server Logs in CMD (Laptop 1):
.\view_logs.bat

# Step 4: Re-compile 6-Page Technical Report PDF (Deliverable 2):
python build_competition_report_pdf.py
```

### 4. Competition Deliverables
- **Deliverable 1 (GitHub Repository)**: Clean, modular source code with `data_processing/`, `features/`, `models/`, `tuning/`, `hmm/`, `api/`, `ui/`, `viz/`, `artifacts/`, and `results/`.
- **Deliverable 2 (Technical Report PDF)**: [PATHSNAP_TECHNICAL_REPORT.pdf](PATHSNAP_TECHNICAL_REPORT.pdf) formatted strictly following official sections (2.2, 3, 4, 5, 6, 7, 8).
powershell
# 1. Dataset inventory audit
python data_processing/inventory.py data

# 2. GPS & road-network quality audit
python data_processing/quality_checks.py data

# 3. Candidate generation & ground-truth alignment
python data_processing/candidates.py data

# 4. Generate EDA visualizations
python viz/eda_plots.py

# 5. Generate interactive Folium map for sample trip
python viz/folium_trip.py data train 00000001

# 6. Run unit and regression tests
python data_processing/test_person1.py
python data_processing/test_candidate_gen.py data
```

---

## Person 2: Feature Engineering, Model Training & Bayesian Optimisation

### 1. Overview
Person 2 is responsible for:
- 19 modular features across geometric, kinematic, and network topology domains.
- Training 5 classification models across distinct families (Logistic Regression, KNN, Random Forest, HistGradientBoosting, MLP).
- Strict 5-fold `GroupKFold` cross-validation grouped by `trip_id` (zero spatial/temporal leakage).
- Observation-level Top-1 accuracy calculation grouping by `(trip_id, point_idx)`.
- Optuna Bayesian hyperparameter optimization (TPESampler) on HistGradientBoosting.
- Serialized production model artifacts (`artifacts/final_model.joblib`).

### 2. Experimental Results Summary

#### 5-Fold GroupKFold Cross-Validation (80 Training Trips, 2,013,074 Candidates)
| Model | Family | Mean CV Top-1 Acc | Mean CV Macro F1 | Mean Log Loss | Avg Train Time / Fold |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear | **79.32%** | 78.21% | 0.2193 | 1.86s |
| **Random Forest** | Bagging Ensemble | **78.41%** | **84.96%** | 0.1585 | 78.46s |
| **HistGradientBoosting** | Gradient Boosted Trees | **77.53%** | **83.65%** | 0.1733 | **9.39s (Fastest)** |
| **Multi-Layer Perceptron** | Neural Network | **76.85%** | **86.51%** | **0.1106** | 382.87s |
| **K-Nearest Neighbours** | Metric / Instance | **72.55%** | 80.30% | 0.1860 | 0.15s |

#### Held-Out Student-Test Evaluation (10 Trips, 220,026 Candidates)
- **Random Forest**: 81.92% Top-1 Accuracy | 85.52% Macro F1 | Loss: 0.1426
- **HistGradientBoosting**: 81.65% Top-1 Accuracy | 84.56% Macro F1 | Loss: 0.1535 (After Optuna: **82.15% Top-1**, **85.47% Macro F1**)
- **Logistic Regression**: 81.03% Top-1 Accuracy | 78.57% Macro F1 | Loss: 0.2045
- **Multi-Layer Perceptron**: 82.68% Top-1 Accuracy | 90.04% Macro F1 | Loss: 0.0792

---

## Person 3: Hidden Markov Model & Viterbi Sequence Decoding

### 1. Algorithm Overview
Implements the Newson & Krumm (2009) global sequence map-matching algorithm:
- **Emission Probabilities**: Person 2's trained ML classifier provides $P_{\text{ML}}(\text{arc}_i \mid \text{GPS}_t)$, fused with orthogonal Gaussian distance decay:
  $$\ln P_{\text{emiss}}(j) = \ln P_{\text{ML}}(j) - \frac{d_{\perp}^2}{2\sigma_z^2}$$
- **Transition Probabilities**: Road network topological shortest path routing via single-source Dijkstra:
  $$P_{\text{trans}}(i \to j) = \frac{1}{\beta} \exp\left( - \frac{|d_{\text{net}}(u, v) - d_{\text{gps}}|}{\beta} \right)$$
- **Viterbi Trellis**: Solves the dynamic programming trellis in log-space, elevating raw point-level accuracy (~82%) to **97.64% sequence route accuracy**!

---

## Person 4: Backend API, Distributed Multi-Node UI & Technical Report

### 1. Two-Laptop Distributed Architecture (As Mandated by Event Guidelines)
- **Host Server (Laptop 1)**: Runs the FastAPI backend service (`api/app.py`) on port `8000`. Houses candidate spatial extraction, machine learning inference, and the HMM Viterbi trellis engine. Serves the **Central Server Monitor** dashboard (`http://localhost:8000/server`), displaying connected client terminals, real-time inbound request feeds, and auto-resetting the mirrored map on client disconnect.
- **Client Terminals (Laptop 2 & Other Laptops)**: Accesses `http://<SERVER_IP>:8000/client` across campus Wi-Fi. Features a clean field submission portal with Student Folder benchmark trips (`student_test` and `train`), drag-and-drop `.track` file upload, and automatic 4-second client heartbeat telemetry.

### 2. REST API Endpoints
- `GET /`: Serves Server Monitor on localhost or Client Portal for external network clients.
- `GET /server`: Serves the dedicated Server Monitor Dashboard (Laptop 1).
- `GET /client`: Serves the dedicated Client Submission Portal (Laptop 2).
- `GET /api/health`: Health status and pipeline model verification.
- `GET /api/trips`: Lists trips prioritizing `student_test` and `train` from the Student Folder.
- `GET /api/trip/{partition}/{trip_id}`: Retrieves raw GPS points for a benchmark trip.
- `POST /api/match`: Full pipeline HMM Viterbi matching on GPS trajectory points.
- `POST /api/upload-and-match`: Uploads a custom `.track` or `.csv` file and snaps it to the network.
- `GET /api/server-stats`: Server telemetry endpoint for active clients and request history.
- `GET /api/client-heartbeat`: 4-second heartbeat signal from active client browsers.
- `POST /api/client-disconnect`: Window unload beacon to trigger clean slate resets.

### 3. Execution Commands
```powershell
# Step 1: Start Server on Laptop 1:
.\run.bat
# (or: python -u run_server.py)

# Step 2: Open Client Portal on Laptop 2 (Client):
# Open Chrome / Edge and navigate to: http://<SERVER_IP>:8000/client

# Step 3: Stream Live Server Logs in CMD (Laptop 1):
.\view_logs.bat

# Step 4: Re-compile 6-Page Technical Report PDF (Deliverable 2):
python build_competition_report_pdf.py
```

### 4. Competition Deliverables
- **Deliverable 1 (GitHub Repository)**: Clean, modular source code with `data_processing/`, `features/`, `models/`, `tuning/`, `hmm/`, `api/`, `ui/`, `viz/`, `artifacts/`, and `results/`.
- **Deliverable 2 (Technical Report PDF)**: [PATHSNAP_TECHNICAL_REPORT.pdf](PATHSNAP_TECHNICAL_REPORT.pdf) formatted strictly following official sections (2.2, 3, 4, 5, 6, 7, 8).

