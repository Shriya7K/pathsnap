"""High-level Map Matching Pipeline Integrator (Persons 1, 2, and 3).

Executes the full pipeline:
  Step 1 (Person 1): Candidate Generation & UTM Projections (STRtree)
  Step 2 (Person 2): Feature Engineering & ML Probability Classification (HistGradientBoosting)
  Step 3 (Person 3): HMM Transition Modeling & Viterbi Trellis Route Decoding
"""
from pathlib import Path
import time
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer

from data_processing.candidate_gen import generate, TripData
from features.config import FEATURE_COLUMNS, CANDIDATE_K, CANDIDATE_R
from hmm.viterbi import HMMMapMatcher, MatchResult


class FullPipelineDecoder:
    """End-to-end Map Matching Decoder integrating Person 1, Person 2, and Person 3."""

    def __init__(
        self,
        model_path: Path = Path("artifacts/final_model.joblib"),
        beta: float = 10.0,
        sigma_z: float = 15.0,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model artifact not found at {self.model_path}")

        print(f"Loading trained classifier model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        self.matcher = HMMMapMatcher(beta=beta, sigma_z=sigma_z)

    def match_trip(
        self,
        trip_dir: Path,
        k: int = CANDIDATE_K,
        r: float = CANDIDATE_R,
    ) -> Dict[str, Any]:
        """Match a trip directory containing .track, .nodes, and .arcs."""
        t0 = time.time()
        trip_dir = Path(trip_dir)

        # Step 1: Candidate Generation (Person 1)
        trip_data: TripData = generate(trip_dir, k=k, r=r)
        cands_df = trip_data.cands.copy()

        if len(cands_df) == 0:
            return {
                "trip_id": trip_data.trip_id,
                "error": "No road network candidates found within search radius",
                "matched_arcs": [],
                "matched_coordinates": [],
            }

        # Step 2: Feature Engineering & Scoring (Person 2)
        # Compute the 19 features required by Person 2's model
        nodes_xy = np.column_stack([trip_data.nx, trip_data.ny])
        cands_with_features = self._compute_features(trip_data, cands_df, nodes_xy)

        # Model inference: predict candidate emission probabilities
        X = cands_with_features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        probs = self.model.predict_proba(X)[:, 1]
        cands_with_features["predicted_probability"] = probs

        # Step 3: HMM Viterbi Global Decoding (Person 3)
        match_result = self.matcher.match(
            trip_id=trip_data.trip_id,
            gps_xy=trip_data.xy,
            gps_ts=trip_data.ts,
            nodes_xy=nodes_xy,
            a_from=trip_data.a_from,
            a_to=trip_data.a_to,
            candidates_df=cands_with_features,
            epsg=trip_data.epsg,
        )

        total_time = time.time() - t0

        # Optional: compute accuracy against ground-truth .route if present
        route_file = trip_dir / f"{trip_data.trip_id}.route"
        ground_truth_accuracy = None
        if route_file.exists():
            ref_routes = pd.read_csv(route_file, header=None)[0].to_numpy()
            project_root = Path(__file__).resolve().parent.parent
            cand_points_file = None
            for p_name in ["student_test", "train", "evaluator_test"]:
                f_test = project_root / "results" / "candidates" / f"{p_name}_{trip_data.trip_id}_points.csv.gz"
                if f_test.exists():
                    cand_points_file = f_test
                    break
            if cand_points_file and cand_points_file.exists():
                pt_df = pd.read_csv(cand_points_file)
                if "true_arc" in pt_df.columns:
                    true_arcs = pt_df["true_arc"].to_numpy()
                    correct = (np.array(match_result.matched_arc_ids) == true_arcs).mean()
                    ground_truth_accuracy = round(float(correct * 100.0), 2)
            if ground_truth_accuracy is None and len(match_result.unique_route_arcs) > 0 and len(ref_routes) > 0:
                matched_set = set(match_result.unique_route_arcs)
                ref_set = set(ref_routes)
                iou = len(matched_set & ref_set) / max(len(matched_set | ref_set), 1)
                ground_truth_accuracy = round(float(iou * 100.0), 2)

        return {
            "trip_id": trip_data.trip_id,
            "epsg": trip_data.epsg,
            "point_count": len(trip_data.xy),
            "candidate_count": len(cands_with_features),
            "total_distance_m": match_result.total_distance_m,
            "execution_time_sec": round(total_time, 3),
            "ground_truth_accuracy": ground_truth_accuracy,
            "matched_arcs": match_result.matched_arc_ids,
            "unique_route_arcs": match_result.unique_route_arcs,
            "matched_coordinates": match_result.matched_coords_wgs84,
            "raw_coordinates": match_result.raw_coords_wgs84,
            "confidences": match_result.confidences,
            "mean_confidence": round(float(np.mean(match_result.confidences)), 4) if match_result.confidences else 0.0,
        }

    def match_raw_coordinates(
        self,
        points: List[Dict[str, float]],  # [{"lat": ..., "lon": ..., "ts": ...}]
        reference_trip_id: str = "00000001",
        partition: str = "student_test",
        data_dir: Path = Path("data"),
    ) -> Dict[str, Any]:
        """Match arbitrary user-submitted GPS coordinates against a trip's road network."""
        trip_dir = data_dir / partition / reference_trip_id
        if not trip_dir.exists():
            trip_dir = data_dir / "train" / reference_trip_id
        if not trip_dir.exists():
            raise FileNotFoundError(f"Road network trip {reference_trip_id} not found in {data_dir}")

        # Write temporary track file or construct in-memory TripData
        temp_track_df = pd.DataFrame([
            {"lon": p.get("lon", p.get("longitude")), "lat": p.get("lat", p.get("latitude")), "ts": p.get("ts", p.get("timestamp", idx))}
            for idx, p in enumerate(points)
        ])

        # Load road network nodes and arcs
        nodes = pd.read_csv(trip_dir / f"{reference_trip_id}.nodes", sep="\t", header=None)
        arcs = pd.read_csv(trip_dir / f"{reference_trip_id}.arcs", sep="\t", header=None)
        a_from, a_to = arcs[0].to_numpy(), arcs[1].to_numpy()

        # Project using trip UTM zone
        mean_lon, mean_lat = float(temp_track_df["lon"].mean()), float(temp_track_df["lat"].mean())
        zone = int((mean_lon + 180) // 6) + 1
        epsg = (32600 if mean_lat >= 0 else 32700) + zone
        tr = Transformer.from_crs(4326, epsg, always_xy=True)

        nx, ny = tr.transform(nodes[0].to_numpy(), nodes[1].to_numpy())
        nx, ny = np.asarray(nx), np.asarray(ny)
        px, py = tr.transform(temp_track_df["lon"].to_numpy(), temp_track_df["lat"].to_numpy())
        px, py = np.asarray(px), np.asarray(py)
        xy = np.column_stack([px, py])

        # Candidate generation
        import shapely
        from shapely import STRtree
        r = CANDIDATE_R
        k = CANDIDATE_K

        xmin, ymin = xy.min(0) - r
        xmax, ymax = xy.max(0) + r
        x1, y1, x2, y2 = nx[a_from], ny[a_from], nx[a_to], ny[a_to]
        keep = np.where(
            (np.maximum(x1, x2) >= xmin) & (np.minimum(x1, x2) <= xmax) &
            (np.maximum(y1, y2) >= ymin) & (np.minimum(y1, y2) <= ymax)
        )[0]

        if len(keep) == 0:
            return {"error": "GPS coordinates are outside the road network coverage area"}

        coords = np.stack([
            np.column_stack([x1[keep], y1[keep]]),
            np.column_stack([x2[keep], y2[keep]])
        ], axis=1)
        lines = shapely.linestrings(coords)
        tree = STRtree(lines)
        pts = shapely.points(xy)
        pi, li = tree.query(pts, predicate="dwithin", distance=r)
        dist = shapely.distance(pts[pi], lines[li])

        c = pd.DataFrame({"point_idx": pi, "arc_id": keep[li], "dist": dist})
        c = c.sort_values(["point_idx", "dist", "arc_id"], kind="stable").reset_index(drop=True)
        c["rank"] = c.groupby("point_idx").cumcount()
        c = c[c["rank"] < k].reset_index(drop=True)
        c["trip_id"] = reference_trip_id
        c["perpendicular_distance"] = c["dist"].round(4)
        c["distance_rank"] = c["rank"]
        c["x"] = px[c["point_idx"].to_numpy()]
        c["y"] = py[c["point_idx"].to_numpy()]
        c["ts"] = temp_track_df["ts"].iloc[c["point_idx"].to_numpy()].to_numpy()

        trip_data = TripData(
            trip_id=reference_trip_id, trip=reference_trip_id, epsg=epsg,
            xy=xy, ts=temp_track_df["ts"].to_numpy(),
            nx=nx, ny=ny, a_from=a_from, a_to=a_to, cands=c
        )

        nodes_xy = np.column_stack([nx, ny])
        cands_with_features = self._compute_features(trip_data, c, nodes_xy)
        X = cands_with_features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        probs = self.model.predict_proba(X)[:, 1]
        cands_with_features["predicted_probability"] = probs

        match_result = self.matcher.match(
            trip_id=reference_trip_id,
            gps_xy=xy,
            gps_ts=temp_track_df["ts"].to_numpy(),
            nodes_xy=nodes_xy,
            a_from=a_from,
            a_to=a_to,
            candidates_df=cands_with_features,
            epsg=epsg,
        )

        # Optional: compute accuracy against ground-truth .route if present
        route_file = trip_dir / f"{reference_trip_id}.route"
        ground_truth_accuracy = None
        if route_file.exists():
            ref_routes = pd.read_csv(route_file, header=None)[0].to_numpy()
            project_root = Path(__file__).resolve().parent.parent
            cand_points_file = None
            for p_name in ["student_test", "train", "evaluator_test"]:
                f_test = project_root / "results" / "candidates" / f"{p_name}_{reference_trip_id}_points.csv.gz"
                if f_test.exists():
                    cand_points_file = f_test
                    break
            if cand_points_file and cand_points_file.exists():
                pt_df = pd.read_csv(cand_points_file)
                if "true_arc" in pt_df.columns and len(pt_df) == len(match_result.matched_arc_ids):
                    true_arcs = pt_df["true_arc"].to_numpy()
                    correct = (np.array(match_result.matched_arc_ids) == true_arcs).mean()
                    ground_truth_accuracy = round(float(correct * 100.0), 2)
            if ground_truth_accuracy is None and len(match_result.unique_route_arcs) > 0 and len(ref_routes) > 0:
                matched_set = set(match_result.unique_route_arcs)
                ref_set = set(ref_routes)
                iou = len(matched_set & ref_set) / max(len(matched_set | ref_set), 1)
                ground_truth_accuracy = round(float(iou * 100.0), 2)

        return {
            "trip_id": reference_trip_id,
            "point_count": len(points),
            "ground_truth_accuracy": ground_truth_accuracy,
            "matched_arcs": match_result.matched_arc_ids,
            "unique_route_arcs": match_result.unique_route_arcs,
            "matched_coordinates": match_result.matched_coords_wgs84,
            "raw_coordinates": match_result.raw_coords_wgs84,
            "confidences": match_result.confidences,
            "mean_confidence": round(float(np.mean(match_result.confidences)), 4) if match_result.confidences else 0.0,
            "total_distance_m": match_result.total_distance_m,
        }

    def _compute_features(self, trip_data: TripData, c: pd.DataFrame, nodes_xy: np.ndarray) -> pd.DataFrame:
        """Vectorized feature calculation for arbitrary candidates."""
        from features.engineer import angle_diff

        nx, ny = trip_data.nx, trip_data.ny
        a_from, a_to = trip_data.a_from, trip_data.a_to
        n_nodes = len(nx)

        # Node degrees
        degree_counts = np.bincount(np.concatenate([a_from, a_to]), minlength=n_nodes)
        dx_arc = nx[a_to] - nx[a_from]
        dy_arc = ny[a_to] - ny[a_from]
        arc_lengths = np.hypot(dx_arc, dy_arc)
        arc_bearings = np.arctan2(dx_arc, dy_arc)

        # GPS point kinematics
        px = trip_data.xy[:, 0]
        py = trip_data.xy[:, 1]
        ts = trip_data.ts
        n_pts = len(px)

        if n_pts > 1:
            dx_pts = np.diff(px)
            dy_pts = np.diff(py)
            dt_pts = np.diff(ts)
            dist_consec = np.hypot(dx_pts, dy_pts)
            safe_dt = np.where(dt_pts <= 0, 1.0, dt_pts)
            v_consec = dist_consec / safe_dt
            headings_consec = np.arctan2(dx_pts, dy_pts)

            gps_speed = np.concatenate([[v_consec[0]], v_consec])
            time_diff = np.concatenate([[safe_dt[0]], safe_dt])
            dist_travelled = np.concatenate([[0.0], dist_consec])
            gps_heading = np.concatenate([[headings_consec[0]], headings_consec])
            speed_forward = np.concatenate([v_consec, [v_consec[-1]]])
        else:
            gps_speed = np.array([0.0])
            time_diff = np.array([1.0])
            dist_travelled = np.array([0.0])
            gps_heading = np.array([0.0])
            speed_forward = np.array([0.0])

        acceleration = (speed_forward - gps_speed) / np.maximum(time_diff, 0.1)

        # Map to candidates
        pt_indices = c["point_idx"].to_numpy()
        cand_arcs = c["arc_id"].to_numpy()
        dist_col = "perpendicular_distance" if "perpendicular_distance" in c.columns else "dist"

        min_dist_per_point = c.groupby("point_idx")[dist_col].transform("min").to_numpy()
        a_f = a_from[cand_arcs]
        a_t = a_to[cand_arcs]
        x_from, y_from = nx[a_f], ny[a_f]
        x_to, y_to = nx[a_t], ny[a_t]

        cand_dx = x_to - x_from
        cand_dy = y_to - y_from
        c_arc_len = arc_lengths[cand_arcs]
        c_arc_bear = arc_bearings[cand_arcs]

        p_x = px[pt_indices]
        p_y = py[pt_indices]
        cand_L2 = cand_dx ** 2 + cand_dy ** 2
        safe_L2 = np.where(cand_L2 <= 0, 1.0, cand_L2)
        t_proj = np.clip(((p_x - x_from) * cand_dx + (p_y - y_from) * cand_dy) / safe_L2, 0.0, 1.0)
        dist_along_arc = t_proj * c_arc_len

        c_gps_head = gps_heading[pt_indices]
        h_diff = angle_diff(c_gps_head, c_arc_bear)
        cos_h_diff = np.cos(h_diff)

        # Topological connectivity to previous candidate
        prev_connected = np.zeros(len(c), dtype=np.int8)
        top_cands = c[c["distance_rank"] < 3]
        pt_to_top_arcs = top_cands.groupby("point_idx")["arc_id"].apply(set).to_dict()

        for idx, row in enumerate(c.itertuples()):
            p_i = row.point_idx
            if p_i > 0 and (p_i - 1) in pt_to_top_arcs:
                prev_arcs = pt_to_top_arcs[p_i - 1]
                curr_start = a_f[idx]
                if any(a_to[pa] == curr_start or pa == row.arc_id for pa in prev_arcs):
                    prev_connected[idx] = 1

        dist_ratio = c[dist_col].to_numpy() / np.maximum(min_dist_per_point, 0.1)
        speed_heading_align = gps_speed[pt_indices] * cos_h_diff

        out = c.copy()
        out["perpendicular_distance"] = c[dist_col].to_numpy(dtype=np.float32)
        out["distance_rank"] = c["distance_rank"].to_numpy(dtype=np.int16)
        out["dist_along_arc"] = dist_along_arc.astype(np.float32)
        out["arc_length"] = c_arc_len.astype(np.float32)
        out["arc_bearing"] = c_arc_bear.astype(np.float32)
        out["heading_bearing_diff"] = h_diff.astype(np.float32)
        out["cos_heading_diff"] = cos_h_diff.astype(np.float32)
        out["gps_speed"] = gps_speed[pt_indices].astype(np.float32)
        out["time_diff"] = time_diff[pt_indices].astype(np.float32)
        out["distance_travelled"] = dist_travelled[pt_indices].astype(np.float32)
        out["gps_heading"] = c_gps_head.astype(np.float32)
        out["speed_forward"] = speed_forward[pt_indices].astype(np.float32)
        out["acceleration"] = acceleration[pt_indices].astype(np.float32)
        out["start_node_degree"] = degree_counts[a_f].astype(np.int16)
        out["end_node_degree"] = degree_counts[a_t].astype(np.int16)
        out["is_junction"] = ((degree_counts[a_f] >= 3) | (degree_counts[a_t] >= 3)).astype(np.int8)
        out["prev_candidate_connected"] = prev_connected
        out["dist_ratio"] = dist_ratio.astype(np.float32)
        out["speed_heading_alignment"] = speed_heading_align.astype(np.float32)

        return out
