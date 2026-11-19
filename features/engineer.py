"""Feature engineering pipeline for PathSnap GPS-to-road map matching."""
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer

from features.config import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    VALID_COLUMN,
    CANDIDATE_K,
    CANDIDATE_R,
)


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def angle_diff(a1, a2):
    """Smallest absolute angular difference between two angles in radians, in [0, pi]."""
    diff = np.abs(a1 - a2) % (2.0 * np.pi)
    return np.where(diff > np.pi, 2.0 * np.pi - diff, diff)


def extract_trip_features(
    partition: str,
    trip_id: str,
    data_dir: Path = Path("data"),
    cand_dir: Path = Path("results/candidates"),
    k: int = CANDIDATE_K,
    r: float = CANDIDATE_R,
    valid_only: bool = False,
) -> pd.DataFrame:
    """Extract all geometric, kinematic, and topological features for a single trip.

    Parameters:
    -----------
    partition : 'train', 'student_test', etc.
    trip_id : e.g. '00000001'
    data_dir : path to raw dataset
    cand_dir : path to candidate dataset
    k : max candidate rank cap
    r : max search radius in meters
    valid_only : filter to label_valid == 1

    Returns:
    --------
    pd.DataFrame containing IDENTIFIER_COLUMNS + FEATURE_COLUMNS (+ label / label_valid if present).
    """
    trip_path = data_dir / partition / trip_id
    cand_file = cand_dir / f"{partition}_{trip_id}_cands.csv.gz"
    points_file = cand_dir / f"{partition}_{trip_id}_points.csv.gz"

    if not cand_file.exists():
        raise FileNotFoundError(f"Candidate file missing: {cand_file}")

    c = pd.read_csv(cand_file, dtype={"trip_id": str, "trip": str})
    p = pd.read_csv(points_file, dtype={"trip_id": str})

    # Filter to requested candidate radius and rank cap
    dist_col = "perpendicular_distance" if "perpendicular_distance" in c.columns else "dist"
    rank_col = "distance_rank" if "distance_rank" in c.columns else "rank"
    c = c[(c[rank_col] < k) & (c[dist_col] <= r)].copy()

    if valid_only and "label_valid" in c.columns:
        c = c[c["label_valid"] == 1].copy()

    if len(c) == 0:
        return pd.DataFrame(columns=IDENTIFIER_COLUMNS + FEATURE_COLUMNS)

    # 1. Load Road Network
    nodes = pd.read_csv(trip_path / f"{trip_id}.nodes", sep="\t", header=None)
    arcs = pd.read_csv(trip_path / f"{trip_id}.arcs", sep="\t", header=None)
    a_from = arcs[0].to_numpy()
    a_to = arcs[1].to_numpy()
    n_nodes = len(nodes)

    # Project Network to local metric UTM
    epsg = utm_epsg(float(nodes[0].mean()), float(nodes[1].mean()))
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    nx, ny = tr.transform(nodes[0].to_numpy(), nodes[1].to_numpy())
    nx, ny = np.asarray(nx), np.asarray(ny)

    # Compute node degrees
    degree_counts = np.bincount(np.concatenate([a_from, a_to]), minlength=n_nodes)

    # Compute arc geometry
    dx_arc = nx[a_to] - nx[a_from]
    dy_arc = ny[a_to] - ny[a_from]
    arc_lengths = np.hypot(dx_arc, dy_arc)
    # Bearing: compass orientation angle in radians [-pi, pi] (0 = North, pi/2 = East)
    arc_bearings = np.arctan2(dx_arc, dy_arc)

    # 2. Compute GPS Observation Kinematics
    px = p["x"].to_numpy()
    py = p["y"].to_numpy()
    ts = pd.to_numeric(p["ts"]).to_numpy()
    n_pts = len(px)

    dx_pts = np.diff(px)
    dy_pts = np.diff(py)
    dt_pts = np.diff(ts)
    dist_consec = np.hypot(dx_pts, dy_pts)

    # Safe dt avoiding division by 0
    safe_dt = np.where(dt_pts <= 0, 1.0, dt_pts)
    v_consec = dist_consec / safe_dt
    headings_consec = np.arctan2(dx_pts, dy_pts)

    # Build per-observation kinematic arrays
    # Observation 0 (first point) takes forward kinematics
    if n_pts > 1:
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

    # 3. Associate Candidate Arcs with Extracted Features
    pt_indices = c["point_idx"].to_numpy()
    cand_arcs = c["arc_id"].to_numpy()

    # Pre-calculate closest candidate distance per point
    min_dist_per_point = c.groupby("point_idx")[dist_col].transform("min").to_numpy()

    # Arc endpoints for candidates
    a_f = a_from[cand_arcs]
    a_t = a_to[cand_arcs]
    x_from, y_from = nx[a_f], ny[a_f]
    x_to, y_to = nx[a_t], ny[a_t]

    cand_dx = x_to - x_from
    cand_dy = y_to - y_from
    c_arc_len = arc_lengths[cand_arcs]
    c_arc_bear = arc_bearings[cand_arcs]

    # Projection distance along arc: t in [0, 1]
    p_x = px[pt_indices]
    p_y = py[pt_indices]
    cand_L2 = cand_dx ** 2 + cand_dy ** 2
    safe_L2 = np.where(cand_L2 <= 0, 1.0, cand_L2)
    t_proj = np.clip(((p_x - x_from) * cand_dx + (p_y - y_from) * cand_dy) / safe_L2, 0.0, 1.0)
    dist_along_arc = t_proj * c_arc_len

    # Heading difference
    c_gps_head = gps_heading[pt_indices]
    h_diff = angle_diff(c_gps_head, c_arc_bear)
    cos_h_diff = np.cos(h_diff)

    # Road network features
    deg_start = degree_counts[a_f]
    deg_end = degree_counts[a_t]
    is_junc = ((deg_start >= 3) | (deg_end >= 3)).astype(np.int8)

    # Interaction features
    dist_ratio = c[dist_col].to_numpy() / np.maximum(min_dist_per_point + 1.0, 1.0)
    speed_align = gps_speed[pt_indices] * cos_h_diff

    # Topological connectivity to top candidates of previous point
    # We find for each point which nodes are touched by its top-3 candidates
    top3_cands = c[c[rank_col] < 3]
    top3_nodes = {}
    for p_idx, group in top3_cands.groupby("point_idx"):
        g_arcs = group["arc_id"].to_numpy()
        nodes_touched = set(np.concatenate([a_from[g_arcs], a_to[g_arcs]]))
        top3_nodes[p_idx] = nodes_touched

    prev_connected = np.zeros(len(c), dtype=np.int8)
    for idx, (p_idx, arc_id) in enumerate(zip(pt_indices, cand_arcs)):
        if p_idx > 0 and (p_idx - 1) in top3_nodes:
            prev_nodes = top3_nodes[p_idx - 1]
            if (a_from[arc_id] in prev_nodes) or (a_to[arc_id] in prev_nodes):
                prev_connected[idx] = 1

    # 4. Assemble Output DataFrame
    features_df = pd.DataFrame({
        "trip_id": trip_id,
        "point_idx": pt_indices,
        "arc_id": cand_arcs,
        "perpendicular_distance": c[dist_col].to_numpy(dtype=np.float32),
        "distance_rank": c[rank_col].to_numpy(dtype=np.int16),
        "dist_along_arc": dist_along_arc.astype(np.float32),
        "arc_length": c_arc_len.astype(np.float32),
        "arc_bearing": c_arc_bear.astype(np.float32),
        "heading_bearing_diff": h_diff.astype(np.float32),
        "cos_heading_diff": cos_h_diff.astype(np.float32),
        "gps_speed": gps_speed[pt_indices].astype(np.float32),
        "time_diff": time_diff[pt_indices].astype(np.float32),
        "distance_travelled": dist_travelled[pt_indices].astype(np.float32),
        "gps_heading": c_gps_head.astype(np.float32),
        "speed_forward": speed_forward[pt_indices].astype(np.float32),
        "acceleration": acceleration[pt_indices].astype(np.float32),
        "start_node_degree": deg_start.astype(np.int16),
        "end_node_degree": deg_end.astype(np.int16),
        "is_junction": is_junc,
        "prev_candidate_connected": prev_connected,
        "dist_ratio": dist_ratio.astype(np.float32),
        "speed_heading_alignment": speed_align.astype(np.float32),
    })

    if TARGET_COLUMN in c.columns:
        features_df[TARGET_COLUMN] = c[TARGET_COLUMN].to_numpy(dtype=np.int8)
    if VALID_COLUMN in c.columns:
        features_df[VALID_COLUMN] = c[VALID_COLUMN].to_numpy(dtype=np.int8)

    # Verify no NaN or infinite values in feature columns
    nan_counts = features_df[FEATURE_COLUMNS].isna().sum().sum()
    if nan_counts > 0:
        features_df[FEATURE_COLUMNS] = features_df[FEATURE_COLUMNS].fillna(0.0)

    return features_df
