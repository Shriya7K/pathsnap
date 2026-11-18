
"""Label-free feature engineering for PathSnap."""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "dist",
    "log_dist",
    "rank",
    "arc_length",
    "projection_ratio",
    "dist_to_start",
    "dist_to_end",
    "gps_speed",
    "gps_move_distance",
    "time_delta",
    "heading_diff",
    "heading_alignment",
    "heading_valid",
    "gps_heading_change",
]


def _gps_features(td):
    """
    Calculate movement features using GPS observations only.

    Movement across invalid or unusually large time gaps is ignored.
    The final GPS point has no outgoing movement information.
    """

    xy = np.asarray(td.xy, dtype=float)
    ts = np.asarray(td.ts, dtype=float)

    n = len(xy)

    heading = np.zeros(n)
    speed = np.zeros(n)
    move_distance = np.zeros(n)
    time_delta = np.zeros(n)
    heading_change = np.zeros(n)
    heading_valid = np.zeros(n, dtype=int)

    if n < 2:
        return (
            heading,
            speed,
            move_distance,
            time_delta,
            heading_change,
            heading_valid,
        )

    # Calculate movement between consecutive GPS observations.
    delta_xy = np.diff(xy, axis=0)
    delta_ts = np.diff(ts)

    distances = np.linalg.norm(delta_xy, axis=1)

    # Accept only positive time gaps up to 10 seconds.
    valid_time = (delta_ts > 0) & (delta_ts <= 10)

    # Ignore stationary or nearly stationary observations.
    valid_move = (distances > 0.5) & valid_time

    segment_heading = np.arctan2(
        delta_xy[:, 1],
        delta_xy[:, 0],
    )

    segment_speed = np.zeros(n - 1)

    segment_speed[valid_move] = (
        distances[valid_move] / delta_ts[valid_move]
    )

    # Assign outgoing movement features to each starting GPS point.
    # The final GPS point has no outgoing segment, so its values remain zero.
    heading[:-1] = segment_heading
    speed[:-1] = segment_speed
    move_distance[:-1] = distances
    time_delta[:-1] = delta_ts

    heading_valid[:-1] = valid_move.astype(int)

    # Calculate direction changes only between two valid segments.
    if n > 2:
        angle_diff = np.arctan2(
            np.sin(np.diff(segment_heading)),
            np.cos(np.diff(segment_heading)),
        )

        valid_pairs = valid_move[:-1] & valid_move[1:]

        heading_change[1:-1] = np.where(
            valid_pairs,
            np.abs(angle_diff),
            0.0,
        )

    return (
        heading,
        speed,
        move_distance,
        time_delta,
        heading_change,
        heading_valid,
    )


def build_features(td):
    """
    Build one feature row per (GPS point, candidate arc).

    Uses only:
        td.xy, td.ts, td.nx, td.ny,
        td.a_from, td.a_to, td.cands

    Does not use route or ground-truth labels.
    """

    c = td.cands.copy().reset_index(drop=True)

    if c.empty:
        return pd.DataFrame(
            columns=["point_idx", "arc_id"] + FEATURE_COLUMNS
        )

    point_idx = c["point_idx"].to_numpy(dtype=int)
    arc_id = c["arc_id"].to_numpy(dtype=int)

    xy = np.asarray(td.xy, dtype=float)
    nx = np.asarray(td.nx, dtype=float)
    ny = np.asarray(td.ny, dtype=float)

    a_from = np.asarray(td.a_from, dtype=int)
    a_to = np.asarray(td.a_to, dtype=int)

    # Validate candidate indices.
    if np.any(arc_id < 0) or np.any(arc_id >= len(a_from)):
        raise ValueError(
            "Candidate arc_id is outside the network array."
        )

    if np.any(point_idx < 0) or np.any(point_idx >= len(xy)):
        raise ValueError(
            "Candidate point_idx is outside the GPS array."
        )

    if np.any(a_from < 0) or np.any(a_from >= len(nx)):
        raise ValueError("Invalid arc start-node index.")

    if np.any(a_to < 0) or np.any(a_to >= len(nx)):
        raise ValueError("Invalid arc end-node index.")

    # Candidate arc geometry.
    start_x = nx[a_from[arc_id]]
    start_y = ny[a_from[arc_id]]

    end_x = nx[a_to[arc_id]]
    end_y = ny[a_to[arc_id]]

    dx = end_x - start_x
    dy = end_y - start_y

    arc_length = np.hypot(dx, dy)

    # Project GPS points onto candidate arc segments.
    px = xy[point_idx, 0]
    py = xy[point_idx, 1]

    length_sq = dx**2 + dy**2

    safe_length_sq = np.where(
        length_sq > 0,
        length_sq,
        1.0,
    )

    projection_ratio = (
        (px - start_x) * dx + (py - start_y) * dy
    ) / safe_length_sq

    projection_ratio = np.clip(
        projection_ratio,
        0.0,
        1.0,
    )

    dist_to_start = np.hypot(
        px - start_x,
        py - start_y,
    )

    dist_to_end = np.hypot(
        px - end_x,
        py - end_y,
    )

    # Candidate arc direction.
    arc_heading = np.arctan2(dy, dx)

    # GPS movement features.
    (
        gps_heading,
        gps_speed,
        gps_move_distance,
        time_delta,
        gps_heading_change,
        heading_valid,
    ) = _gps_features(td)

    point_heading = gps_heading[point_idx]
    valid_heading = heading_valid[point_idx]

    raw_heading_diff = np.arctan2(
        np.sin(point_heading - arc_heading),
        np.cos(point_heading - arc_heading),
    )

    heading_diff = np.where(
        valid_heading == 1,
        np.abs(raw_heading_diff),
        0.0,
    )

    heading_alignment = np.where(
        valid_heading == 1,
        np.cos(raw_heading_diff),
        0.0,
    )

    # Construct the feature dataset.
    features = pd.DataFrame({
        "point_idx": point_idx,
        "arc_id": arc_id,
        "dist": c["dist"].to_numpy(dtype=float),
        "log_dist": np.log1p(
            c["dist"].to_numpy(dtype=float)
        ),
        "rank": c["rank"].to_numpy(dtype=float),
        "arc_length": arc_length,
        "projection_ratio": projection_ratio,
        "dist_to_start": dist_to_start,
        "dist_to_end": dist_to_end,
        "gps_speed": gps_speed[point_idx],
        "gps_move_distance": gps_move_distance[point_idx],
        "time_delta": time_delta[point_idx],
        "heading_diff": heading_diff,
        "heading_alignment": heading_alignment,
        "heading_valid": valid_heading,
        "gps_heading_change": gps_heading_change[point_idx],
    })

    # Replace infinite and missing model inputs.
    features[FEATURE_COLUMNS] = (
        features[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return features
