"""Data loading utility for Person 2 (Features & Modeling).

Provides clean access to candidate datasets with filtering options:
- load_trip(partition, trip_id, k, r, valid_only): Loads candidate dataframe for a trip.
- load_all(partition, k, r, valid_only): Loads concatenated candidates across all trips.
- list_trips(partition): Lists all available processed trip IDs.
"""
from pathlib import Path
import pandas as pd

CAND_DIR = Path("results/candidates")
K_DEFAULT, R_DEFAULT = 30, 100.0


def list_trips(partition=None):
    """Return [(partition, trip_id), ...] for every generated trip."""
    out = []
    pattern = f"{partition}_*_cands.csv.gz" if partition else "*_cands.csv.gz"
    for f in sorted(CAND_DIR.glob(pattern)):
        name = f.name.replace("_cands.csv.gz", "")
        part, trip = name.rsplit("_", 1)
        out.append((part, trip))
    return out


def load_trip(partition, trip_id, k=K_DEFAULT, r=R_DEFAULT, valid_only=False):
    """Load candidate dataset for a specific trip.

    Parameters:
    -----------
    partition : str ('train', 'student_test', etc.)
    trip_id : str (e.g. '00000001')
    k : int, maximum candidate rank to include (default 30)
    r : float, maximum perpendicular distance in meters to include (default 100.0)
    valid_only : bool, if True filters to observations with label_valid == 1 (default False)

    Returns:
    --------
    pd.DataFrame with columns:
        - trip_id (str): Identifier of the trip
        - point_idx (int): 0-indexed sequential GPS point within trip
        - arc_id (int): 0-indexed local road arc identifier within trip network
        - label (int): 1 if candidate is the ground-truth reference arc, 0 otherwise
        - perpendicular_distance (float): Orthogonal distance in meters from point to arc
        - distance_rank (int): Rank of candidate by proximity (0 = closest)
        - label_valid (int): 1 if point has reliable ground truth (<=50m), 0 if off-route
        - x (float): UTM Easting coordinate (meters)
        - y (float): UTM Northing coordinate (meters)
        - ts (float): Timestamp of GPS observation
    """
    c_path = CAND_DIR / f"{partition}_{trip_id}_cands.csv.gz"
    if not c_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {c_path}. Run candidates.py first.")

    c = pd.read_csv(c_path, dtype={"trip_id": str, "trip": str})

    # Filter to requested candidate radius and rank cap
    dist_col = "perpendicular_distance" if "perpendicular_distance" in c.columns else "dist"
    rank_col = "distance_rank" if "distance_rank" in c.columns else "rank"
    c = c[(c[rank_col] < k) & (c[dist_col] <= r)].copy()

    if valid_only:
        c = c[c["label_valid"] == 1].copy()

    # Standardize column names
    rename_map = {"dist": "perpendicular_distance", "rank": "distance_rank", "trip": "trip_id"}
    for old_col, new_col in rename_map.items():
        if old_col in c.columns and new_col not in c.columns:
            c[new_col] = c[old_col]

    primary_cols = [
        "trip_id", "point_idx", "arc_id", "label",
        "perpendicular_distance", "distance_rank",
        "label_valid", "x", "y", "ts"
    ]
    avail_cols = [col for col in primary_cols if col in c.columns]
    # Preserve backwards compatibility aliases
    if "perpendicular_distance" in c.columns and "dist" not in c.columns:
        c["dist"] = c["perpendicular_distance"]
    if "distance_rank" in c.columns and "rank" not in c.columns:
        c["rank"] = c["distance_rank"]

    return c.reset_index(drop=True)


def load_all(partition="train", k=K_DEFAULT, r=R_DEFAULT, valid_only=True):
    """Load and concatenate all trips in a partition into a single DataFrame.

    Ideal for Person 2 feature engineering and group-aware cross-validation.
    """
    trips = list_trips(partition=partition)
    if not trips:
        raise FileNotFoundError(f"No trips found for partition '{partition}' in {CAND_DIR}")

    dfs = [load_trip(part, trip, k=k, r=r, valid_only=valid_only) for part, trip in trips]
    combined = pd.concat(dfs, ignore_index=True)
    return combined


if __name__ == "__main__":
    trips = list_trips()
    print(f"Discovered {len(trips)} pre-processed trips in {CAND_DIR}.")
    if trips:
        part, sample_trip = trips[0]
        sample_df = load_trip(part, sample_trip)
        print(f"Sample trip ({part}/{sample_trip}) loaded: {len(sample_df)} candidate rows.")
        print("Schema columns:", list(sample_df.columns))
        print("First 3 rows:\n", sample_df.head(3))
