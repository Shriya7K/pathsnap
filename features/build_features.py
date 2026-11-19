"""Batch feature dataset builder for PathSnap Person 2."""
import argparse
import time
from pathlib import Path
import pandas as pd

from features.config import CANDIDATE_K, CANDIDATE_R
from features.engineer import extract_trip_features


def build_partition_features(
    partition: str = "train",
    data_dir: Path = Path("data"),
    cand_dir: Path = Path("results/candidates"),
    out_dir: Path = Path("results/features"),
    k: int = CANDIDATE_K,
    r: float = CANDIDATE_R,
    valid_only: bool = True,
    sample_trips: int = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{partition}_features.csv.gz"

    pattern = f"{partition}_*_cands.csv.gz"
    cand_files = sorted(cand_dir.glob(pattern))

    if not cand_files:
        raise FileNotFoundError(f"No candidate files found for partition '{partition}' in {cand_dir}")

    trips = [f.name.replace("_cands.csv.gz", "").rsplit("_", 1)[1] for f in cand_files]
    if sample_trips and len(trips) > sample_trips:
        trips = trips[:sample_trips]

    print(f"Building features for {partition} partition ({len(trips)} trips, K={k}, R={r}m)...")
    t0 = time.time()
    dfs = []
    for idx, trip_id in enumerate(trips, 1):
        try:
            df = extract_trip_features(
                partition=partition,
                trip_id=trip_id,
                data_dir=data_dir,
                cand_dir=cand_dir,
                k=k,
                r=r,
                valid_only=valid_only,
            )
            if len(df):
                dfs.append(df)
            if idx % 10 == 0 or idx == len(trips):
                print(f"  Processed {idx}/{len(trips)} trips ({time.time() - t0:.1f}s)...")
        except Exception as e:
            print(f"  [ERROR] Failed to extract features for trip {trip_id}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_file, index=False, compression="gzip")
    elapsed = time.time() - t0
    pos_count = int(combined["label"].sum()) if "label" in combined.columns else 0
    print(f"Saved {out_file} ({len(combined):,} rows, {pos_count:,} positives, {elapsed:.1f}s)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Build feature datasets for Person 2.")
    parser.add_argument("--k", type=int, default=CANDIDATE_K, help="Max candidates per observation")
    parser.add_argument("--r", type=float, default=CANDIDATE_R, help="Max candidate radius (m)")
    parser.add_argument("--sample", type=int, default=None, help="Optional trip sample count")
    args = parser.parse_args()

    # Build train partition
    build_partition_features(partition="train", k=args.k, r=args.r, valid_only=True, sample_trips=args.sample)
    # Build student_test partition
    build_partition_features(partition="student_test", k=args.k, r=args.r, valid_only=True, sample_trips=args.sample)


if __name__ == "__main__":
    main()
