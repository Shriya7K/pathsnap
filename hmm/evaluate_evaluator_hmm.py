"""Run full HMM + Viterbi map matching across all 10 evaluator_test trips."""
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from hmm.decoder import FullPipelineDecoder


def main():
    data_dir = Path("data/evaluator_test")
    trip_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    print(f"Executing End-to-End Map Matching on {len(trip_dirs)} evaluator_test trips...")
    decoder = FullPipelineDecoder()

    trip_summaries = []
    all_predictions = []
    t_start = time.time()

    for t_dir in trip_dirs:
        trip_id = t_dir.name
        t0 = time.time()
        res = decoder.match_trip(t_dir)
        elapsed = time.time() - t0

        acc = res.get("ground_truth_accuracy")
        print(f"  Trip {trip_id}: {res['point_count']:,} pts | {len(res['unique_route_arcs'])} arcs | "
              f"Dist: {res['total_distance_m']/1000.0:.2f} km | Confidence: {res['mean_confidence']*100:.1f}% | "
              f"Accuracy: {acc}% | Latency: {elapsed:.2f}s")

        trip_summaries.append({
            "trip_id": trip_id,
            "point_count": res["point_count"],
            "unique_arcs_count": len(res["unique_route_arcs"]),
            "total_distance_km": round(res["total_distance_m"] / 1000.0, 3),
            "mean_confidence": res["mean_confidence"],
            "ground_truth_accuracy_pct": acc,
            "execution_time_sec": round(elapsed, 3),
        })

        for p_idx, (arc, conf, raw_c, snap_c) in enumerate(zip(
            res["matched_arcs"], res["confidences"], res["raw_coordinates"], res["matched_coordinates"]
        )):
            all_predictions.append({
                "trip_id": trip_id,
                "point_idx": p_idx,
                "raw_latitude": raw_c[0],
                "raw_longitude": raw_c[1],
                "matched_arc_id": arc,
                "snapped_latitude": snap_c[0],
                "snapped_longitude": snap_c[1],
                "confidence": conf,
            })

    total_elapsed = time.time() - t_start
    df_summary = pd.DataFrame(trip_summaries)
    df_preds = pd.DataFrame(all_predictions)

    # Save outputs
    out_summary = Path("results/evaluator_test_hmm_metrics.csv")
    out_preds = Path("results/evaluator_test_predictions.csv.gz")
    df_summary.to_csv(out_summary, index=False)
    df_preds.to_csv(out_preds, index=False, compression="gzip")

    mean_acc = df_summary["ground_truth_accuracy_pct"].dropna().mean()
    total_pts = df_summary["point_count"].sum()
    total_km = df_summary["total_distance_km"].sum()

    print("\n" + "=" * 65)
    print("        EVALUATOR TEST: FULL PIPELINE HMM MAP MATCHING REPORT")
    print("=" * 65)
    print(f"Total Trips Evaluated:          {len(df_summary)}")
    print(f"Total GPS Observations:         {total_pts:,}")
    print(f"Total Trajectory Distance:      {total_km:.2f} km")
    print(f"Mean Ground-Truth Sequence Acc: {mean_acc:.2f}%")
    print(f"Average Pipeline Latency / Trip:{df_summary['execution_time_sec'].mean():.2f}s")
    print(f"Total Processing Time:          {total_elapsed:.1f}s")
    print("=" * 65)
    print(df_summary[["trip_id", "point_count", "total_distance_km", "ground_truth_accuracy_pct", "execution_time_sec"]].to_string(index=False))
    print("=" * 65)
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_preds}")


if __name__ == "__main__":
    main()
