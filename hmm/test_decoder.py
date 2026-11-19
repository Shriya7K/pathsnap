"""Verification script for HMM decoder."""
import sys
import time
from pathlib import Path

# Add project root
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from hmm.decoder import FullPipelineDecoder

def main():
    print("Testing FullPipelineDecoder on train trip 00000001...")
    t0 = time.time()
    decoder = FullPipelineDecoder()
    result = decoder.match_trip("data/train/00000001")
    elapsed = time.time() - t0

    print("=" * 60)
    print("           HMM MAP MATCHING TEST RESULTS")
    print("=" * 60)
    print(f"Trip ID:                 {result['trip_id']}")
    print(f"GPS Observations:        {result['point_count']:,}")
    print(f"Total Road Distance:     {result['total_distance_m']:,.1f} meters")
    print(f"Unique Matched Arcs:     {len(result['unique_route_arcs'])}")
    print(f"Mean Confidence Score:   {result['mean_confidence'] * 100:.2f}%")
    print(f"Ground Truth Accuracy:   {result.get('ground_truth_accuracy')}%")
    print(f"Total End-to-End Latency:{result['execution_time_sec']}s (Full test: {elapsed:.2f}s)")
    print("=" * 60)

    # Verify output integrity
    assert result["point_count"] > 0, "No points processed"
    assert len(result["matched_arcs"]) == result["point_count"], "Mismatch between points and arcs"
    assert len(result["matched_coordinates"]) == result["point_count"], "Mismatch in coordinate length"
    print("All assertions passed successfully!")

if __name__ == "__main__":
    main()
