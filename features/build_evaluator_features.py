import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from features.build_features import build_partition_features

if __name__ == "__main__":
    build_partition_features(partition="evaluator_test", valid_only=True)
