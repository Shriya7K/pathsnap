"""Pre-flight validator for a classifier probability file (run this before anything else).

    python -m hmm.check_probs --probs <file.csv|.csv.gz|.parquet> [--partitions train student_test]

Exit code 0 = no blocking problems, 1 = blocking problems (listed). It never modifies the file; the
optional fix flags (--renormalise, --fill-missing, --drop-duplicates, --accept-renames) only show what
the other scripts WOULD change when given the same flags. All logic lives in hmm/probs_io.py.
"""
import argparse
import sys

from .graph import DATA_ROOT, DEV_PARTITIONS
from .probs_io import CANDS_DIR, ProbsError, add_args, load_probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--partitions", nargs="+", default=list(DEV_PARTITIONS))
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--cands-dir", default=str(CANDS_DIR))
    add_args(ap)
    a = ap.parse_args(argv)
    try:
        lp = load_probs(a.probs, a.partitions, a.data_root, a.cands_dir, accept_renames=a.accept_renames,
                        renormalise=a.renormalise, fill_missing=a.fill_missing, drop_duplicates=a.drop_duplicates,
                        allow_partial=a.allow_partial, candidates="require")
    except ProbsError as e:
        print(f"=== probability file check: {a.probs} ===\nBLOCKING PROBLEM(S):\n{e}\nresult: FAILED")
        return 1
    print(lp.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
