from __future__ import annotations

import argparse
from pathlib import Path
import sys

from utils.data_loader import quick_sanity_check, summarize_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2 verifier for UTD-MHAD raw dataset")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data root containing RGB/Depth/Skeleton/Inertial folders",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if dataset is incomplete or sanity check fails",
    )
    args = parser.parse_args()

    raw_root = args.raw_root
    summary = summarize_dataset(raw_root)

    print("=" * 64)
    print("Step 2 Dataset Verification")
    print("=" * 64)
    print(f"Raw root: {raw_root.resolve()}")
    print("\nModality sample counts:")
    for modality, count in summary["counts"].items():
        print(f"  - {modality:8s}: {count}")

    print("\nCross-modality coverage:")
    print(f"  - Union samples : {summary['union_samples']}")
    print(f"  - Common samples: {summary['common_samples']}")

    print("\nMissing prefixes by modality (relative to union):")
    for modality, missing in summary["missing_by_modality"].items():
        print(f"  - {modality:8s}: {missing}")

    print("\nInvalid naming counts:")
    for modality, invalid in summary["naming_invalid_counts"].items():
        print(f"  - {modality:8s}: {invalid}")

    sanity = quick_sanity_check(raw_root)
    print("\nQuick modality load check:")
    if sanity.get("ok"):
        print(f"  - prefix          : {sanity['prefix']}")
        print(f"  - skeleton shape  : {sanity['skeleton_shape']}")
        print(f"  - inertial shape  : {sanity['inertial_shape']}")
        print(f"  - depth shape     : {sanity['depth_shape']}")
        print(f"  - RGB frame shape : {sanity['rgb_frame_shape']}")
    else:
        print(f"  - failed: {sanity.get('reason', 'unknown reason')}")

    ready = (
        summary["common_samples"] > 0
        and all(v == 0 for v in summary["naming_invalid_counts"].values())
        and bool(sanity.get("ok"))
    )

    print("\nReadiness verdict:")
    print(f"  - STEP 2 complete: {'YES' if ready else 'NO'}")

    if args.strict and not ready:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
