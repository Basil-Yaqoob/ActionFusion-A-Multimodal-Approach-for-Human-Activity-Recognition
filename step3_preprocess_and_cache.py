from __future__ import annotations

import argparse
from pathlib import Path
import sys

from utils.preprocessing import PreprocessConfig, build_dataset, save_feature_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3 preprocessing and cache builder")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["skeleton", "inertial", "depth", "rgb"],
        choices=["skeleton", "inertial", "depth", "rgb"],
        help="Modalities to preprocess",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = PreprocessConfig()
    features, labels, subjects, prefixes = build_dataset(
        raw_root=args.raw_root,
        modalities=args.modalities,
        config=config,
        max_samples=args.max_samples,
    )

    save_feature_cache(
        output_dir=args.out_dir,
        features=features,
        labels=labels,
        subjects=subjects,
        prefixes=prefixes,
    )

    print("=" * 64)
    print("Step 3 Completed")
    print("=" * 64)
    print(f"Raw root: {args.raw_root.resolve()}")
    print(f"Output : {args.out_dir.resolve()}")
    print(f"Samples: {labels.shape[0]}")
    print("Feature shapes:")
    for modality, array in features.items():
        print(f"  - {modality:8s}: {tuple(array.shape)}")
    print(f"Labels shape  : {tuple(labels.shape)}")
    print(f"Subjects shape: {tuple(subjects.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
