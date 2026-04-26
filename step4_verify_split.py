from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from utils.splits import get_stratified_kfold_splits, get_subject_split, summarize_split


def _parse_subjects(value: str) -> list[int]:
    chunks = [x.strip() for x in value.split(",") if x.strip()]
    if not chunks:
        raise argparse.ArgumentTypeError("test-subjects must contain at least one subject ID")
    try:
        return [int(x) for x in chunks]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("test-subjects must be comma-separated integers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4 split verification")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--test-subjects",
        type=_parse_subjects,
        default=[1, 3, 5, 7],
        help="Comma-separated test subjects for subject split (default: 1,3,5,7)",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    processed_dir = args.processed_dir
    labels_path = processed_dir / "labels.npy"
    subjects_path = processed_dir / "subjects.npy"

    if not labels_path.exists() or not subjects_path.exists():
        print("[ERROR] Missing labels.npy or subjects.npy. Run Step 3 first.")
        return 1

    labels = np.load(labels_path)
    subjects = np.load(subjects_path)

    split = get_subject_split(subjects=subjects, test_subjects=args.test_subjects)
    summary = summarize_split(
        labels=labels,
        subjects=subjects,
        train_idx=split.train_idx,
        test_idx=split.test_idx,
    )

    cv_splits = get_stratified_kfold_splits(
        labels=labels,
        n_splits=args.cv_folds,
        random_state=args.random_state,
    )

    print("=" * 64)
    print("Step 4 Split Verification")
    print("=" * 64)
    print(f"Processed dir: {processed_dir.resolve()}")
    print(f"Total samples : {labels.shape[0]}")
    print("\nSubject-based split summary:")
    print(f"  - train samples   : {summary['train_samples']}")
    print(f"  - test samples    : {summary['test_samples']}")
    print(f"  - train subjects  : {summary['train_subjects']}")
    print(f"  - test subjects   : {summary['test_subjects']}")
    print(f"  - subject overlap : {summary['subject_overlap']}")
    print(f"  - train classes   : {len(summary['train_classes'])}")
    print(f"  - test classes    : {len(summary['test_classes'])}")
    print(f"  - class overlap   : {summary['class_overlap_count']}")

    print("\nStratified K-Fold summary:")
    print(f"  - folds           : {len(cv_splits)}")
    first_train, first_test = cv_splits[0]
    print(f"  - fold[0] sizes   : train={first_train.shape[0]}, test={first_test.shape[0]}")

    ok = (
        summary["train_samples"] > 0
        and summary["test_samples"] > 0
        and len(summary["subject_overlap"]) == 0
        and len(cv_splits) == args.cv_folds
    )

    print("\nReadiness verdict:")
    print(f"  - STEP 4 complete: {'YES' if ok else 'NO'}")

    if args.strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
