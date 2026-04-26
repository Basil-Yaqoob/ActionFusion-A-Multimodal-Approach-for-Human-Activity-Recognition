from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from models.data_level_fusion import train_data_level_fusion
from utils.evaluation import compute_metrics, save_confusion_matrix_plot, save_metrics_table
from utils.splits import get_subject_split


def _load_processed_features(processed_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    modality_files = {
        "skeleton": processed_dir / "skeleton_features.npy",
        "inertial": processed_dir / "inertial_features.npy",
        "depth": processed_dir / "depth_features.npy",
        "rgb": processed_dir / "rgb_features.npy",
    }
    labels_path = processed_dir / "labels.npy"
    subjects_path = processed_dir / "subjects.npy"

    missing = [str(path) for path in list(modality_files.values()) + [labels_path, subjects_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing processed files:\n" + "\n".join(missing))

    features = {name: np.load(path) for name, path in modality_files.items()}
    labels = np.load(labels_path)
    subjects = np.load(subjects_path)
    return features, labels, subjects


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 5: Data-level (early) fusion run")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-subjects", type=str, default="1,3,5,7")
    args = parser.parse_args()

    test_subjects = [int(x.strip()) for x in args.test_subjects.split(",") if x.strip()]

    features, labels, subjects = _load_processed_features(args.processed_dir)
    split = get_subject_split(subjects=subjects, test_subjects=test_subjects)

    y_true, y_pred, _model = train_data_level_fusion(
        features=features,
        labels=labels,
        train_idx=split.train_idx,
        test_idx=split.test_idx,
    )

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)

    tables_dir = args.results_dir / "tables"
    plots_dir = args.results_dir / "plots"
    save_metrics_table(metrics=metrics, output_csv=tables_dir / "step5_data_level_metrics.csv")
    save_confusion_matrix_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_png=plots_dir / "step5_data_level_confusion_matrix.png",
        title="Step 5 - Data-Level Fusion Confusion Matrix",
    )

    print("=" * 64)
    print("Step 5 Completed: Data-Level Fusion")
    print("=" * 64)
    print(f"Train samples: {split.train_idx.shape[0]}")
    print(f"Test samples : {split.test_idx.shape[0]}")
    print(f"Train subjects: {list(split.train_subjects)}")
    print(f"Test subjects : {list(split.test_subjects)}")
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  - {name:16s}: {value:.4f}")
    print("\nArtifacts:")
    print(f"  - {tables_dir / 'step5_data_level_metrics.csv'}")
    print(f"  - {plots_dir / 'step5_data_level_confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
