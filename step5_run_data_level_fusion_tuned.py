from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from models.data_level_fusion import train_data_level_fusion_tuned
from step5_run_data_level_fusion import _load_processed_features
from utils.evaluation import compute_metrics, save_confusion_matrix_plot, save_metrics_table
from utils.splits import get_subject_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 5 (tuned): Data-level fusion run")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-subjects", type=str, default="1,3,5,7")
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()

    test_subjects = [int(x.strip()) for x in args.test_subjects.split(",") if x.strip()]

    features, labels, subjects = _load_processed_features(args.processed_dir)
    split = get_subject_split(subjects=subjects, test_subjects=test_subjects)

    y_true, y_pred, search = train_data_level_fusion_tuned(
        features=features,
        labels=labels,
        train_idx=split.train_idx,
        test_idx=split.test_idx,
        pca_components=args.pca_components,
        cv_folds=args.cv_folds,
    )

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    metrics["best_cv_f1_macro"] = float(search.best_score_)
    metrics["best_params"] = str(search.best_params_)

    tables_dir = args.results_dir / "tables"
    plots_dir = args.results_dir / "plots"
    save_metrics_table(metrics=metrics, output_csv=tables_dir / "step5_data_level_tuned_metrics.csv")
    save_confusion_matrix_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_png=plots_dir / "step5_data_level_tuned_confusion_matrix.png",
        title="Step 5 - Data-Level Fusion (Tuned) Confusion Matrix",
    )

    print("=" * 64)
    print("Step 5 Completed: Data-Level Fusion (Tuned)")
    print("=" * 64)
    print(f"Train samples: {split.train_idx.shape[0]}")
    print(f"Test samples : {split.test_idx.shape[0]}")
    print(f"Best CV params: {search.best_params_}")
    print(f"Best CV F1-macro: {search.best_score_:.4f}")
    print("\nMetrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  - {name:20s}: {value:.4f}")
        else:
            print(f"  - {name:20s}: {value}")
    print("\nArtifacts:")
    print(f"  - {tables_dir / 'step5_data_level_tuned_metrics.csv'}")
    print(f"  - {plots_dir / 'step5_data_level_tuned_confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
