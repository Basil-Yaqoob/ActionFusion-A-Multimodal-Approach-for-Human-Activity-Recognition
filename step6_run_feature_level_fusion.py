from __future__ import annotations

import argparse
from pathlib import Path
import sys

from models.feature_level_fusion import train_feature_level_fusion
from step5_run_data_level_fusion import _load_processed_features
from utils.evaluation import compute_metrics, save_confusion_matrix_plot, save_metrics_table
from utils.splits import get_subject_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 6: Feature-level fusion run")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-subjects", type=str, default="1,3,5,7")
    parser.add_argument("--threshold-percentile", type=float, default=75.0)
    args = parser.parse_args()

    test_subjects = [int(x.strip()) for x in args.test_subjects.split(",") if x.strip()]

    features, labels, subjects = _load_processed_features(args.processed_dir)
    split = get_subject_split(subjects=subjects, test_subjects=test_subjects)

    y_true, y_pred, selected_counts, _clf, _scaler = train_feature_level_fusion(
        features=features,
        labels=labels,
        train_idx=split.train_idx,
        test_idx=split.test_idx,
        threshold_percentile=args.threshold_percentile,
    )

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    total_selected = int(sum(selected_counts.values()))
    metrics["threshold_percentile"] = float(args.threshold_percentile)
    metrics["selected_total"] = float(total_selected)
    for modality, count in selected_counts.items():
        metrics[f"selected_{modality}"] = float(count)

    tables_dir = args.results_dir / "tables"
    plots_dir = args.results_dir / "plots"
    save_metrics_table(metrics=metrics, output_csv=tables_dir / "step6_feature_level_metrics.csv")
    save_confusion_matrix_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_png=plots_dir / "step6_feature_level_confusion_matrix.png",
        title="Step 6 - Feature-Level Fusion Confusion Matrix",
    )

    print("=" * 64)
    print("Step 6 Completed: Feature-Level Fusion")
    print("=" * 64)
    print(f"Train samples: {split.train_idx.shape[0]}")
    print(f"Test samples : {split.test_idx.shape[0]}")
    print("\nSelected features per modality:")
    for modality, count in selected_counts.items():
        print(f"  - {modality:8s}: {count}")
    print(f"  - total    : {total_selected}")
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  - {name:20s}: {value:.4f}")
    print("\nArtifacts:")
    print(f"  - {tables_dir / 'step6_feature_level_metrics.csv'}")
    print(f"  - {plots_dir / 'step6_feature_level_confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
