from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models.data_level_fusion import train_data_level_fusion
from models.decision_level_fusion import (
    decision_level_fusion_learned,
    decision_level_majority_vote,
    train_unimodal_classifiers,
)
from models.feature_level_fusion import train_feature_level_fusion
from utils.evaluation import compute_metrics, save_confusion_matrix_plot
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

    missing = [
        str(path)
        for path in list(modality_files.values()) + [labels_path, subjects_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing processed files:\n" + "\n".join(missing))

    features = {name: np.load(path) for name, path in modality_files.items()}
    labels = np.load(labels_path)
    subjects = np.load(subjects_path)
    return features, labels, subjects


def _save_comparison_plot(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = frame.copy()
    plot_frame = plot_frame.sort_values("accuracy", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_frame, x="method", y="accuracy", hue="method", palette="viridis", dodge=False)
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title("Step 9 - Fusion Strategy Accuracy Comparison")
    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=20, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _to_row(method: str, metrics: Dict[str, float]) -> Dict[str, float | str]:
    return {
        "method": method,
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 9: Run all fusion strategies end-to-end")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-subjects", type=str, default="1,3,5,7")
    parser.add_argument("--feature-threshold", type=float, default=75.0)
    parser.add_argument("--late-epochs", type=int, default=120)
    parser.add_argument("--late-lr", type=float, default=1e-3)
    args = parser.parse_args()

    test_subjects = [int(x.strip()) for x in args.test_subjects.split(",") if x.strip()]

    features, labels, subjects = _load_processed_features(args.processed_dir)
    split = get_subject_split(subjects=subjects, test_subjects=test_subjects)

    tables_dir = args.results_dir / "tables"
    plots_dir = args.results_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Step 5: Data-level fusion
    y_true_data, y_pred_data, _ = train_data_level_fusion(
        features=features,
        labels=labels,
        train_idx=split.train_idx,
        test_idx=split.test_idx,
    )
    data_metrics = compute_metrics(y_true=y_true_data, y_pred=y_pred_data)
    save_confusion_matrix_plot(
        y_true=y_true_data,
        y_pred=y_pred_data,
        output_png=plots_dir / "step9_data_level_confusion_matrix.png",
        title="Step 9 - Data-Level Fusion",
    )

    # Step 6: Feature-level fusion
    y_true_feature, y_pred_feature, selected_counts, _, _ = train_feature_level_fusion(
        features=features,
        labels=labels,
        train_idx=split.train_idx,
        test_idx=split.test_idx,
        threshold_percentile=args.feature_threshold,
    )
    feature_metrics = compute_metrics(y_true=y_true_feature, y_pred=y_pred_feature)
    save_confusion_matrix_plot(
        y_true=y_true_feature,
        y_pred=y_pred_feature,
        output_png=plots_dir / "step9_feature_level_confusion_matrix.png",
        title="Step 9 - Feature-Level Fusion",
    )

    # Step 7: Decision-level fusion
    modality_order = ("skeleton", "inertial", "depth", "rgb")
    score_train, score_test, _, _ = train_unimodal_classifiers(
        features=features,
        labels=labels,
        train_idx=split.train_idx,
        test_idx=split.test_idx,
        modality_order=modality_order,
    )

    y_test = labels[split.test_idx]
    y_train = labels[split.train_idx]

    y_true_vote, y_pred_vote = decision_level_majority_vote(
        score_test=score_test,
        labels_test=y_test,
        modality_order=modality_order,
    )
    vote_metrics = compute_metrics(y_true=y_true_vote, y_pred=y_pred_vote)
    save_confusion_matrix_plot(
        y_true=y_true_vote,
        y_pred=y_pred_vote,
        output_png=plots_dir / "step9_decision_level_vote_confusion_matrix.png",
        title="Step 9 - Decision-Level Vote",
    )

    y_true_learned, y_pred_learned, _ = decision_level_fusion_learned(
        score_train=score_train,
        score_test=score_test,
        labels_train=y_train,
        labels_test=y_test,
        modality_order=modality_order,
        epochs=args.late_epochs,
        lr=args.late_lr,
    )
    learned_metrics = compute_metrics(y_true=y_true_learned, y_pred=y_pred_learned)
    save_confusion_matrix_plot(
        y_true=y_true_learned,
        y_pred=y_pred_learned,
        output_png=plots_dir / "step9_decision_level_learned_confusion_matrix.png",
        title="Step 9 - Decision-Level Learned",
    )

    rows = [
        _to_row("Data-Level", data_metrics),
        _to_row("Feature-Level", feature_metrics),
        _to_row("Decision-Level Vote", vote_metrics),
        _to_row("Decision-Level Learned", learned_metrics),
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(tables_dir / "step9_fusion_comparison.csv", index=False)

    ranking = comparison.sort_values(["accuracy", "f1_macro"], ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    ranking.to_csv(tables_dir / "step9_fusion_ranking.csv", index=False)

    _save_comparison_plot(comparison, plots_dir / "step9_fusion_accuracy_comparison.png")

    print("=" * 64)
    print("Step 9 Completed: End-to-End Fusion Execution")
    print("=" * 64)
    print(f"Train samples: {split.train_idx.shape[0]}")
    print(f"Test samples : {split.test_idx.shape[0]}")
    print("\nFeature-level selected features:")
    print(f"  - total: {sum(selected_counts.values())}")
    for modality, count in selected_counts.items():
        print(f"  - {modality:8s}: {count}")

    print("\nFusion metrics:")
    for _, row in comparison.iterrows():
        print(
            f"  - {row['method']:22s} | "
            f"acc={row['accuracy']:.4f} | "
            f"f1={row['f1_macro']:.4f}"
        )

    best = ranking.iloc[0]
    print("\nBest method:")
    print(f"  - {best['method']} (accuracy={best['accuracy']:.4f}, f1={best['f1_macro']:.4f})")

    print("\nArtifacts:")
    print(f"  - {tables_dir / 'step9_fusion_comparison.csv'}")
    print(f"  - {tables_dir / 'step9_fusion_ranking.csv'}")
    print(f"  - {plots_dir / 'step9_fusion_accuracy_comparison.png'}")
    print(f"  - {plots_dir / 'step9_data_level_confusion_matrix.png'}")
    print(f"  - {plots_dir / 'step9_feature_level_confusion_matrix.png'}")
    print(f"  - {plots_dir / 'step9_decision_level_vote_confusion_matrix.png'}")
    print(f"  - {plots_dir / 'step9_decision_level_learned_confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
