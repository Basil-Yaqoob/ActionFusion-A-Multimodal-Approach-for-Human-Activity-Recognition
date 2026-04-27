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
from utils.splits import get_stratified_kfold_splits, get_subject_split


METHOD_ORDER = (
    "Data-Level",
    "Feature-Level",
    "Decision-Level Vote",
    "Decision-Level Learned",
)
MODALITY_ORDER = ("skeleton", "inertial", "depth", "rgb")


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


def _evaluate_split(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_threshold: float,
    late_epochs: int,
    late_lr: float,
) -> tuple[dict[str, Dict[str, float]], dict[str, np.ndarray], dict[str, int]]:
    y_true_data, y_pred_data, _ = train_data_level_fusion(
        features=features,
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    data_metrics = compute_metrics(y_true=y_true_data, y_pred=y_pred_data)

    y_true_feature, y_pred_feature, selected_counts, _, _ = train_feature_level_fusion(
        features=features,
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        threshold_percentile=feature_threshold,
    )
    feature_metrics = compute_metrics(y_true=y_true_feature, y_pred=y_pred_feature)

    score_train, score_test, _, _ = train_unimodal_classifiers(
        features=features,
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        modality_order=MODALITY_ORDER,
    )

    y_test = labels[test_idx]
    y_train = labels[train_idx]

    y_true_vote, y_pred_vote = decision_level_majority_vote(
        score_test=score_test,
        labels_test=y_test,
        modality_order=MODALITY_ORDER,
    )
    vote_metrics = compute_metrics(y_true=y_true_vote, y_pred=y_pred_vote)

    y_true_learned, y_pred_learned, _ = decision_level_fusion_learned(
        score_train=score_train,
        score_test=score_test,
        labels_train=y_train,
        labels_test=y_test,
        modality_order=MODALITY_ORDER,
        epochs=late_epochs,
        lr=late_lr,
    )
    learned_metrics = compute_metrics(y_true=y_true_learned, y_pred=y_pred_learned)

    metrics = {
        "Data-Level": data_metrics,
        "Feature-Level": feature_metrics,
        "Decision-Level Vote": vote_metrics,
        "Decision-Level Learned": learned_metrics,
    }
    predictions = {
        "Data-Level": y_pred_data,
        "Feature-Level": y_pred_feature,
        "Decision-Level Vote": y_pred_vote,
        "Decision-Level Learned": y_pred_learned,
    }
    return metrics, predictions, selected_counts


def _evaluate_subject_split(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
    test_subjects: list[int],
    feature_threshold: float,
    late_epochs: int,
    late_lr: float,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, np.ndarray], np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    split = get_subject_split(subjects=subjects, test_subjects=test_subjects)
    metrics, predictions, selected_counts = _evaluate_split(
        features=features,
        labels=labels,
        train_idx=split.train_idx,
        test_idx=split.test_idx,
        feature_threshold=feature_threshold,
        late_epochs=late_epochs,
        late_lr=late_lr,
    )
    comparison = pd.DataFrame([_to_row(method, metrics[method]) for method in METHOD_ORDER])
    return (
        comparison,
        selected_counts,
        predictions,
        labels[split.test_idx],
        labels[split.train_idx],
        split.train_subjects,
        split.test_subjects,
    )


def _evaluate_kfold(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    n_splits: int,
    random_state: int,
    feature_threshold: float,
    late_epochs: int,
    late_lr: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, list[int]], list[int], list[int]]:
    splits = get_stratified_kfold_splits(labels=labels, n_splits=n_splits, random_state=random_state)

    out_of_fold_predictions = {
        method: np.full(labels.shape[0], -1, dtype=np.int64)
        for method in METHOD_ORDER
    }
    fold_rows: list[dict[str, float | int | str]] = []
    selected_by_modality: dict[str, list[int]] = {modality: [] for modality in MODALITY_ORDER}
    train_sizes: list[int] = []
    test_sizes: list[int] = []

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n--- Fold {fold_number}/{n_splits} ---")
        metrics, predictions, selected_counts = _evaluate_split(
            features=features,
            labels=labels,
            train_idx=train_idx,
            test_idx=test_idx,
            feature_threshold=feature_threshold,
            late_epochs=late_epochs,
            late_lr=late_lr,
        )

        train_sizes.append(int(train_idx.shape[0]))
        test_sizes.append(int(test_idx.shape[0]))

        for method in METHOD_ORDER:
            out_of_fold_predictions[method][test_idx] = predictions[method]
            fold_rows.append({"fold": fold_number, **_to_row(method, metrics[method])})

        for modality, count in selected_counts.items():
            selected_by_modality[modality].append(int(count))

    if any(np.any(prediction == -1) for prediction in out_of_fold_predictions.values()):
        raise RuntimeError("K-fold predictions were not fully populated")

    comparison = pd.DataFrame([
        _to_row(method, compute_metrics(y_true=labels, y_pred=out_of_fold_predictions[method]))
        for method in METHOD_ORDER
    ])
    fold_metrics = pd.DataFrame(fold_rows)
    return comparison, fold_metrics, out_of_fold_predictions, selected_by_modality, train_sizes, test_sizes


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 9: Run all fusion strategies end-to-end")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--eval-mode",
        choices=("kfold", "subject"),
        default="subject",
        help="Run k-fold cross-validation or the original subject holdout split.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-subjects", type=str, default="1,3,5,7")
    parser.add_argument("--feature-threshold", type=float, default=75.0)
    parser.add_argument("--late-epochs", type=int, default=120)
    parser.add_argument("--late-lr", type=float, default=1e-3)
    args = parser.parse_args()

    tables_dir = args.results_dir / "tables"
    plots_dir = args.results_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    features, labels, subjects = _load_processed_features(args.processed_dir)

    if args.eval_mode == "subject":
        test_subjects = [int(x.strip()) for x in args.test_subjects.split(",") if x.strip()]
        (
            comparison,
            selected_counts,
            predictions,
            y_true_reference,
            y_train_reference,
            train_subjects,
            test_subjects_observed,
        ) = _evaluate_subject_split(
            features=features,
            labels=labels,
            subjects=subjects,
            test_subjects=test_subjects,
            feature_threshold=args.feature_threshold,
            late_epochs=args.late_epochs,
            late_lr=args.late_lr,
        )
        artifact_prefix = "step9"
        split_label = "Subject holdout"
        train_samples_display = str(int(y_train_reference.shape[0]))
        test_samples_display = str(int(y_true_reference.shape[0]))
        selected_rows: list[dict[str, float | str]] = []
    else:
        (
            comparison,
            fold_metrics,
            predictions,
            selected_by_modality,
            train_sizes,
            test_sizes,
        ) = _evaluate_kfold(
            features=features,
            labels=labels,
            n_splits=args.cv_folds,
            random_state=args.random_state,
            feature_threshold=args.feature_threshold,
            late_epochs=args.late_epochs,
            late_lr=args.late_lr,
        )
        artifact_prefix = "step9_kfold"
        split_label = f"Stratified {args.cv_folds}-fold cross-validation"
        y_true_reference = labels
        train_subjects = tuple(sorted(set(int(x) for x in subjects.tolist())))
        test_subjects_observed = train_subjects
        train_samples_display = f"{np.mean(train_sizes):.1f} avg"
        test_samples_display = f"{np.mean(test_sizes):.1f} avg"
        selected_rows = []

        fold_metrics.to_csv(tables_dir / f"{artifact_prefix}_fold_metrics.csv", index=False)
        for modality in MODALITY_ORDER:
            values = np.asarray(selected_by_modality[modality], dtype=np.float64)
            selected_rows.append(
                {
                    "modality": modality,
                    "mean_selected": float(values.mean()),
                    "std_selected": float(values.std(ddof=0)),
                    "min_selected": float(values.min()),
                    "max_selected": float(values.max()),
                }
            )
        pd.DataFrame(selected_rows).to_csv(tables_dir / f"{artifact_prefix}_selected_feature_counts.csv", index=False)

    ranking = comparison.sort_values(["accuracy", "f1_macro"], ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)

    comparison.to_csv(tables_dir / f"{artifact_prefix}_fusion_comparison.csv", index=False)
    ranking.to_csv(tables_dir / f"{artifact_prefix}_fusion_ranking.csv", index=False)

    _save_comparison_plot(comparison, plots_dir / f"{artifact_prefix}_fusion_accuracy_comparison.png")

    for method in METHOD_ORDER:
        method_slug = method.lower().replace(" ", "_")
        save_confusion_matrix_plot(
            y_true=y_true_reference,
            y_pred=predictions[method],
            output_png=plots_dir / f"{artifact_prefix}_{method_slug}_confusion_matrix.png",
            title=f"Step 9 - {method} ({args.eval_mode})",
        )

    print("=" * 64)
    print("Step 9 Completed: End-to-End Fusion Execution")
    print("=" * 64)
    print(f"Evaluation mode: {split_label}")
    print(f"Train samples  : {train_samples_display}")
    print(f"Test samples   : {test_samples_display}")
    print("\nFeature-level selected features:")
    if args.eval_mode == "kfold":
        for row in selected_rows:
            print(
                f"  - {row['modality']:8s}: mean={row['mean_selected']:.1f}, std={row['std_selected']:.1f}"
            )
    else:
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
    print(f"  - {tables_dir / f'{artifact_prefix}_fusion_comparison.csv'}")
    print(f"  - {tables_dir / f'{artifact_prefix}_fusion_ranking.csv'}")
    if args.eval_mode == "kfold":
        print(f"  - {tables_dir / f'{artifact_prefix}_fold_metrics.csv'}")
        print(f"  - {tables_dir / f'{artifact_prefix}_selected_feature_counts.csv'}")
    print(f"  - {plots_dir / f'{artifact_prefix}_fusion_accuracy_comparison.png'}")
    for method in METHOD_ORDER:
        method_slug = method.lower().replace(" ", "_")
        print(f"  - {plots_dir / f'{artifact_prefix}_{method_slug}_confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
