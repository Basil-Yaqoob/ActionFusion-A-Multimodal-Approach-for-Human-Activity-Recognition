from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from models.decision_level_fusion import (
    decision_level_fusion_learned,
    decision_level_majority_vote,
    train_unimodal_classifiers,
)
from step5_run_data_level_fusion import _load_processed_features
from utils.evaluation import compute_metrics, save_confusion_matrix_plot, save_metrics_table
from utils.splits import get_subject_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 7: Decision-level fusion run")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-subjects", type=str, default="1,3,5,7")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    test_subjects = [int(x.strip()) for x in args.test_subjects.split(",") if x.strip()]

    features, labels, subjects = _load_processed_features(args.processed_dir)
    split = get_subject_split(subjects=subjects, test_subjects=test_subjects)

    modality_order = ("skeleton", "inertial", "depth", "rgb")

    score_train, score_test, unimodal_pred, _models = train_unimodal_classifiers(
        features=features,
        labels=labels,
        train_idx=split.train_idx,
        test_idx=split.test_idx,
        modality_order=modality_order,
    )

    y_test = labels[split.test_idx]
    y_train = labels[split.train_idx]

    unimodal_rows = []
    for modality in modality_order:
        metrics = compute_metrics(y_true=y_test, y_pred=unimodal_pred[modality])
        metrics["modality"] = modality
        unimodal_rows.append(metrics)

    y_true_vote, y_pred_vote = decision_level_majority_vote(
        score_test=score_test,
        labels_test=y_test,
        modality_order=modality_order,
    )
    vote_metrics = compute_metrics(y_true=y_true_vote, y_pred=y_pred_vote)

    y_true_learned, y_pred_learned, _fusion_model = decision_level_fusion_learned(
        score_train=score_train,
        score_test=score_test,
        labels_train=y_train,
        labels_test=y_test,
        modality_order=modality_order,
        epochs=args.epochs,
        lr=args.learning_rate,
    )
    learned_metrics = compute_metrics(y_true=y_true_learned, y_pred=y_pred_learned)

    tables_dir = args.results_dir / "tables"
    plots_dir = args.results_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(unimodal_rows).to_csv(tables_dir / "step7_unimodal_metrics.csv", index=False)
    save_metrics_table(vote_metrics, tables_dir / "step7_decision_level_vote_metrics.csv")
    save_metrics_table(learned_metrics, tables_dir / "step7_decision_level_learned_metrics.csv")

    save_confusion_matrix_plot(
        y_true=y_true_vote,
        y_pred=y_pred_vote,
        output_png=plots_dir / "step7_decision_level_vote_confusion_matrix.png",
        title="Step 7 - Decision-Level Fusion (Vote)",
    )
    save_confusion_matrix_plot(
        y_true=y_true_learned,
        y_pred=y_pred_learned,
        output_png=plots_dir / "step7_decision_level_learned_confusion_matrix.png",
        title="Step 7 - Decision-Level Fusion (Learned)",
    )

    print("=" * 64)
    print("Step 7 Completed: Decision-Level Fusion")
    print("=" * 64)
    print(f"Train samples: {split.train_idx.shape[0]}")
    print(f"Test samples : {split.test_idx.shape[0]}")

    print("\nUnimodal metrics:")
    for row in unimodal_rows:
        print(
            f"  - {row['modality']:8s} | "
            f"acc={row['accuracy']:.4f}, "
            f"f1={row['f1_macro']:.4f}"
        )

    print("\nDecision-level fusion metrics:")
    print(
        f"  - vote    | acc={vote_metrics['accuracy']:.4f}, "
        f"f1={vote_metrics['f1_macro']:.4f}"
    )
    print(
        f"  - learned | acc={learned_metrics['accuracy']:.4f}, "
        f"f1={learned_metrics['f1_macro']:.4f}"
    )

    print("\nArtifacts:")
    print(f"  - {tables_dir / 'step7_unimodal_metrics.csv'}")
    print(f"  - {tables_dir / 'step7_decision_level_vote_metrics.csv'}")
    print(f"  - {tables_dir / 'step7_decision_level_learned_metrics.csv'}")
    print(f"  - {plots_dir / 'step7_decision_level_vote_confusion_matrix.png'}")
    print(f"  - {plots_dir / 'step7_decision_level_learned_confusion_matrix.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
