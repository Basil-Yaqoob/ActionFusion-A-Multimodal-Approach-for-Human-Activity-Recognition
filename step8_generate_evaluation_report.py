from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CORE_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def _read_metrics_row(csv_path: Path, method_name: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    row = {"method": method_name}
    for metric in CORE_METRICS:
        row[metric] = float(frame.loc[0, metric])
    return pd.DataFrame([row])


def _read_unimodal_rows(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    rows = []
    for _, data in frame.iterrows():
        rows.append(
            {
                "method": f"Unimodal-{data['modality']}",
                "accuracy": float(data["accuracy"]),
                "precision_macro": float(data["precision_macro"]),
                "recall_macro": float(data["recall_macro"]),
                "f1_macro": float(data["f1_macro"]),
            }
        )
    return pd.DataFrame(rows)


def _save_metric_bar_plot(frame: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(metric, ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=ordered, x="method", y=metric, hue="method", palette="viridis", dodge=False)
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=35, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    tables_dir = project_root / "results" / "tables"
    plots_dir = project_root / "results" / "plots"

    required = {
        "step5_data_level": tables_dir / "step5_data_level_metrics.csv",
        "step6_feature": tables_dir / "step6_feature_level_metrics.csv",
        "step7_vote": tables_dir / "step7_decision_level_vote_metrics.csv",
        "step7_learned": tables_dir / "step7_decision_level_learned_metrics.csv",
        "step7_unimodal": tables_dir / "step7_unimodal_metrics.csv",
    }

    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        print("[ERROR] Missing required metrics files:")
        for item in missing:
            print(f"  - {item}")
        return 1

    fusion_rows = pd.concat(
        [
            _read_metrics_row(required["step5_data_level"], "Data-Level"),
            _read_metrics_row(required["step6_feature"], "Feature-Level"),
            _read_metrics_row(required["step7_vote"], "Decision-Level Vote"),
            _read_metrics_row(required["step7_learned"], "Decision-Level Learned"),
        ],
        ignore_index=True,
    )

    unimodal_rows = _read_unimodal_rows(required["step7_unimodal"])
    all_rows = pd.concat([fusion_rows, unimodal_rows], ignore_index=True)

    fusion_out = tables_dir / "step8_fusion_comparison.csv"
    all_out = tables_dir / "step8_all_methods_comparison.csv"
    rank_out = tables_dir / "step8_method_ranking.csv"

    fusion_rows.to_csv(fusion_out, index=False)
    all_rows.to_csv(all_out, index=False)

    ranking = all_rows.sort_values(["accuracy", "f1_macro"], ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    ranking.to_csv(rank_out, index=False)

    _save_metric_bar_plot(
        frame=fusion_rows,
        metric="accuracy",
        output_path=plots_dir / "step8_fusion_accuracy_comparison.png",
        title="Step 8 - Fusion Method Accuracy Comparison",
    )
    _save_metric_bar_plot(
        frame=fusion_rows,
        metric="f1_macro",
        output_path=plots_dir / "step8_fusion_f1_comparison.png",
        title="Step 8 - Fusion Method F1-Macro Comparison",
    )
    _save_metric_bar_plot(
        frame=all_rows,
        metric="accuracy",
        output_path=plots_dir / "step8_all_methods_accuracy_comparison.png",
        title="Step 8 - All Methods Accuracy Comparison",
    )

    best = ranking.iloc[0]
    print("=" * 64)
    print("Step 8 Completed: Evaluation Report")
    print("=" * 64)
    print(f"Best method: {best['method']}")
    print(f"Best accuracy: {best['accuracy']:.4f}")
    print(f"Best f1_macro: {best['f1_macro']:.4f}")
    print("\nArtifacts:")
    print(f"  - {fusion_out}")
    print(f"  - {all_out}")
    print(f"  - {rank_out}")
    print(f"  - {plots_dir / 'step8_fusion_accuracy_comparison.png'}")
    print(f"  - {plots_dir / 'step8_fusion_f1_comparison.png'}")
    print(f"  - {plots_dir / 'step8_all_methods_accuracy_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
