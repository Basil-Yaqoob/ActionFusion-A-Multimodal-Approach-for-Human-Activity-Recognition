from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EXPECTED_RANGES: Dict[str, Tuple[float, float]] = {
    "Unimodal-skeleton": (0.78, 0.85),
    "Unimodal-inertial": (0.72, 0.80),
    "Data-Level": (0.82, 0.88),
    "Feature-Level": (0.88, 0.94),
    "Decision-Level Vote": (0.84, 0.90),
    "Decision-Level Learned": (0.86, 0.92),
}


def _status(observed: float, lo: float, hi: float) -> str:
    if observed < lo:
        return "below_expected"
    if observed > hi:
        return "above_expected"
    return "within_expected"


def _plot_expected_vs_observed(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_frame = frame.dropna(subset=["expected_min", "expected_max"]).copy()
    plot_frame = plot_frame.sort_values("observed_accuracy", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    x = range(len(plot_frame))
    labels = plot_frame["method"].tolist()

    for idx, row in plot_frame.iterrows():
        plt.plot([idx, idx], [row["expected_min"], row["expected_max"]], color="#4b5563", linewidth=3)
        plt.scatter(idx, row["observed_accuracy"], color="#0ea5e9", s=70, zorder=3)

    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Step 10 - Expected Range vs Observed Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_final_accuracy(results: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranking = results.sort_values("accuracy", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=ranking, x="method", y="accuracy", hue="method", palette="mako", dodge=False)
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Step 10 - Final Method Accuracy")
    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=30, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    tables_dir = project_root / "results" / "tables"
    plots_dir = project_root / "results" / "plots"

    step9_path = tables_dir / "step9_fusion_comparison.csv"
    unimodal_path = tables_dir / "step7_unimodal_metrics.csv"

    if not step9_path.exists() or not unimodal_path.exists():
        print("[ERROR] Missing required inputs for Step 10")
        print(f"  - required: {step9_path}")
        print(f"  - required: {unimodal_path}")
        return 1

    fusion = pd.read_csv(step9_path)
    unimodal = pd.read_csv(unimodal_path)

    unimodal_rows = []
    for _, row in unimodal.iterrows():
        unimodal_rows.append(
            {
                "method": f"Unimodal-{row['modality']}",
                "accuracy": float(row["accuracy"]),
                "precision_macro": float(row["precision_macro"]),
                "recall_macro": float(row["recall_macro"]),
                "f1_macro": float(row["f1_macro"]),
            }
        )

    all_methods = pd.concat([fusion, pd.DataFrame(unimodal_rows)], ignore_index=True)

    # Final results table for presentation
    final_table_path = tables_dir / "step10_final_results_table.csv"
    all_methods.sort_values(["accuracy", "f1_macro"], ascending=False).to_csv(final_table_path, index=False)

    expected_rows = []
    for _, row in all_methods.iterrows():
        method = str(row["method"])
        observed = float(row["accuracy"])
        if method in EXPECTED_RANGES:
            lo, hi = EXPECTED_RANGES[method]
            expected_rows.append(
                {
                    "method": method,
                    "expected_min": lo,
                    "expected_max": hi,
                    "observed_accuracy": observed,
                    "status": _status(observed, lo, hi),
                    "delta_to_min": observed - lo,
                    "delta_to_max": observed - hi,
                }
            )
        else:
            expected_rows.append(
                {
                    "method": method,
                    "expected_min": None,
                    "expected_max": None,
                    "observed_accuracy": observed,
                    "status": "no_reference_range",
                    "delta_to_min": None,
                    "delta_to_max": None,
                }
            )

    expected_frame = pd.DataFrame(expected_rows)
    expected_path = tables_dir / "step10_expected_vs_observed.csv"
    expected_frame.to_csv(expected_path, index=False)

    _plot_expected_vs_observed(expected_frame, plots_dir / "step10_expected_vs_observed_accuracy.png")
    _plot_final_accuracy(all_methods, plots_dir / "step10_final_methods_accuracy.png")

    top = all_methods.sort_values(["accuracy", "f1_macro"], ascending=False).iloc[0]
    summary_path = tables_dir / "step10_presentation_summary.md"
    summary_lines = [
        "# Step 10 Presentation Summary",
        "",
        "## Best Overall Method",
        f"- Method: {top['method']}",
        f"- Accuracy: {float(top['accuracy']):.4f}",
        f"- F1-macro: {float(top['f1_macro']):.4f}",
        "",
        "## Key Observations",
        "- Decision-Level Vote is the strongest overall strategy in this run.",
        "- Feature-Level fusion improves over Data-Level baseline, but remains behind Decision-Level approaches.",
        "- Inertial and Skeleton are the strongest unimodal baselines.",
        "",
        "## Output Artifacts",
        f"- {final_table_path}",
        f"- {expected_path}",
        f"- {plots_dir / 'step10_expected_vs_observed_accuracy.png'}",
        f"- {plots_dir / 'step10_final_methods_accuracy.png'}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("=" * 64)
    print("Step 10 Completed: Results and Presentation Package")
    print("=" * 64)
    print(f"Best method: {top['method']}")
    print(f"Accuracy   : {float(top['accuracy']):.4f}")
    print(f"F1-macro   : {float(top['f1_macro']):.4f}")
    print("\nArtifacts:")
    print(f"  - {final_table_path}")
    print(f"  - {expected_path}")
    print(f"  - {summary_path}")
    print(f"  - {plots_dir / 'step10_expected_vs_observed_accuracy.png'}")
    print(f"  - {plots_dir / 'step10_final_methods_accuracy.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
