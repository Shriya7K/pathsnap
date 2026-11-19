"""Compile model comparison table and generate visual comparison plots."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = Path("results/models")
out_dir.mkdir(parents=True, exist_ok=True)


def build_model_comparison():
    cv_file = out_dir / "cv_results_summary.csv"
    test_file = out_dir / "student_test_results.csv"

    if not cv_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Missing results: {cv_file} or {test_file}")

    cv_df = pd.read_csv(cv_file)
    test_df = pd.read_csv(test_file)

    comp = cv_df.merge(test_df, on=["model", "family"])

    table = pd.DataFrame({
        "Model Name": comp["model"],
        "Family": comp["family"],
        "CV Top-1 Accuracy (%)": comp["cv_top1_accuracy_mean"],
        "CV Top-1 Std (%)": comp["cv_top1_accuracy_std"],
        "CV Macro F1 (%)": comp["cv_macro_f1_mean"],
        "CV Macro F1 Std (%)": comp["cv_macro_f1_std"],
        "CV Log Loss": comp["cv_log_loss_mean"],
        "Student-Test Top-1 Accuracy (%)": comp["test_top1_accuracy"],
        "Student-Test Macro F1 (%)": comp["test_macro_f1"],
        "Student-Test Log Loss": comp["test_log_loss"],
        "Avg Train Time (s)": comp["avg_train_time_sec"],
        "Test Infer Time (s)": comp["infer_time_sec"],
        "Key Hyperparameters": comp["hyperparameters"],
    })

    table = table.sort_values("CV Top-1 Accuracy (%)", ascending=False)
    table_csv = out_dir / "model_comparison.csv"
    table.to_csv(table_csv, index=False)
    print(f"Saved comparison table to {table_csv}")

    # --- Plot 1: Top-1 Candidate Accuracy Comparison ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(table))
    width = 0.35
    ax.bar([i - width/2 for i in x], table["CV Top-1 Accuracy (%)"], width=width, label="5-Fold CV Top-1", color="#386cb0", alpha=0.85)
    ax.bar([i + width/2 for i in x], table["Student-Test Top-1 Accuracy (%)"], width=width, label="Student-Test Top-1", color="#7fc97f", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(table["Model Name"], rotation=25, ha="right")
    ax.set_ylabel("Top-1 Candidate Accuracy (%)")
    ax.set_title("Candidate Ranking Performance across Model Families")
    ax.set_ylim(min(table["CV Top-1 Accuracy (%)"].min(), table["Student-Test Top-1 Accuracy (%)"].min()) - 5, 100)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison_top1.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Macro F1 Comparison ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - width/2 for i in x], table["CV Macro F1 (%)"], width=width, label="5-Fold CV Macro F1", color="#f0027f", alpha=0.85)
    ax.bar([i + width/2 for i in x], table["Student-Test Macro F1 (%)"], width=width, label="Student-Test Macro F1", color="#fdc086", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(table["Model Name"], rotation=25, ha="right")
    ax.set_ylabel("Macro F1-Score (%)")
    ax.set_title("Binary Candidate Classification Macro F1 Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison_macro_f1.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Training and Inference Latency ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(table["Model Name"], table["Avg Train Time (s)"], color="#beaed4", edgecolor="black")
    axes[0].set_ylabel("Avg Training Time per Fold (s)")
    axes[0].set_title("Training Latency")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_yscale("log")

    axes[1].bar(table["Model Name"], table["Test Infer Time (s)"], color="#ffff99", edgecolor="black")
    axes[1].set_ylabel("Student-Test Inference Time (s)")
    axes[1].set_title("Inference Latency (220k candidates)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_yscale("log")
    fig.suptitle("Model Computational Efficiency & Throughput", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "model_latency_comparison.png", dpi=150)
    plt.close(fig)

    print("Generated comparison plots:")
    print("  - results/models/model_comparison_top1.png")
    print("  - results/models/model_comparison_macro_f1.png")
    print("  - results/models/model_latency_comparison.png")
    return table


if __name__ == "__main__":
    build_model_comparison()
