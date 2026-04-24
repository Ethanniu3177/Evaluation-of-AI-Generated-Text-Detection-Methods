"""
Output file with:
1) ROC curves
2) Bar chart of AUROC scores
3) Score distribution plots (histogram)
4) FPR@95TPR bar chart
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

results_dir = "test_result"
plots_dir = "plots"

# Human-readable labels for each experiment file
EXPERIMENT_LABELS = {
    "baseline_3": "Baseline 3\nWatermarked vs. Plain AI",
    "baseline_4": "Baseline 4\nWatermarked vs. Human",
    "baseline_5": "Baseline 5\nPlain AI vs. Human",
    "paraphrasing_2": "Paraphrasing 2\nParaphrased Watermarked vs. Plain AI",
    "paraphrasing_3": "Paraphrasing 3\nParaphrased Watermarked vs. Human",
}

SHORT_LABELS = {
    "baseline_3":     "B3: Wm vs. Plain AI",
    "baseline_4":     "B4: Wm vs. Human",
    "baseline_5":     "B5: Plain AI vs. Human",
    "paraphrasing_2": "P2: Para-Wm vs. Plain AI",
    "paraphrasing_3": "P3: Para-Wm vs. Human",
}

# Colour per experiment — consistent across all plots
COLORS = {
    "baseline_3":     "#2196F3",
    "baseline_4":     "#4CAF50",
    "baseline_5":     "#9E9E9E",
    "paraphrasing_2": "#FF9800",
    "paraphrasing_3": "#F44336",
}

# Display order
ORDER = ["baseline_3", "baseline_4", "baseline_5", "paraphrasing_2", "paraphrasing_3"]


def load_results(results_dir):
    data = {}
    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue
        key = filename.replace(".json", "")
        with open(os.path.join(results_dir, filename), "r") as f:
            data[key] = json.load(f)
    return data


def compute_metrics(entry):
    y_true = np.array(entry["y_true"])
    y_score = np.array(entry["y_score"])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    idx = np.searchsorted(tpr, 0.95)
    fpr_at_95 = float(fpr[min(idx, len(fpr) - 1)])
    return fpr, tpr, auroc, fpr_at_95


# ── 1. ROC Curves ────────────────────────────────────────────────────────────
def plot_roc_curves(data, plots_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    for key in ORDER:
        if key not in data:
            continue
        fpr, tpr, auroc, _ = compute_metrics(data[key])
        ax.plot(fpr, tpr, label=f"{SHORT_LABELS[key]} (AUC={auroc:.3f})",
                color=COLORS[key], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Watermark Detector", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 2. AUROC Bar Chart ────────────────────────────────────────────────────────
def plot_auroc_bar(data, plots_dir):
    keys = [k for k in ORDER if k in data]
    aurocs = []
    for key in keys:
        _, _, auroc, _ = compute_metrics(data[key])
        aurocs.append(auroc)

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, aurocs, color=[COLORS[k] for k in keys],
                  width=0.55, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[k] for k in keys], fontsize=9)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("AUROC by Experiment — Watermark Detector", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, "auroc_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 3. Score Distribution Histograms ─────────────────────────────────────────
def plot_score_distributions(data, plots_dir):
    keys = [k for k in ORDER if k in data]
    n = len(keys)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, key in enumerate(keys):
        ax = axes[i]
        entry = data[key]
        y_true = np.array(entry["y_true"])
        y_score = np.array(entry["y_score"])

        scores_neg = y_score[y_true == 0]
        scores_pos = y_score[y_true == 1]

        bins = np.linspace(y_score.min(), y_score.max(), 40)
        ax.hist(scores_neg, bins=bins, alpha=0.6, color="#9E9E9E", label="Negative class", density=True)
        ax.hist(scores_pos, bins=bins, alpha=0.6, color=COLORS[key], label="Positive class", density=True)

        _, _, auroc, _ = compute_metrics(entry)
        ax.set_title(f"{SHORT_LABELS[key]}\n(AUC={auroc:.3f})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Watermark Z-Score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

    fig.suptitle("Score Distributions — Watermark Detector", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(plots_dir, "score_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 4. FPR@95TPR Bar Chart ────────────────────────────────────────────────────
def plot_fpr_bar(data, plots_dir):
    keys = [k for k in ORDER if k in data]
    fprs = []
    for key in keys:
        _, _, _, fpr_at_95 = compute_metrics(data[key])
        fprs.append(fpr_at_95)

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, fprs, color=[COLORS[k] for k in keys],
                  width=0.55, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, fprs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0.05, color="green", linestyle="--", linewidth=1, label="5% FPR target")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[k] for k in keys], fontsize=9)
    ax.set_ylabel("FPR @ 95% TPR", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("FPR@95TPR by Experiment — Watermark Detector", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, "fpr_at_95tpr_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(plots_dir, exist_ok=True)
    data = load_results(results_dir)

    if not data:
        print(f"No JSON files found in '{results_dir}'")
        return

    print(f"Loaded experiments: {list(data.keys())}\n")

    plot_roc_curves(data, plots_dir)
    plot_auroc_bar(data, plots_dir)
    plot_score_distributions(data, plots_dir)
    plot_fpr_bar(data, plots_dir)

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()