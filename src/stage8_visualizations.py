import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

os.makedirs("results", exist_ok=True)

COLORS = {
    "sst2": "#4C72B0",
    "imdb": "#DD8452",
    "distilbert": "#7B3F9E",
    "gemini": "#C0392B",
    "low": "#E74C3C",
    "high": "#2ECC71"
}

def plot_vader_vs_llm():
    data = {
        "Dataset": ["SST-2", "IMDb", "SST-2", "IMDb"],
        "Model": ["VADER", "VADER", "DistilBERT", "DistilBERT"],
        "Accuracy": [0.802, 0.735, 0.985, 0.865]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35

    vader_vals = [0.802, 0.735]
    db_vals = [0.985, 0.865]

    bars1 = ax.bar(x - width/2, vader_vals, width, label="VADER",
                   color="#95A5A6", edgecolor="white")
    bars2 = ax.bar(x + width/2, db_vals, width, label="DistilBERT",
                   color=COLORS["distilbert"], edgecolor="white")

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Figure 1 — VADER vs DistilBERT Accuracy", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["SST-2", "IMDb"])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=10)

    ax.axhline(y=0.85, color="gray", linestyle="--", alpha=0.5, label="85% threshold")
    plt.tight_layout()
    plt.savefig("results/fig1_vader_vs_distilbert.png", dpi=150)
    plt.close()
    print("Saved fig1_vader_vs_distilbert.png")

def plot_probe_accuracy():
    probe_df = pd.read_csv("data/probe_results.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(probe_df["layer"], probe_df["accuracy"],
                  color=COLORS["distilbert"], edgecolor="white", alpha=0.85)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Accuracy", fontsize=12)
    ax.set_title("Figure 2 — Sentiment Probe Accuracy by DistilBERT Layer",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(probe_df["layer"])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=10)

    best_layer = probe_df.loc[probe_df["accuracy"].idxmax(), "layer"]
    ax.get_children()[best_layer - 1].set_color("#E67E22")
    ax.annotate("Peak", xy=(best_layer, probe_df["accuracy"].max()),
                xytext=(best_layer + 0.3, probe_df["accuracy"].max() + 0.05),
                fontsize=10, color="#E67E22",
                arrowprops=dict(arrowstyle="->", color="#E67E22"))

    plt.tight_layout()
    plt.savefig("results/fig2_probe_accuracy.png", dpi=150)
    plt.close()
    print("Saved fig2_probe_accuracy.png")

def plot_faithfulness_gap():
    data = {
        "Dataset": ["SST-2", "SST-2", "IMDb", "IMDb"],
        "Model": ["DistilBERT", "Gemini", "DistilBERT", "Gemini"],
        "Top-K Flip Rate": [0.295, 0.455, 0.105, 0.065],
        "Random Flip Rate": [0.045, 0.040, 0.055, 0.030],
        "Gap": [0.250, 0.415, 0.050, 0.035]
    }
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, dataset in enumerate(["SST-2", "IMDb"]):
        ax = axes[idx]
        sub = df[df["Dataset"] == dataset]
        x = np.arange(2)
        width = 0.3

        ax.bar(x - width/2, sub["Top-K Flip Rate"].values, width,
               label="Top-K flip rate", color=[COLORS["distilbert"], COLORS["gemini"]],
               edgecolor="white", alpha=0.9)
        ax.bar(x + width/2, sub["Random Flip Rate"].values, width,
               label="Random flip rate", color=["#BDC3C7", "#BDC3C7"],
               edgecolor="white", alpha=0.9)

        ax.set_title(f"{dataset}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["DistilBERT", "Gemini"])
        ax.set_ylim(0, 0.65)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        for i, (topk, rand, gap) in enumerate(zip(
            sub["Top-K Flip Rate"].values,
            sub["Random Flip Rate"].values,
            sub["Gap"].values
        )):
            ax.annotate(f"gap={gap:.3f}",
                       xy=(i, topk + 0.02),
                       ha="center", fontsize=9,
                       color=COLORS["distilbert"] if i == 0 else COLORS["gemini"])

    topk_patch = mpatches.Patch(color="#7B3F9E", label="Top-K flip rate")
    rand_patch = mpatches.Patch(color="#BDC3C7", label="Random flip rate")
    fig.legend(handles=[topk_patch, rand_patch], loc="upper right", fontsize=10)
    fig.suptitle("Figure 3 — Faithfulness Gap: Top-K vs Random Masking",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/fig3_faithfulness_gap.png", dpi=150)
    plt.close()
    print("Saved fig3_faithfulness_gap.png")

def plot_rts_distribution():
    df = pd.read_csv("data/rts_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, dataset in enumerate(["sst2", "imdb"]):
        ax = axes[idx]
        sub = df[df["dataset"] == dataset]

        ax.hist(sub["rts"], bins=30, color=COLORS[dataset],
                edgecolor="white", alpha=0.85)
        ax.axvline(sub["rts"].mean(), color="red", linestyle="--",
                   label=f"Mean={sub['rts'].mean():.4f}")
        ax.set_title(f"{dataset.upper()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("RTS Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)

    fig.suptitle("Figure 4 — RTS Score Distribution per Dataset",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/fig4_rts_distribution.png", dpi=150)
    plt.close()
    print("Saved fig4_rts_distribution.png")

def plot_failure_mode():
    df = pd.read_csv("data/analysis_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, dataset in enumerate(["sst2", "imdb"]):
        ax = axes[idx]
        sub = df[df["dataset"] == dataset]

        both = sub["both_failed"].sum()
        only_db = sub["only_db_failed"].sum()
        only_gm = sub["only_gm_failed"].sum()
        neither = len(sub) - both - only_db - only_gm

        sizes = [both, only_db, only_gm, neither]
        labels = [
            f"Both failed\n({both})",
            f"Only DistilBERT\n({only_db})",
            f"Only Gemini\n({only_gm})",
            f"Neither failed\n({neither})"
        ]
        colors = ["#E74C3C", COLORS["distilbert"], COLORS["gemini"], "#2ECC71"]
        explode = (0.05, 0.05, 0.05, 0.05)

        ax.pie(sizes, labels=labels, colors=colors, explode=explode,
               autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
        ax.set_title(f"{dataset.upper()}", fontsize=13, fontweight="bold")

    fig.suptitle("Figure 5 — Cross-Model Failure Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/fig5_failure_mode.png", dpi=150)
    plt.close()
    print("Saved fig5_failure_mode.png")

def plot_category_faithfulness():
    df = pd.read_csv("data/analysis_results.csv")

    cat_stats = df.groupby("dominant_category").agg(
        mean_fg=("fg", "mean"),
        mean_rts=("rts", "mean"),
        count=("rts", "count")
    ).sort_values("mean_fg", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#E74C3C" if v < 0.15 else "#F39C12" if v < 0.30
              else "#2ECC71" for v in cat_stats["mean_fg"]]

    bars = ax.barh(cat_stats.index, cat_stats["mean_fg"],
                   color=colors, edgecolor="white", alpha=0.9)

    for bar, count in zip(bars, cat_stats["count"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"n={count}", va="center", fontsize=9)

    ax.set_xlabel("Mean Faithfulness Gap (FG)", fontsize=12)
    ax.set_title("Figure 6 — Faithfulness Gap by Dominant Word Category",
                 fontsize=14, fontweight="bold")
    ax.axvline(0.20, color="gray", linestyle="--", alpha=0.5,
               label="0.20 moderate threshold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("results/fig6_category_faithfulness.png", dpi=150)
    plt.close()
    print("Saved fig6_category_faithfulness.png")

if __name__ == "__main__":
    print("Generating all figures...")
    plot_vader_vs_llm()
    plot_probe_accuracy()
    plot_faithfulness_gap()
    plot_rts_distribution()
    plot_failure_mode()
    plot_category_faithfulness()
    print("\nAll figures saved to results/")