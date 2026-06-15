import pandas as pd
import numpy as np
import os

def run_failure_mode_analysis():
    if os.path.exists("data/analysis_results.csv"):
        os.remove("data/analysis_results.csv")

    df = pd.read_csv("data/rts_results.csv")

    # ── define low RTS threshold ──
    threshold = df["rts"].quantile(0.25)
    print(f"Low-RTS threshold (bottom 25%): {threshold:.4f}")

    low_rts = df[df["rts"] <= threshold].copy()
    high_rts = df[df["rts"] > threshold].copy()

    print(f"Low-RTS sentences:  {len(low_rts)} / {len(df)}")
    print(f"High-RTS sentences: {len(high_rts)} / {len(df)}")

    # ── FINDING 1: which dataset has more low-RTS ──
    print("\n=== Finding 1: Low-RTS distribution by dataset ===")
    for dataset in df["dataset"].unique():
        total = len(df[df["dataset"] == dataset])
        low = len(low_rts[low_rts["dataset"] == dataset])
        print(f"  {dataset}: {low}/{total} sentences are low-RTS ({low/total:.1%})")

    # ── FINDING 2: why — dominant category in low vs high RTS ──
    print("\n=== Finding 2: Dominant word category — low vs high RTS ===")
    print("\n  LOW-RTS sentences:")
    print(low_rts["dominant_category"].value_counts().to_string())
    print("\n  HIGH-RTS sentences:")
    print(high_rts["dominant_category"].value_counts().to_string())

    # ── FINDING 3: faithfulness gap by category ──
    print("\n=== Finding 3: Mean faithfulness gap by word category ===")
    cat_stats = df.groupby("dominant_category").agg(
        mean_fg=("fg", "mean"),
        mean_rts=("rts", "mean"),
        mean_cma=("cma", "mean"),
        count=("rts", "count")
    ).sort_values("mean_fg", ascending=False)
    print(cat_stats.round(4).to_string())

    # ── FINDING 4: did both models fail for same reason ──
    print("\n=== Finding 4: Cross-model failure analysis ===")
    df["db_failed"] = (df["fg_db"] < 0.1).astype(int)
    df["gm_failed"] = (df["fg_gm"] < 0.1).astype(int)
    df["both_failed"] = ((df["db_failed"] == 1) & (df["gm_failed"] == 1)).astype(int)
    df["only_db_failed"] = ((df["db_failed"] == 1) & (df["gm_failed"] == 0)).astype(int)
    df["only_gm_failed"] = ((df["db_failed"] == 0) & (df["gm_failed"] == 1)).astype(int)

    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        print(f"\n  {dataset.upper()}")
        print(f"  Both models failed:       {sub['both_failed'].sum()} ({sub['both_failed'].mean():.1%})")
        print(f"  Only DistilBERT failed:   {sub['only_db_failed'].sum()} ({sub['only_db_failed'].mean():.1%})")
        print(f"  Only Gemini failed:       {sub['only_gm_failed'].sum()} ({sub['only_gm_failed'].mean():.1%})")

    # ── FINDING 5: category breakdown for sentences where both models failed ──
    print("\n=== Finding 5: Why did both models fail? (word category breakdown) ===")
    both_failed = df[df["both_failed"] == 1]
    print(both_failed["dominant_category"].value_counts().to_string())

    # ── FINDING 6: examples — sentences both models failed on ──
    print("\n=== Finding 6: Example sentences both models failed on ===")
    examples = both_failed.nsmallest(5, "rts")[["dataset", "rts", "cma", "dominant_category", "distilbert_words", "gemini_words", "text"]]
    for _, row in examples.iterrows():
        print(f"\n  [{row['dataset']}] RTS={row['rts']:.4f} CMA={row['cma']:.4f} cat={row['dominant_category']}")
        print(f"  DistilBERT: {row['distilbert_words']}")
        print(f"  Gemini:     {row['gemini_words']}")
        print(f"  Text: {row['text'][:120]}")

    # ── save enriched results ──
    df.to_csv("data/analysis_results.csv", index=False)
    print("\nSaved to data/analysis_results.csv")
    return df

if __name__ == "__main__":
    run_failure_mode_analysis()