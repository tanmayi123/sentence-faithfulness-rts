import pandas as pd
import numpy as np
import os

def compute_fg(topk_flip_rate, random_flip_rate):
    gap = topk_flip_rate - random_flip_rate
    return max(0.0, round(gap, 4))

def compute_confidence(fg, topk_flipped, random_flipped, cma):
    # C1 — stability: are flip counts similar?
    if max(topk_flipped, random_flipped) == 0:
        c1 = 1.0
    else:
        c1 = min(topk_flipped, random_flipped) / max(topk_flipped, random_flipped)

    # C2 — coverage: same as CMA (cross-model agreement)
    c2 = cma

    # C3 — decisiveness: how far is FG from 0.5?
    c3 = abs(fg - 0.5) * 2

    confidence = 0.30 * c1 + 0.40 * c2 + 0.30 * c3
    return round(confidence, 4)

def compute_rts(fg, lq, cma, confidence):
    rts = fg * lq * cma * confidence
    return round(rts, 4)

def run_rts_scoring():
    if os.path.exists("data/rts_results.csv"):
        os.remove("data/rts_results.csv")

    df = pd.read_csv("data/tagger_results.csv")

    # normalize FG to 0-1 across full dataset
    df["fg_raw"] = df["topk_flipped"] - df["random_flipped"]
    df["fg_raw"] = df["fg_raw"].clip(lower=0)
    max_fg = df["fg_raw"].max()
    df["fg"] = df["fg_raw"] / max_fg if max_fg > 0 else df["fg_raw"]

    records = []
    for i, row in df.iterrows():
        fg = row["fg"]
        lq = row["lq_score"]

        # CMA = 1.0 placeholder until Track B is built
        cma = 1.0

        confidence = compute_confidence(
            fg,
            row["topk_flipped"],
            row["random_flipped"],
            cma
        )

        rts = compute_rts(fg, lq, cma, confidence)

        records.append({
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "text": row["text"],
            "dominant_category": row["dominant_category"],
            "topk_flipped": row["topk_flipped"],
            "random_flipped": row["random_flipped"],
            "fg": fg,
            "lq": lq,
            "cma": cma,
            "confidence": confidence,
            "rts": rts
        })

    results_df = pd.DataFrame(records)
    results_df.to_csv("data/rts_results.csv", index=False)

    print("\n--- RTS Summary per Dataset ---")
    for dataset in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == dataset]
        print(f"\n  {dataset.upper()}")
        print(f"  Mean RTS:    {sub['rts'].mean():.4f}")
        print(f"  Max RTS:     {sub['rts'].max():.4f}")
        print(f"  Min RTS:     {sub['rts'].min():.4f}")
        print(f"  Mean FG:     {sub['fg'].mean():.4f}")
        print(f"  Mean LQ:     {sub['lq'].mean():.4f}")
        print(f"  Mean conf:   {sub['confidence'].mean():.4f}")

    print("\n--- Top 5 Most Faithful Sentences (highest RTS) ---")
    top5 = results_df.nlargest(5, "rts")[["dataset", "rts", "fg", "lq", "text"]]
    for _, row in top5.iterrows():
        print(f"\n  [{row['dataset']}] RTS={row['rts']:.4f} FG={row['fg']:.4f} LQ={row['lq']:.4f}")
        print(f"  {row['text'][:100]}")

    print("\n--- Bottom 5 Least Faithful Sentences (lowest RTS) ---")
    bot5 = results_df[results_df["rts"] > 0].nsmallest(5, "rts")[["dataset", "rts", "fg", "lq", "text"]]
    for _, row in bot5.iterrows():
        print(f"\n  [{row['dataset']}] RTS={row['rts']:.4f} FG={row['fg']:.4f} LQ={row['lq']:.4f}")
        print(f"  {row['text'][:100]}")

    print("\nSaved to data/rts_results.csv")
    return results_df

if __name__ == "__main__":
    run_rts_scoring()