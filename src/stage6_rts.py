import pandas as pd
import numpy as np
import ast
import os

def compute_fg(topk_flipped, random_flipped):
    gap = topk_flipped - random_flipped
    return max(0.0, round(gap, 4))

def compute_cma(distilbert_words, gemini_words):
    # deduplicate both lists
    db_words = set(w.lower().strip().replace("##", "") for w in distilbert_words)
    gm_words = set(w.lower().strip() for w in gemini_words)

    # remove empty strings
    db_words = {w for w in db_words if len(w) > 1}
    gm_words = {w for w in gm_words if len(w) > 1}

    if not db_words or not gm_words:
        return 0.0

    overlap = len(db_words & gm_words)
    union = len(db_words | gm_words)
    # jaccard similarity — overlap over union
    cma = overlap / union if union > 0 else 0.0
    return round(cma, 4)

def compute_confidence(fg, topk_flipped_db, topk_flipped_gm, cma):
    # C1 — stability: are both models flipping at similar rates?
    if max(topk_flipped_db, topk_flipped_gm) == 0:
        c1 = 1.0
    else:
        c1 = min(topk_flipped_db, topk_flipped_gm) / max(topk_flipped_db, topk_flipped_gm)

    # C2 — coverage: cross-model agreement
    c2 = cma

    # C3 — decisiveness: how far is FG from ambiguous midpoint
    c3 = abs(fg - 0.5) * 2

    confidence = 0.30 * c1 + 0.40 * c2 + 0.30 * c3
    return round(confidence, 4)

def compute_rts(fg, lq, cma, confidence):
    return round(fg * lq * cma * confidence, 4)

def run_rts_scoring():
    if os.path.exists("data/rts_results.csv"):
        os.remove("data/rts_results.csv")

    tagger_df = pd.read_csv("data/tagger_results.csv")
    gemini_df = pd.read_csv("data/gemini_predictions.csv")[
        ["sample_id", "stated_words", "topk_flipped", "random_flipped"]
    ].rename(columns={
        "stated_words": "gemini_words",
        "topk_flipped": "gemini_topk_flipped",
        "random_flipped": "gemini_random_flipped"
    })

    df = tagger_df.merge(gemini_df, on="sample_id")

    # normalize FG per model then average
    df["fg_db_raw"] = (df["topk_flipped"] - df["random_flipped"]).clip(lower=0)
    df["fg_gm_raw"] = (df["gemini_topk_flipped"] - df["gemini_random_flipped"]).clip(lower=0)

    max_db = df["fg_db_raw"].max()
    max_gm = df["fg_gm_raw"].max()

    df["fg_db"] = df["fg_db_raw"] / max_db if max_db > 0 else df["fg_db_raw"]
    df["fg_gm"] = df["fg_gm_raw"] / max_gm if max_gm > 0 else df["fg_gm_raw"]
    df["fg"] = ((df["fg_db"] + df["fg_gm"]) / 2).round(4)

    records = []
    for i, row in df.iterrows():
        db_words = ast.literal_eval(row["top_k_words"])
        gm_words = ast.literal_eval(row["gemini_words"])

        cma = compute_cma(db_words, gm_words)
        lq = row["lq_score"]
        fg = row["fg"]

        confidence = compute_confidence(
            fg,
            row["topk_flipped"],
            row["gemini_topk_flipped"],
            cma
        )

        rts = compute_rts(fg, lq, cma, confidence)

        records.append({
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "text": row["text"],
            "dominant_category": row["dominant_category"],
            "distilbert_words": row["top_k_words"],
            "gemini_words": row["gemini_words"],
            "fg_db": row["fg_db"],
            "fg_gm": row["fg_gm"],
            "fg": fg,
            "lq": lq,
            "cma": cma,
            "confidence": confidence,
            "rts": rts,
            "topk_flipped_db": row["topk_flipped"],
            "topk_flipped_gm": row["gemini_topk_flipped"],
        })

    results_df = pd.DataFrame(records)
    results_df.to_csv("data/rts_results.csv", index=False)

    print("\n=== RTS Summary per Dataset ===")
    for dataset in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == dataset]
        print(f"\n  {dataset.upper()}")
        print(f"  Mean RTS:    {sub['rts'].mean():.4f}")
        print(f"  Mean FG:     {sub['fg'].mean():.4f}")
        print(f"  Mean LQ:     {sub['lq'].mean():.4f}")
        print(f"  Mean CMA:    {sub['cma'].mean():.4f}")
        print(f"  Mean conf:   {sub['confidence'].mean():.4f}")

    print("\n=== DistilBERT vs Gemini Faithfulness Gap ===")
    for dataset in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == dataset]
        print(f"\n  {dataset.upper()}")
        print(f"  DistilBERT gap: {sub['fg_db'].mean():.4f}")
        print(f"  Gemini gap:     {sub['fg_gm'].mean():.4f}")

    print("\n=== Top 5 Most Faithful Sentences ===")
    top5 = results_df.nlargest(5, "rts")
    for _, row in top5.iterrows():
        print(f"\n  [{row['dataset']}] RTS={row['rts']:.4f} FG={row['fg']:.4f} CMA={row['cma']:.4f}")
        print(f"  DistilBERT: {row['distilbert_words']}")
        print(f"  Gemini:     {row['gemini_words']}")
        print(f"  Text: {row['text'][:100]}")

    print("\n=== Bottom 5 Least Faithful Sentences ===")
    bot5 = results_df[results_df["rts"] > 0].nsmallest(5, "rts")
    for _, row in bot5.iterrows():
        print(f"\n  [{row['dataset']}] RTS={row['rts']:.4f} FG={row['fg']:.4f} CMA={row['cma']:.4f}")
        print(f"  DistilBERT: {row['distilbert_words']}")
        print(f"  Gemini:     {row['gemini_words']}")
        print(f"  Text: {row['text'][:100]}")

    print("\nSaved to data/rts_results.csv")
    return results_df

if __name__ == "__main__":
    run_rts_scoring()