import torch
import pandas as pd
import numpy as np
import ast
import random
import re
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
TOP_K = 3

# ── output files (separate from originals) ────────────────────────────────
OUT_MASKING  = "data/improved_masking_results.csv"
OUT_TAGGER   = "data/improved_tagger_results.csv"
OUT_RTS      = "data/improved_rts_results.csv"
OUT_ANALYSIS = "data/improved_analysis_results.csv"

# ── word category config ───────────────────────────────────────────────────
CATEGORY_MULTIPLIERS = {
    "sentiment": 1.0, "intensifier": 0.8, "negation": 0.7,
    "other": 0.5, "positional": 0.3
}
NEGATIONS = {
    "not","never","no","neither","nor","barely","hardly","scarcely",
    "without","nobody","nothing","nowhere","none","isn","wasn","doesn",
    "didn","won","wouldn","couldn","shouldn","don","cant","cannot","nt"
}
INTENSIFIERS = {
    "very","absolutely","extremely","really","quite","incredibly",
    "utterly","deeply","highly","terribly","completely","totally",
    "entirely","awfully","remarkably","so","too","such","truly",
    "genuinely","pretty","rather","exceptionally","overwhelmingly"
}
SENTIMENT_WORDS = {
    "good","great","excellent","amazing","wonderful","fantastic","brilliant",
    "superb","outstanding","perfect","beautiful","loved","best","masterpiece",
    "gem","enjoy","enjoyed","love","entertaining","engrossing","compelling",
    "moving","hilarious","funny","clever","refreshing","impressive","delightful",
    "charming","captivating","thrilling","powerful","bad","terrible","awful",
    "horrible","dreadful","boring","worst","disappointing","waste","poor",
    "mediocre","stupid","ridiculous","garbage","trash","disaster","mess","dull",
    "flat","weak","pointless","predictable","forgettable","annoying","painful",
    "unbearable","offensive","tedious","pretentious","pathetic","atrocious",
    "abysmal","deplorable","unwatchable"
}

# ── MODEL LOAD ─────────────────────────────────────────────────────────────
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=-1).item()

# ── STAGE 4: IMPROVED MASKING ─────────────────────────────────────────────
def mask_words(text, words_to_mask):
    """
    Improvement: whole-word masking.
    When a subword fragment (##xxx) appears in top-K, the entire
    root word is masked rather than just the piece.
    This prevents the case where ##ing is masked but 'entertain'
    remains, leaving the word functionally intact in the sentence.
    """
    masked = text
    for word in words_to_mask:
        clean = word.replace("##", "").strip()
        if not clean:
            continue
        if word.startswith("##"):
            masked = re.sub(
                r'\b\w*' + re.escape(clean) + r'\b',
                '[MASK]', masked, flags=re.IGNORECASE
            )
        else:
            masked = re.sub(
                r'\b' + re.escape(clean) + r'\b',
                '[MASK]', masked, flags=re.IGNORECASE
            )
    return masked

def get_random_words(text, tokenizer, k, exclude_words):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens = [t for t in tokens if t not in ["[CLS]","[SEP]","[PAD]"]
              and t not in exclude_words]
    if len(tokens) < k:
        return tokens
    return random.sample(tokens, k)

def run_stage4(tokenizer, model):
    print("\n--- Stage 4: Improved Masking ---")
    df = pd.read_csv("data/distilbert_predictions.csv")
    random.seed(42)
    records = []

    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  {i}/{len(df)}...")

        text         = row["text"]
        original     = row["pred_label"]
        top_k_words  = ast.literal_eval(row["top_k_words"])

        masked_topk  = mask_words(text, top_k_words)
        pred_topk    = predict(masked_topk, tokenizer, model)
        topk_flipped = int(pred_topk != original)

        random_words  = get_random_words(text, tokenizer, TOP_K, top_k_words)
        masked_random = mask_words(text, random_words)
        pred_random   = predict(masked_random, tokenizer, model)
        rand_flipped  = int(pred_random != original)

        records.append({
            "sample_id":              row["sample_id"],
            "dataset":                row["dataset"],
            "text":                   text,
            "true_label":             row["true_label"],
            "pred_label":             original,
            "correct":                row["correct"],
            "top_k_words":            row["top_k_words"],
            "top_k_scores":           row["top_k_scores"],
            "masked_topk_text":       masked_topk,
            "pred_after_topk_mask":   pred_topk,
            "topk_flipped":           topk_flipped,
            "random_words":           str(random_words),
            "masked_random_text":     masked_random,
            "pred_after_random_mask": pred_random,
            "random_flipped":         rand_flipped,
        })

    out = pd.DataFrame(records)
    out.to_csv(OUT_MASKING, index=False)

    print("\n  Faithfulness Gap: Improved vs Original")
    orig = pd.read_csv("data/masking_results.csv")
    for dataset in out["dataset"].unique():
        new_sub  = out[out["dataset"] == dataset]
        orig_sub = orig[orig["dataset"] == dataset]
        new_gap  = new_sub["topk_flipped"].mean() - new_sub["random_flipped"].mean()
        orig_gap = orig_sub["topk_flipped"].mean() - orig_sub["random_flipped"].mean()
        print(f"  {dataset.upper()}: original={orig_gap:.3f}  improved={new_gap:.3f}  delta={new_gap-orig_gap:+.3f}")

    return out

# ── STAGE 5: TAGGER ───────────────────────────────────────────────────────
def clean_token(word):
    return word.replace("##", "").lower().strip(".,!?'\"")

def categorize_word(word, position, total_tokens):
    clean = clean_token(word)
    if position == 0 or position == total_tokens - 1:
        return "positional"
    if clean in NEGATIONS:
        return "negation"
    if clean in INTENSIFIERS:
        return "intensifier"
    if any(s.startswith(clean) or clean.startswith(s)
           for s in SENTIMENT_WORDS if len(clean) > 3):
        return "sentiment"
    return "other"

def tag_words(top_k_words, top_k_scores, text):
    all_tokens = text.lower().split()
    total = len(all_tokens)
    tagged = []
    for word, score in zip(top_k_words, top_k_scores):
        clean = clean_token(word)
        try:
            position = all_tokens.index(clean)
        except ValueError:
            position = next(
                (i for i, t in enumerate(all_tokens) if clean in t), 1
            )
        category = categorize_word(word, position, total)
        tagged.append({
            "word": word, "attention_score": score,
            "category": category,
            "multiplier": CATEGORY_MULTIPLIERS[category]
        })
    return tagged

def compute_lq(tagged):
    total = sum(w["attention_score"] for w in tagged)
    if total == 0:
        return 0.5
    return round(sum(w["attention_score"] * w["multiplier"] for w in tagged) / total, 4)

def run_stage5(masking_df):
    print("\n--- Stage 5: Tagger ---")
    df = masking_df.copy()
    records = []

    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  {i}/{len(df)}...")

        top_k_words  = ast.literal_eval(row["top_k_words"])
        top_k_scores = ast.literal_eval(row["top_k_scores"])

        tagged = tag_words(top_k_words, top_k_scores, row["text"])
        if not tagged:
            tagged = [{"word":"unknown","attention_score":0.0,
                       "category":"other","multiplier":0.5}]
        lq = compute_lq(tagged)
        dominant = max(tagged, key=lambda x: x["attention_score"])

        records.append({
            "sample_id":         row["sample_id"],
            "dataset":           row["dataset"],
            "text":              row["text"],
            "top_k_words":       row["top_k_words"],
            "tagged_words":      str([(w["word"], w["category"]) for w in tagged]),
            "dominant_category": dominant["category"],
            "lq_score":          lq,
            "topk_flipped":      row["topk_flipped"],
            "random_flipped":    row["random_flipped"],
        })

    out = pd.DataFrame(records)
    out.to_csv(OUT_TAGGER, index=False)
    print("  Saved:", OUT_TAGGER)
    return out

# ── STAGE 6: RTS SCORING ──────────────────────────────────────────────────
def compute_cma(distilbert_words, gemini_words):
    db = {w.lower().strip().replace("##","") for w in distilbert_words if len(w.replace("##","")) > 1}
    gm = {w.lower().strip() for w in gemini_words if len(w) > 1}
    if not db or not gm:
        return 0.0
    return round(len(db & gm) / len(db | gm), 4)

def compute_confidence(fg, topk_db, topk_gm, cma):
    c1 = min(topk_db, topk_gm) / max(topk_db, topk_gm) if max(topk_db, topk_gm) > 0 else 1.0
    c2 = cma
    c3 = abs(fg - 0.5) * 2
    return round(0.30*c1 + 0.40*c2 + 0.30*c3, 4)

def run_stage6(tagger_df):
    print("\n--- Stage 6: RTS Scoring ---")
    gemini_df = pd.read_csv("data/gemini_predictions.csv")[
        ["sample_id","stated_words","topk_flipped","random_flipped"]
    ].rename(columns={
        "stated_words":   "gemini_words",
        "topk_flipped":   "gemini_topk_flipped",
        "random_flipped": "gemini_random_flipped"
    })
    df = tagger_df.merge(gemini_df, on="sample_id")

    df["fg_db_raw"] = (df["topk_flipped"]        - df["random_flipped"]).clip(lower=0)
    df["fg_gm_raw"] = (df["gemini_topk_flipped"] - df["gemini_random_flipped"]).clip(lower=0)
    max_db = df["fg_db_raw"].max()
    max_gm = df["fg_gm_raw"].max()
    df["fg_db"] = df["fg_db_raw"] / max_db if max_db > 0 else df["fg_db_raw"]
    df["fg_gm"] = df["fg_gm_raw"] / max_gm if max_gm > 0 else df["fg_gm_raw"]
    df["fg"]    = ((df["fg_db"] + df["fg_gm"]) / 2).round(4)

    records = []
    for _, row in df.iterrows():
        db_words = ast.literal_eval(row["top_k_words"])
        gm_words = ast.literal_eval(row["gemini_words"])
        cma  = compute_cma(db_words, gm_words)
        fg   = row["fg"]
        lq   = row["lq_score"]
        conf = compute_confidence(fg, row["topk_flipped"], row["gemini_topk_flipped"], cma)
        rts  = round(fg * lq * cma * conf, 4)
        records.append({
            "sample_id":       row["sample_id"],
            "dataset":         row["dataset"],
            "text":            row["text"],
            "dominant_category": row["dominant_category"],
            "distilbert_words":  row["top_k_words"],
            "gemini_words":      row["gemini_words"],
            "fg_db":  row["fg_db"],
            "fg_gm":  row["fg_gm"],
            "fg":     fg,
            "lq":     lq,
            "cma":    cma,
            "confidence": conf,
            "rts":    rts,
            "topk_flipped_db": row["topk_flipped"],
            "topk_flipped_gm": row["gemini_topk_flipped"],
        })

    out = pd.DataFrame(records)
    out.to_csv(OUT_RTS, index=False)

    print("\n  RTS Summary: Improved vs Original")
    orig = pd.read_csv("data/rts_results.csv")
    for dataset in out["dataset"].unique():
        new_mean  = out[out["dataset"]==dataset]["rts"].mean()
        orig_mean = orig[orig["dataset"]==dataset]["rts"].mean()
        print(f"  {dataset.upper()}: original={orig_mean:.4f}  improved={new_mean:.4f}  delta={new_mean-orig_mean:+.4f}")

    return out

# ── STAGE 7: FAILURE MODE ANALYSIS ───────────────────────────────────────
def run_stage7(rts_df):
    print("\n--- Stage 7: Failure Mode Analysis ---")
    df = rts_df.copy()
    df["db_failed"]     = (df["fg_db"] < 0.1).astype(int)
    df["gm_failed"]     = (df["fg_gm"] < 0.1).astype(int)
    df["both_failed"]   = ((df["db_failed"]==1) & (df["gm_failed"]==1)).astype(int)
    df["only_db_failed"]= ((df["db_failed"]==1) & (df["gm_failed"]==0)).astype(int)
    df["only_gm_failed"]= ((df["db_failed"]==0) & (df["gm_failed"]==1)).astype(int)

    print("\n  Cross-Model Failure: Improved vs Original")
    orig = pd.read_csv("data/analysis_results.csv")
    for dataset in df["dataset"].unique():
        new_sub  = df[df["dataset"]==dataset]
        orig_sub = orig[orig["dataset"]==dataset]
        print(f"\n  {dataset.upper()}")
        print(f"  Both failed:     orig={orig_sub['both_failed'].mean():.1%}  new={new_sub['both_failed'].mean():.1%}")
        print(f"  Only DistilBERT: orig={orig_sub['only_db_failed'].mean():.1%}  new={new_sub['only_db_failed'].mean():.1%}")
        print(f"  Only Gemini:     orig={orig_sub['only_gm_failed'].mean():.1%}  new={new_sub['only_gm_failed'].mean():.1%}")

    df.to_csv(OUT_ANALYSIS, index=False)
    print("\n  Saved:", OUT_ANALYSIS)
    return df

# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model...")
    tokenizer, model = load_model()

    masking_df  = run_stage4(tokenizer, model)
    tagger_df   = run_stage5(masking_df)
    rts_df      = run_stage6(tagger_df)
    analysis_df = run_stage7(rts_df)

    print("\nAll improved outputs saved with prefix: improved_")