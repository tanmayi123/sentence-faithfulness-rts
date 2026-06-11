import pandas as pd
import ast

CATEGORY_MULTIPLIERS = {
    "sentiment": 1.0,
    "intensifier": 0.8,
    "negation": 0.7,
    "positional": 0.3,
    "other": 0.5
}

NEGATIONS = {
    "not", "never", "no", "neither", "nor", "barely", "hardly", "scarcely",
    "without", "nobody", "nothing", "nowhere", "none", "isn", "wasn",
    "doesn", "didn", "won", "wouldn", "couldn", "shouldn", "don", "cant",
    "cannot", "nt"
}

INTENSIFIERS = {
    "very", "absolutely", "extremely", "really", "quite", "incredibly",
    "utterly", "deeply", "highly", "terribly", "completely", "totally",
    "entirely", "awfully", "remarkably", "so", "too", "such", "truly",
    "genuinely", "pretty", "rather", "exceptionally", "overwhelmingly"
}

SENTIMENT_WORDS = {
    # positive
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "brilliant", "superb", "outstanding", "perfect", "beautiful", "loved",
    "best", "masterpiece", "gem", "enjoy", "enjoyed", "love", "loved",
    "entertaining", "engrossing", "compelling", "moving", "touching",
    "hilarious", "funny", "clever", "smart", "refreshing", "impressive",
    "delightful", "charming", "captivating", "thrilling", "powerful",
    # negative
    "bad", "terrible", "awful", "horrible", "dreadful", "boring", "worst",
    "disappointing", "waste", "poor", "mediocre", "stupid", "ridiculous",
    "garbage", "trash", "disaster", "mess", "dull", "flat", "weak",
    "pointless", "predictable", "forgettable", "annoying", "painful",
    "unbearable", "offensive", "tedious", "pretentious", "laughable",
    "pathetic", "atrocious", "abysmal", "deplorable", "unwatchable"
}

def categorize_word(word, position, total_tokens):
    clean = word.replace("##", "").lower().strip(".,!?")

    if position == 0 or position == total_tokens - 1:
        return "positional"
    if clean in NEGATIONS:
        return "negation"
    if clean in INTENSIFIERS:
        return "intensifier"
    if clean in SENTIMENT_WORDS:
        return "sentiment"
    return "other"

def clean_token(word):
    return word.replace("##", "").lower().strip(".,!?'\"")

def tag_words(top_k_words, top_k_scores, text):
    all_tokens = text.lower().split()
    total = len(all_tokens)

    tagged = []
    for word, score in zip(top_k_words, top_k_scores):
        clean = clean_token(word)

        # try to find position in original text
        try:
            position = all_tokens.index(clean)
        except ValueError:
            # subword piece — find the word it belongs to
            position = next(
                (i for i, t in enumerate(all_tokens) if clean in t),
                1  # default to middle
            )

        # for subword pieces, also check the full word it came from
        full_word = next(
            (t for t in all_tokens if clean in t.lower()),
            clean
        )

        # categorize using both the piece and the full word
        category = "other"
        for candidate in [clean, full_word]:
            if position == 0 or position == total - 1:
                category = "positional"
                break
            if candidate in NEGATIONS:
                category = "negation"
                break
            if candidate in INTENSIFIERS:
                category = "intensifier"
                break
            if candidate in SENTIMENT_WORDS:
                category = "sentiment"
                break
            # partial match — e.g. "terri" matches "terrible"
            if any(s.startswith(candidate) or candidate.startswith(s)
                   for s in SENTIMENT_WORDS if len(candidate) > 3):
                category = "sentiment"
                break

        tagged.append({
            "word": word,
            "attention_score": score,
            "category": category,
            "multiplier": CATEGORY_MULTIPLIERS[category]
        })

    return tagged

def compute_lq(tagged_words):
    total_attention = sum(w["attention_score"] for w in tagged_words)
    if total_attention == 0:
        return 0.5
    lq = sum(w["attention_score"] * w["multiplier"] for w in tagged_words) / total_attention
    return round(lq, 4)

import os
if os.path.exists("data/tagger_results.csv"):
    os.remove("data/tagger_results.csv")

def run_tagger():
    df = pd.read_csv("data/masking_results.csv")
    scores_df = pd.read_csv("data/distilbert_predictions.csv")[["sample_id", "top_k_scores"]]
    df = df.merge(scores_df, on="sample_id")

    records = []
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Tagging {i}/{len(df)}...")

        top_k_words = ast.literal_eval(row["top_k_words"])
        top_k_scores = ast.literal_eval(row["top_k_scores"])

        tagged = tag_words(top_k_words, top_k_scores, row["text"])
        lq = compute_lq(tagged)
        if not tagged:
            tagged = [{"word": "unknown", "attention_score": 0.0,
                      "category": "other", "multiplier": 0.5}]
        dominant = max(tagged, key=lambda x: x["attention_score"])

        records.append({
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "text": row["text"],
            "top_k_words": row["top_k_words"],
            "tagged_words": str([(w["word"], w["category"]) for w in tagged]),
            "dominant_category": dominant["category"],
            "lq_score": lq,
            "topk_flipped": row["topk_flipped"],
            "random_flipped": row["random_flipped"],
        })

    results_df = pd.DataFrame(records)
    results_df.to_csv("data/tagger_results.csv", index=False)

    print("\n--- Dominant Category Distribution ---")
    for dataset in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == dataset]
        print(f"\n  {dataset.upper()}")
        print(sub["dominant_category"].value_counts().to_string())

    print("\n--- Average LQ Score ---")
    for dataset in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == dataset]
        print(f"  {dataset}: {sub['lq_score'].mean():.3f}")

    print("\nSaved to data/tagger_results.csv")
    return results_df

if __name__ == "__main__":
    run_tagger()