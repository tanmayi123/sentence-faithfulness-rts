import pandas as pd
import google.generativeai as genai
import os
import time
import ast
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

TOP_K = 3

def elicit_explanation(text, label_str):
    prompt = f"""You are a sentiment analysis model. 
The following sentence was classified as {label_str}:

"{text}"

List exactly {TOP_K} words from the sentence that most influenced this classification.
Reply with ONLY a Python list of strings, nothing else.
Example: ["word1", "word2", "word3"]"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = raw.replace("```python", "").replace("```", "").strip()
        words = ast.literal_eval(raw)
        if isinstance(words, list):
            return [str(w).lower().strip() for w in words[:TOP_K]]
    except Exception as e:
        print(f"  Parse error: {e}")
    return []

def mask_words(text, words):
    masked = text
    for word in words:
        masked = masked.replace(word, "[MASK]")
    return masked

def gemini_predict(text):
    prompt = f"""Classify the sentiment of this sentence as exactly one word: positive or negative.

"{text}"

Reply with ONLY one word: positive or negative."""
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip().lower()
        if "positive" in raw:
            return 1
        elif "negative" in raw:
            return 0
    except Exception as e:
        print(f"  Prediction error: {e}")
    return -1

def run_gemini_track():
    df = pd.read_csv("data/sentiment_samples.csv")
    records = []

    print(f"Running Gemini on {len(df)} sentences...")
    for i, row in df.iterrows():
        if i % 20 == 0:
            print(f"  Processing {i}/{len(df)}...")

        text = row["text"]
        true_label = row["label"]

        # Step 1 — get prediction
        pred_label = gemini_predict(text)
        pred_label_str = "positive" if pred_label == 1 else "negative"

        # Step 2 — elicit explanation
        stated_words = elicit_explanation(text, pred_label_str)

        # Step 3 — mask stated words and re-predict
        if stated_words:
            masked_text = mask_words(text, stated_words)
            pred_after_mask = gemini_predict(masked_text)
            topk_flipped = int(pred_after_mask != pred_label)
        else:
            topk_flipped = 0

        # Step 4 — random masking baseline
        all_words = [w for w in text.lower().split()
                     if len(w) > 2 and w not in stated_words]
        import random
        random.seed(i)
        random_words = random.sample(all_words, min(TOP_K, len(all_words)))
        masked_random = mask_words(text, random_words)
        pred_after_random = gemini_predict(masked_random)
        random_flipped = int(pred_after_random != pred_label)

        records.append({
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": int(pred_label == true_label),
            "stated_words": str(stated_words),
            "topk_flipped": topk_flipped,
            "random_flipped": random_flipped,
        })

        # rate limit — Gemini free tier allows ~60 requests/min
        time.sleep(1.5)

    results_df = pd.DataFrame(records)
    results_df.to_csv("data/gemini_predictions.csv", index=False)

    print("\n--- Gemini Accuracy ---")
    for dataset in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == dataset]
        acc = sub["correct"].mean()
        print(f"  {dataset}: {acc:.1%}")

    print("\n--- Gemini Faithfulness Gap ---")
    for dataset in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == dataset]
        topk_rate = sub["topk_flipped"].mean()
        random_rate = sub["random_flipped"].mean()
        gap = topk_rate - random_rate
        print(f"  {dataset}: gap={gap:.3f} (top-K={topk_rate:.1%}, random={random_rate:.1%})")

    print("\nSaved to data/gemini_predictions.csv")
    return results_df

if __name__ == "__main__":
    run_gemini_track()

