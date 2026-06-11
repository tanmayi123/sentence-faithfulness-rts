import torch
import pandas as pd
import ast
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
TOP_K = 3

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    return pred

def mask_words(text, words_to_mask):
    masked = text
    for word in words_to_mask:
        # clean the token (remove ## for subword tokens)
        clean = word.replace("##", "")
        masked = masked.replace(clean, "[MASK]")
    return masked

def get_random_words(text, tokenizer, k, exclude_words):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens = [t for t in tokens if t not in ["[CLS]", "[SEP]", "[PAD]"] and t not in exclude_words]
    if len(tokens) < k:
        return tokens
    return random.sample(tokens, k)

def run_masking():
    df = pd.read_csv("data/distilbert_predictions.csv")
    tokenizer, model = load_model()

    random.seed(42)
    records = []

    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Processing {i}/{len(df)}...")

        text = row["text"]
        original_pred = row["pred_label"]
        top_k_words = ast.literal_eval(row["top_k_words"])

        # top-K masking
        masked_topk = mask_words(text, top_k_words)
        pred_topk = predict(masked_topk, tokenizer, model)
        topk_flipped = int(pred_topk != original_pred)

        # random masking
        random_words = get_random_words(text, tokenizer, TOP_K, top_k_words)
        masked_random = mask_words(text, random_words)
        pred_random = predict(masked_random, tokenizer, model)
        random_flipped = int(pred_random != original_pred)

        records.append({
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "text": text,
            "true_label": row["true_label"],
            "pred_label": original_pred,
            "correct": row["correct"],
            "top_k_words": row["top_k_words"],
            "masked_topk_text": masked_topk,
            "pred_after_topk_mask": pred_topk,
            "topk_flipped": topk_flipped,
            "random_words": str(random_words),
            "masked_random_text": masked_random,
            "pred_after_random_mask": pred_random,
            "random_flipped": random_flipped,
        })

    results_df = pd.DataFrame(records)
    results_df.to_csv("data/masking_results.csv", index=False)

    print("\n--- Faithfulness Gap per Dataset ---")
    for dataset in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == dataset]
        topk_rate = sub["topk_flipped"].mean()
        random_rate = sub["random_flipped"].mean()
        gap = topk_rate - random_rate
        print(f"\n  {dataset.upper()}")
        print(f"  Top-K flip rate:    {topk_rate:.1%}")
        print(f"  Random flip rate:   {random_rate:.1%}")
        print(f"  Faithfulness gap:   {gap:.3f}")

    print("\nSaved to data/masking_results.csv")
    return results_df

if __name__ == "__main__":
    run_masking()