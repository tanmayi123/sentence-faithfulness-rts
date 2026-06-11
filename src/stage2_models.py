import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def load_distilbert():
    print("Loading DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, output_attentions=True
    )
    model.eval()
    return tokenizer, model

def predict_with_attention(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # prediction
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_label].item()

    # attention — last layer, average across all heads
    # shape: (num_heads, seq_len, seq_len)
    last_layer_attention = outputs.attentions[-1][0]
    # average across heads → (seq_len, seq_len)
    avg_attention = last_layer_attention.mean(dim=0)
    # attention FROM [CLS] token TO every other token
    cls_attention = avg_attention[0].tolist()

    # map back to words
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_attention = list(zip(tokens, cls_attention))

    # remove special tokens
    word_attention = [
        (tok, score) for tok, score in word_attention
        if tok not in ["[CLS]", "[SEP]", "[PAD]"]
    ]

    # sort by attention score descending
    word_attention_sorted = sorted(word_attention, key=lambda x: x[1], reverse=True)

    return {
        "pred_label": pred_label,
        "pred_label_str": "positive" if pred_label == 1 else "negative",
        "confidence": round(confidence, 4),
        "word_attention": word_attention_sorted
    }

def run_distilbert_inference():
    df = pd.read_csv("data/sentiment_samples.csv")
    tokenizer, model = load_distilbert()

    records = []
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Processing {i}/{len(df)}...")

        result = predict_with_attention(row["text"], tokenizer, model)
        records.append({
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "text": row["text"],
            "true_label": row["label"],
            "true_label_str": row["label_str"],
            "pred_label": result["pred_label"],
            "pred_label_str": result["pred_label_str"],
            "correct": int(result["pred_label"] == row["label"]),
            "confidence": result["confidence"],
            "top_k_words": str([w for w, s in result["word_attention"][:3]]),
            "top_k_scores": str([round(s, 4) for w, s in result["word_attention"][:3]])
        })

    results_df = pd.DataFrame(records)
    results_df.to_csv("data/distilbert_predictions.csv", index=False)

    print("\n--- DistilBERT Accuracy ---")
    for dataset in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == dataset]
        acc = sub["correct"].mean()
        print(f"  {dataset}: {acc:.1%}")

    print("\nSaved to data/distilbert_predictions.csv")
    return results_df

if __name__ == "__main__":
    run_distilbert_inference()