import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "distilbert-base-uncased"

def get_all_layer_embeddings(texts, tokenizer, model):
    all_layers = [[] for _ in range(6)]  # DistilBERT has 6 layers

    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"  Extracting embeddings {i}/{len(texts)}...")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of 7 tensors (embedding layer + 6 transformer layers)
        # each tensor shape: (1, seq_len, 768)
        # we take the [CLS] token (index 0) from each of the 6 transformer layers
        hidden_states = outputs.hidden_states

        for layer_idx in range(6):
            # +1 because index 0 is the embedding layer, not a transformer layer
            cls_vector = hidden_states[layer_idx + 1][0][0].numpy()
            all_layers[layer_idx].append(cls_vector)

    return [np.array(layer) for layer in all_layers]


def run_probing():
    df = pd.read_csv("data/sentiment_samples.csv")
    labels = df["label"].values

    print("Loading base DistilBERT (no classification head)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    print("\nExtracting CLS vectors from all 6 layers...")
    layer_embeddings = get_all_layer_embeddings(df["text"].tolist(), tokenizer, model)

    print("\nTraining logistic probe on each layer...")
    results = []
    for layer_idx, embeddings in enumerate(layer_embeddings):
        scaler = StandardScaler()
        X = scaler.fit_transform(embeddings)
        y = labels

        # simple train/test split — first 80% train, last 20% test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        results.append({
            "layer": layer_idx + 1,
            "accuracy": round(acc, 4)
        })
        print(f"  Layer {layer_idx + 1}: {acc:.1%}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/probe_results.csv", index=False)
    print("\nSaved to data/probe_results.csv")

    best = results_df.loc[results_df["accuracy"].idxmax()]
    print(f"\nBest layer: {int(best['layer'])} with accuracy {best['accuracy']:.1%}")
    return results_df

if __name__ == "__main__":
    run_probing()