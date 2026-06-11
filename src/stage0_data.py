import pandas as pd
from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

MAX_CHARS = 1500
SAMPLES_PER_CLASS = 100

def truncate(text):
    return str(text)[:MAX_CHARS]

def load_sst2():
    dataset = load_dataset("sst2", split="train")
    df = pd.DataFrame(dataset)
    df = df.rename(columns={"sentence": "text", "label": "label"})
    neg = df[df.label == 0].sample(SAMPLES_PER_CLASS, random_state=42)
    pos = df[df.label == 1].sample(SAMPLES_PER_CLASS, random_state=42)
    df = pd.concat([neg, pos]).reset_index(drop=True)
    df["text"] = df["text"].apply(truncate)
    df["dataset"] = "sst2"
    df["label_str"] = df["label"].map({0: "negative", 1: "positive"})
    df["sample_id"] = ["sst2_" + str(i).zfill(4) for i in range(len(df))]
    return df[["sample_id", "dataset", "text", "label", "label_str"]]

def load_imdb():
    dataset = load_dataset("imdb", split="train")
    df = pd.DataFrame(dataset)
    neg = df[df.label == 0].sample(SAMPLES_PER_CLASS, random_state=42)
    pos = df[df.label == 1].sample(SAMPLES_PER_CLASS, random_state=42)
    df = pd.concat([neg, pos]).reset_index(drop=True)
    df["text"] = df["text"].apply(truncate)
    df["dataset"] = "imdb"
    df["label_str"] = df["label"].map({0: "negative", 1: "positive"})
    df["sample_id"] = ["imdb_" + str(i).zfill(4) for i in range(len(df))]
    return df[["sample_id", "dataset", "text", "label", "label_str"]]

def load_financial():
    dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train")
    df = pd.DataFrame(dataset)
    df = df.rename(columns={"sentence": "text"})
    # in this version: 0=negative, 1=neutral, 2=positive
    neg = df[df.label == 0].sample(SAMPLES_PER_CLASS, random_state=42)
    pos = df[df.label == 2].sample(SAMPLES_PER_CLASS, random_state=42)
    df = pd.concat([neg, pos]).reset_index(drop=True)
    df["text"] = df["text"].apply(truncate)
    df["dataset"] = "financial"
    df["label"] = df["label"].map({0: 0, 2: 1})
    df["label_str"] = df["label"].map({0: "negative", 1: "positive"})
    df["sample_id"] = ["financial_" + str(i).zfill(4) for i in range(len(df))]
    return df[["sample_id", "dataset", "text", "label", "label_str"]]

def build_dataset():
    print("Loading SST-2...")
    sst2 = load_sst2()
    print("Loading IMDb...")
    imdb = load_imdb()

    combined = pd.concat([sst2, imdb]).reset_index(drop=True)
    combined.to_csv("data/sentiment_samples.csv", index=False)
    print(f"\nDone. Total samples: {len(combined)}")
    print(combined.groupby(["dataset", "label_str"]).size())
    return combined

if __name__ == "__main__":
    build_dataset()