import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def vader_predict(text, analyzer):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return 1  # positive
    elif score <= -0.05:
        return 0  # negative
    else:
        return -1  # neutral / uncertain

def run_vader_baseline():
    df = pd.read_csv("data/sentiment_samples.csv")
    analyzer = SentimentIntensityAnalyzer()

    df["vader_pred"] = df["text"].apply(lambda x: vader_predict(x, analyzer))
    df["vader_neutral"] = (df["vader_pred"] == -1)

    results = []
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset].copy()
        # only evaluate on non-neutral predictions
        decided = sub[sub["vader_pred"] != -1]
        correct = (decided["vader_pred"] == decided["label"]).sum()
        accuracy = correct / len(decided) if len(decided) > 0 else 0
        neutral_rate = sub["vader_neutral"].mean()

        results.append({
            "dataset": dataset,
            "total": len(sub),
            "decided": len(decided),
            "correct": correct,
            "accuracy": round(accuracy, 3),
            "neutral_rate": round(neutral_rate, 3)
        })

        print(f"\n{dataset.upper()}")
        print(f"  Total:       {len(sub)}")
        print(f"  Decided:     {len(decided)}  (neutral/skipped: {sub['vader_neutral'].sum()})")
        print(f"  Accuracy:    {accuracy:.1%}")
        print(f"  Neutral rate:{neutral_rate:.1%}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/vader_baseline.csv", index=False)
    print("\nSaved to data/vader_baseline.csv")
    return results_df

if __name__ == "__main__":
    run_vader_baseline()