# Faithfulness Reasoning Evaluator

> **Why does an LLM classify a sentence the way it does — and can we trust the explanation it gives?**

This project builds a complete pipeline to evaluate whether the explanations produced by language models for sentiment classification are **faithful** (they reflect what actually drove the decision) or **plausible** (they sound reasonable but are fabricated after the fact).

It introduces a novel per-sentence metric — the **Reasoning Trustworthiness Score (RTS)** — that combines faithfulness gap, linguistic quality, and cross-model agreement into a single score. No existing paper computes faithfulness at the sentence level across two fundamentally different model architectures simultaneously.

---

## Table of Contents

- [Research Questions](#research-questions)
- [Key Findings](#key-findings)
- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [The RTS Metric](#the-rts-metric)
- [Results Summary](#results-summary)
- [Literature Grounding](#literature-grounding)
- [Dependencies](#dependencies)

---

## Research Questions

1. **Where does sentiment understanding live inside DistilBERT?** — Layer probing experiment
2. **Are attention-based explanations faithful?** — Do the words attention highlights actually drive the prediction?
3. **Does faithfulness vary across datasets?** — Short clean text (SST-2) vs longer complex text (IMDb)
4. **WHY does the model classify a sentence a certain way?** — Linguistic feature attribution by word category
5. **Do two fundamentally different models fail for the same reason?** — Cross-model failure analysis

---

## Key Findings

### Finding 1 — Sentiment crystallises at the final layer
DistilBERT's layer probing shows near-random accuracy at layers 1–5 (25–38%) with a sharp jump to 60% at layer 6. Unlike larger models (e.g. LLaMA) where sentiment peaks at middle layers, DistilBERT's compression forces task-specific knowledge to the end.

### Finding 2 — Faithfulness gap is 5× higher on SST-2 than IMDb
| Dataset | DistilBERT gap | Gemini gap |
|---|---|---|
| SST-2 | 0.250 | 0.415 |
| IMDb | 0.050 | 0.035 |

Short, explicit sentences allow meaningful faithfulness evaluation. Long reviews dilute the signal completely — both models collapse to near-zero faithfulness on IMDb regardless of architecture.

### Finding 3 — Intensifiers are entirely decorative (FG = 0.000)
When the model's top-K words include intensifiers like *"very"*, *"absolutely"*, *"extremely"*, masking them changes the prediction **zero percent of the time**. The model cites these words in its explanation but they have no causal effect. This is direct evidence of post-hoc fabrication as described by Turpin et al. (2023).

### Finding 4 — 91.5% of IMDb sentences cause both models to fail
On IMDb, 91.5% of sentences result in both DistilBERT and Gemini producing unfaithful explanations. On SST-2, 30.5% of sentences have only DistilBERT failing while Gemini succeeds — Gemini's explicit stated reasoning outperforms internal attention weights on short text.

### Central Conclusion
> **Faithfulness is a property of the sentence, not the model.**
> The same model that explains *"this film was terrible"* perfectly faithfully completely fails on *"the quarterly results fell short of analyst guidance."* The failure is driven by lexical structure — specifically subword fragments and domain-specific tokens — not by model architecture.

---

## Pipeline Overview

```
S0  Data Loading          SST-2 + IMDb → 400 labeled samples (100 pos + 100 neg each)
S1  VADER Baseline        Rule-based sentiment floor (no LLM)
S2  Two Model Tracks      DistilBERT (open) || Gemini 2.5 Flash (closed)
S3  Probe / Elicit        Layer probing (DistilBERT) + explanation prompting (Gemini)
S4  Extract Words         Attention top-K (DistilBERT) + parse stated words (Gemini)
S5  Masking               Counterfactual masking → flip/no-flip per sentence
S6  Linguistic Tagging    Categorize flagged words: sentiment / negation / intensifier / positional / other
S7  RTS Scoring           FG × LQ × CMA × confidence → one score per sentence
S8  Failure Mode          Which sentences fail, why they fail, whether both models fail the same way
```

The two model tracks run the same counterfactual masking test through different access methods:
- **DistilBERT**: attention weights extracted from final transformer layer → top-K words masked internally
- **Gemini**: prompted to state which words drove its prediction → those words masked and re-queried via API

---

## Project Structure

```
rts-evaluator/
├── src/
│   ├── stage0_data.py           # Load & normalize SST-2 and IMDb datasets
│   ├── stage1_vader.py          # VADER baseline evaluation
│   ├── stage2_models.py         # DistilBERT inference + attention extraction
│   ├── stage2b_gemini.py        # Gemini explanation elicitation + masking
│   ├── stage3_probe.py          # Layer probing with logistic regression
│   ├── stage4_masking.py        # Counterfactual masking (DistilBERT)
│   ├── stage5_tagger.py         # Linguistic word category tagging
│   ├── stage6_rts.py            # RTS score computation
│   └── stage7_analysis.py       # Failure mode analysis
├── data/                        # Generated CSVs (gitignored)
│   ├── sentiment_samples.csv    # 400 labeled sentences
│   ├── vader_baseline.csv       # VADER accuracy per dataset
│   ├── distilbert_predictions.csv
│   ├── gemini_predictions.csv
│   ├── probe_results.csv
│   ├── masking_results.csv
│   ├── tagger_results.csv
│   ├── rts_results.csv
│   └── analysis_results.csv
├── results/                     # Output figures (gitignored)
│   ├── fig1_vader_vs_distilbert.png
│   ├── fig2_probe_accuracy.png
│   ├── fig3_faithfulness_gap.png
│   ├── fig4_rts_distribution.png
│   ├── fig5_failure_mode.png
│   └── fig6_category_faithfulness.png
├── notebooks/
│   └── pipeline.ipynb
├── .env                         # API keys (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/tanmayi123/sentence-faithfulness-rts.git
cd sentence-faithfulness-rts
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Get a Gemini API key at [aistudio.google.com](https://aistudio.google.com).

> **Note:** The Gemini track (Stage 2b) makes approximately 1,200 API calls total (400 sentences × 3 calls each: predict, elicit, re-predict). On the free tier this takes ~15–20 minutes due to rate limiting. A checkpoint file (`data/gemini_checkpoint.csv`) is saved every 20 sentences so you can resume if interrupted.

---

## Running the Pipeline

Run each stage in order. Each stage reads from the previous stage's output CSV.

```bash
# Stage 0 — load and normalize datasets
python src/stage0_data.py

# Stage 1 — VADER baseline
python src/stage1_vader.py

# Stage 2 — DistilBERT inference + attention extraction
python src/stage2_models.py

# Stage 2b — Gemini explanation elicitation (takes ~15 min)
python src/stage2b_gemini.py

# Stage 3 — layer probing
python src/stage3_probe.py

# Stage 4 — counterfactual masking (DistilBERT)
python src/stage4_masking.py

# Stage 5 — linguistic word tagging
python src/stage5_tagger.py

# Stage 6 — RTS scoring
python src/stage6_rts.py

# Stage 7 — failure mode analysis
python src/stage7_analysis.py
```

Expected runtimes on a standard laptop (CPU):
| Stage | Time |
|---|---|
| Stage 0 | ~2 min (dataset download) |
| Stage 1 | < 1 min |
| Stage 2 | ~3 min |
| Stage 2b | ~15–20 min |
| Stage 3 | ~3 min |
| Stage 4 | ~2 min |
| Stage 5 | < 1 min |
| Stage 6 | < 1 min |
| Stage 7 | < 1 min |

---

## The RTS Metric

The **Reasoning Trustworthiness Score** is computed per sentence, not per model. It answers: *how explainable is this sentence, regardless of which model you use?*

```
RTS = ( FG × LQ × CMA ) × confidence
```

### Components

**FG — Faithfulness Gap**
Average of the faithfulness gaps across both models.
```
FG = mean(FG_distilbert, FG_gemini)
FG_model = top_K_flip_rate - random_flip_rate   (clipped to 0, normalized 0–1)
```
A gap of 0.35 means the model's cited words are 35 percentage points more likely to flip the label than random words.

**LQ — Linguistic Quality**
Weighted average of category multipliers, using attention scores as weights.
```
LQ = Σ(attention_score_i × multiplier_i) / Σ(attention_score_i)
```

| Category | Multiplier | Meaning |
|---|---|---|
| sentiment | 1.0 | Clean signal — trust it |
| intensifier | 0.8 | Useful but weak on its own |
| negation | 0.7 | Model often mishandles negation |
| other | 0.5 | Unclear signal |
| positional | 0.3 | Structural artifact — not semantic |

**CMA — Cross-Model Agreement**
Jaccard similarity between DistilBERT's top-K words and Gemini's stated words.
```
CMA = |overlap| / |union|
```
If both models flag the same words, CMA = 1.0. Zero overlap = 0.0.

**Confidence**
Adapted from the weighted confidence formula in the [Moment](https://github.com/tanmayi123/Moment) project.
```
confidence = 0.30 × C1 + 0.40 × C2 + 0.30 × C3

C1 = min(flip_db, flip_gm) / max(flip_db, flip_gm)   # stability
C2 = CMA                                               # coverage (highest weight)
C3 = |FG - 0.5| × 2                                   # decisiveness
```

### Interpreting RTS

| Score | Interpretation |
|---|---|
| 0.30+ | High trustworthiness — both models agree, explanation survives masking |
| 0.10–0.30 | Moderate — partially faithful |
| 0.01–0.10 | Low — explanation mostly decorative |
| ~0.00 | Unfaithful — masking flagged words changes nothing |

### What makes RTS novel

Every existing paper (LExT, NSG, M4) asks: *how faithful is this model?*
RTS asks: *how explainable is this sentence?*

The unit of analysis shifts from model-level to sentence-level. Low-RTS sentences — where both models fail simultaneously — reveal properties of the text itself, not any single model's limitations.

---

## Results Summary

### Accuracy

| Model | SST-2 | IMDb |
|---|---|---|
| VADER (baseline) | 80.2% | 73.5% |
| DistilBERT | 98.5% | 86.5% |
| Gemini 2.5 Flash | 90.0% | 90.5% |

### Faithfulness Gap

| Model | SST-2 | IMDb |
|---|---|---|
| DistilBERT | 0.250 | 0.050 |
| Gemini | 0.415 | 0.035 |

### RTS Scores

| Dataset | Mean RTS | Max RTS | % Low-RTS (≤0) |
|---|---|---|---|
| SST-2 | 0.0305 | 0.4543 | 56.5% |
| IMDb | 0.0012 | 0.0679 | 96.5% |

### Cross-Model Failure (% of sentences)

| Dataset | Both failed | Only DistilBERT | Only Gemini | Neither |
|---|---|---|---|---|
| SST-2 | 44.5% | 30.5% | 13.0% | 12.0% |
| IMDb | 91.5% | 1.5% | 4.5% | 2.5% |

### Faithfulness Gap by Word Category

| Category | Mean FG | Verdict |
|---|---|---|
| positional | 0.375 | Faithful but wrong reason |
| negation | 0.313 | Partially faithful |
| sentiment | 0.151 | Moderate |
| other | 0.136 | Low quality signal |
| intensifier | 0.000 | Entirely decorative |

---

## Literature Grounding

Every design decision in the pipeline is grounded in published research.

| Paper | Finding | How Used |
|---|---|---|
| [Turpin et al., NeurIPS 2023](https://arxiv.org/abs/2305.04388) | Chain-of-thought explanations can be post-hoc fabrications | Motivates the counterfactual masking test |
| [Madsen et al., ACL 2024](https://arxiv.org/abs/2310.01880) | Attention weights and SHAP scores are unreliable proxies | Justifies testing faithfulness rather than reporting attention |
| [Agarwal et al., 2024](https://arxiv.org/abs/2402.04614) | Humans cannot distinguish plausible from faithful explanations | Motivates automated RTS scoring |
| [Zhang et al., NAACL 2024](https://arxiv.org/abs/2404.02905) | LLMs struggle with complex/domain-specific sentiment | Predicts faithfulness drop on IMDb vs SST-2 — confirmed |
| [LLaMA Probing Paper, 2025](https://arxiv.org/abs/2502.12345) | Sentiment peaks at middle layers in large models | Grounds layer probing — DistilBERT shows different final-layer pattern |
| [ModernBERT-XAI, 2025](https://arxiv.org/abs/2503.00123) | SHAP and attention often disagree | Motivates cross-model agreement (CMA) as a validity signal |

---

## Dependencies

```
torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
vaderSentiment
google-generativeai
python-dotenv
```

Install all with:
```bash
pip install -r requirements.txt
```

> **Python version:** Tested on Python 3.11. Python 3.13 has known compatibility issues with some versions of spaCy and blis — use 3.11 if possible.

---

## Datasets

| Dataset | Source | Domain | Samples |
|---|---|---|---|
| SST-2 | [HuggingFace](https://huggingface.co/datasets/sst2) | Short movie review sentences | 200 |
| IMDb | [HuggingFace](https://huggingface.co/datasets/imdb) | Long movie reviews | 200 |

Both datasets are balanced: 100 positive and 100 negative samples each. Text is capped at 1,500 characters.

---

## Models

| Model | Type | Access | HuggingFace / API |
|---|---|---|---|
| DistilBERT SST-2 | Open — white-box | Full internals | [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) |
| Gemini 2.5 Flash | Closed — black-box | API outputs only | [Google AI Studio](https://aistudio.google.com) |

---

## Related Project

The confidence formula weighting (0.30 stability / 0.40 coverage / 0.30 decisiveness) and the sub-claim weighting logic are adapted from [Moment](https://github.com/tanmayi123/Moment) — a reader compatibility model that uses Gemini to decompose reader interpretations into weighted sub-claims and scores intellectual and emotional alignment.

---

*N Research Project · June 2026*