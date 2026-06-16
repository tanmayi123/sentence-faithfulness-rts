import streamlit as st
import pandas as pd
import numpy as np
import ast
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Faithfulness Reasoning Evaluator",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLES ────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* Fix expander label overlap */
details > summary p {
    display: none !important;
}
details > summary p {
    display: none !important;
}
[data-testid="stExpander"] summary span {
    font-family: "Times New Roman", Times, serif !important;
    color: #0D1B2A !important;
}
/* Import Times New Roman fallback via Google Fonts serif */
* {
    font-family: "Times New Roman", Times, serif !important;
    box-sizing: border-box;
}

/* Force light background everywhere */
html, body, .stApp, .main, section.main, [data-testid="stAppViewContainer"] {
    background-color: #F8F6F1 !important;
}

/* All text dark by default */
p, span, li, label, div, td, th, small, strong, em, a {
    color: #1A1A1A !important;
}

/* Sidebar dark */
[data-testid="stSidebar"] {
    background-color: #0D1B2A !important;
}
[data-testid="stSidebar"] * {
    color: #E8F4F8 !important;
}
[data-testid="stSidebar"] .stRadio label span {
    color: #E8F4F8 !important;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #0D1B2A !important;
    font-family: "Times New Roman", Times, serif !important;
}

/* Streamlit widget labels */
[data-testid="stWidgetLabel"] p,
.stSelectbox label, .stTextInput label,
.stSlider label, .stRadio label {
    color: #1A1A1A !important;
    font-weight: bold;
}

/* Metric labels and values */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: #0D1B2A !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #555 !important;
    font-family: "Times New Roman", Times, serif !important;
}
.stTabs [aria-selected="true"] {
    color: #00B4A6 !important;
    border-bottom: 2px solid #00B4A6 !important;
}

/* Expander */
details summary {
    color: #0D1B2A !important;
}

/* Input boxes */
input, textarea, select {
    color: #1A1A1A !important;
    background: #FFFFFF !important;
}

/* Selectbox text */
[data-baseweb="select"] * {
    color: #1A1A1A !important;
    background: #FFFFFF !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* Custom components */
.hero-box {
    background: linear-gradient(135deg, #0D1B2A 0%, #1A3A5C 100%);
    border-radius: 6px;
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin-bottom: 2rem;
}
.hero-box h1 {
    color: #FFFFFF !important;
    font-size: 2.6rem;
    margin: 0.2rem 0 0.5rem 0;
    line-height: 1.15;
}
.hero-subtitle {
    color: #8BAAB8 !important;
    font-style: italic;
    font-size: 1.05rem;
    margin-bottom: 1.2rem;
}
.hero-label {
    color: #00B4A6 !important;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.pill-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.pill {
    border: 1px solid #00B4A6;
    color: #00B4A6 !important;
    border-radius: 3px;
    padding: 0.2rem 0.75rem;
    font-size: 0.82rem;
    background: rgba(0,180,166,0.1);
}
.section-eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: #00B4A6 !important;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
}
.section-heading {
    font-size: 1.65rem;
    color: #0D1B2A !important;
    border-bottom: 2px solid #0D1B2A;
    padding-bottom: 0.3rem;
    margin-bottom: 1.2rem;
}
.info-box {
    background: #EBF6F5;
    border-left: 3px solid #00B4A6;
    padding: 0.85rem 1.1rem;
    border-radius: 0 4px 4px 0;
    margin: 0.75rem 0 1.25rem 0;
    font-size: 0.93rem;
    line-height: 1.55;
}
.info-box p, .info-box span, .info-box strong {
    color: #1A3A5C !important;
}
.formula-box {
    background: #0D1B2A;
    border-radius: 4px;
    padding: 1.1rem 1.5rem;
    text-align: center;
    font-size: 1.15rem;
    margin: 0.75rem 0;
    letter-spacing: 0.02em;
}
.formula-box span {
    color: #E8F4F8 !important;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #D8D0C4;
    border-radius: 4px;
    padding: 1rem 1.1rem;
}
.metric-big {
    font-size: 2.1rem;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.metric-big.teal  { color: #00B4A6 !important; }
.metric-big.amber { color: #B5720A !important; }
.metric-big.coral { color: #B03030 !important; }
.metric-big.navy  { color: #0D1B2A !important; }
.metric-sub {
    font-size: 0.8rem;
    color: #6A604E !important;
    line-height: 1.3;
}
.sent-card {
    background: #FFFFFF;
    border-left: 3px solid #00B4A6;
    padding: 0.9rem 1.1rem 0.75rem 1.1rem;
    margin-bottom: 0.7rem;
    border-radius: 0 4px 4px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.sent-text {
    font-style: italic;
    font-size: 0.97rem;
    line-height: 1.5;
    color: #1A1A1A !important;
}
.badge-row {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-top: 0.45rem;
}
.badge {
    font-size: 0.72rem;
    padding: 0.12rem 0.5rem;
    border-radius: 2px;
}
.b-teal   { background:#D0F0ED; color:#005C56 !important; }
.b-amber  { background:#FAE8C8; color:#7A4800 !important; }
.b-coral  { background:#FAD4D4; color:#780000 !important; }
.b-gray   { background:#E4E0D8; color:#3A3328 !important; }
.b-purple { background:#E4D4FA; color:#3A0070 !important; }
.rts-bar-bg {
    background: #E4E0D8;
    border-radius: 2px;
    height: 6px;
    margin-top: 0.45rem;
    overflow: hidden;
}
.rts-bar-fill {
    height: 100%;
    border-radius: 2px;
}
.score-box {
    background: #0D1B2A;
    border-radius: 6px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.2rem;
}
.score-number {
    font-size: 3.8rem;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.score-label {
    font-size: 0.85rem;
    color: #8BAAB8 !important;
}
.pipeline-step {
    background: #FFFFFF;
    border: 1px solid #D8D0C4;
    border-radius: 4px;
    padding: 1.1rem 1.3rem 1rem 1.3rem;
    margin-bottom: 0.5rem;
}
.step-num {
    font-size: 0.7rem;
    background: #0D1B2A;
    color: #FFFFFF !important;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-right: 0.5rem;
    font-weight: bold;
}
.finding-card {
    background: #FFFFFF;
    border: 1px solid #D8D0C4;
    border-radius: 4px;
    padding: 1.1rem 1.2rem;
}
.finding-card p, .finding-card span {
    color: #2A2A2A !important;
    font-size: 0.92rem;
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    probe    = pd.read_csv("data/probe_results.csv")
    analysis = pd.read_csv("data/analysis_results.csv")
    masking  = pd.read_csv("data/masking_results.csv")
    gemini   = pd.read_csv("data/gemini_predictions.csv")
    tagger   = pd.read_csv("data/tagger_results.csv")
    return probe, analysis, masking, gemini, tagger

try:
    probe_df, analysis_df, masking_df, gemini_df, tagger_df = load_data()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    ERR = str(e)

# ── PLOTLY DEFAULTS ───────────────────────────────────────────────────────
PAL = dict(teal="#00B4A6", navy="#0D1B2A", amber="#B5720A",
           coral="#B03030", purple="#7B5EA7", muted="#8BAAB8",
           bg="#F8F6F1", white="#FFFFFF", gray="#6A604E")

def base_fig(title="", h=380):
    return dict(
        title=dict(text=title, font=dict(family="Times New Roman", size=14, color=PAL["navy"]), x=0.01),
        plot_bgcolor=PAL["white"], paper_bgcolor=PAL["bg"],
        font=dict(family="Times New Roman", color="#1A1A1A"),
        height=h, margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(gridcolor="#E4E0D8", linecolor="#D0C8BC",
                   tickfont=dict(family="Times New Roman", color="#1A1A1A")),
        yaxis=dict(gridcolor="#E4E0D8", linecolor="#D0C8BC",
                   tickfont=dict(family="Times New Roman", color="#1A1A1A")),
        legend=dict(font=dict(family="Times New Roman", color="#1A1A1A"))
    )

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Faithfulness Evaluator")
    st.markdown("---")
    page = st.radio("Navigate", [
        "Overview", "Pipeline Explainer", "Results Dashboard",
        "Sentence Explorer", "Live Scorer"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Datasets**")
    st.markdown("SST-2 and IMDb")
    st.markdown("**Models**")
    st.markdown("DistilBERT and Gemini 2.5 Flash")
    st.markdown("**Metric**")
    st.markdown("RTS (novel)")
    st.markdown("---")
    st.markdown("NLP Research Project, June 2026")

# ── HERO ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
  <div class="hero-label">NLP Research Project</div>
  <h1>Faithfulness Reasoning Evaluator</h1>
  <div class="hero-subtitle">Why does an LLM classify a sentence the way it does, and can we trust the explanation it gives?</div>
  <div class="pill-row">
    <span class="pill">400 sentences</span>
    <span class="pill">2 model tracks</span>
    <span class="pill">Novel RTS metric</span>
    <span class="pill">6 research findings</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not DATA_OK:
    st.error(f"Could not load CSVs from data/ folder. Make sure all pipeline stages have been run. Error: {ERR}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="section-eyebrow">What this project asks</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">The Core Question</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <p>When a language model classifies a sentence as positive or negative, it often names the words
    that drove its decision. This project tests whether those explanations are
    <strong>faithful</strong> (the cited words actually caused the prediction) or
    <strong>plausible</strong> (they sound convincing but were fabricated after the fact).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-big teal">0.415</div>
        <div class="metric-sub">Gemini faithfulness gap on SST-2, the highest observed</div>
      </div>
      <div class="metric-card">
        <div class="metric-big amber">0.000</div>
        <div class="metric-sub">Intensifier faithfulness gap, entirely decorative</div>
      </div>
      <div class="metric-card">
        <div class="metric-big coral">91.5%</div>
        <div class="metric-sub">IMDb sentences where both models fail simultaneously</div>
      </div>
      <div class="metric-card">
        <div class="metric-big navy">25x</div>
        <div class="metric-sub">SST-2 mean RTS divided by IMDb mean RTS</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="section-eyebrow">Finding 01</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding-card"><strong>Sentiment crystallises at layer 6</strong><p>DistilBERT shows near-random accuracy at layers 1 to 5 (25 to 38%), then jumps to 60% at layer 6. Compression forces task-specific knowledge to the final layer, unlike larger models where sentiment peaks in the middle.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-eyebrow">Finding 02</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding-card"><strong>Intensifiers are entirely decorative</strong><p>When the model cites words like "very", "absolutely", or "extremely", masking them changes the prediction zero percent of the time. The model fabricates these citations after its decision is already made.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="section-eyebrow">Finding 03</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding-card"><strong>Faithfulness is a sentence property</strong><p>91.5% of IMDb sentences cause both DistilBERT and Gemini to fail simultaneously. The failure is determined by lexical structure, not model architecture. Short explicit sentences are explainable; long complex ones are not.</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-eyebrow">The novel metric</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Reasoning Trustworthiness Score</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
      <span>RTS &nbsp;=&nbsp; ( FG &nbsp;&times;&nbsp; LQ &nbsp;&times;&nbsp; CMA ) &nbsp;&times;&nbsp; confidence</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**FG: Faithfulness Gap**")
        st.markdown("Top-K flip rate minus random flip rate, averaged across both models.")
    with col2:
        st.markdown("**LQ: Linguistic Quality**")
        st.markdown("Weighted average of word category multipliers using attention scores.")
    with col3:
        st.markdown("**CMA: Cross-Model Agreement**")
        st.markdown("Jaccard similarity between DistilBERT's and Gemini's flagged words.")
    with col4:
        st.markdown("**Confidence**")
        st.markdown("0.30 times stability plus 0.40 times coverage plus 0.30 times decisiveness.")

    st.markdown("""
    <div class="info-box">
    <p>Every existing paper (LExT, NSG, M4) scores a <em>model</em> on a dataset. RTS scores a
    <em>sentence</em> across two models simultaneously. Low-RTS sentences reveal properties
    of the text itself, not any single model.</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE EXPLAINER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Pipeline Explainer":
    st.markdown('<div class="section-eyebrow">How the project works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Pipeline Stage by Stage</div>', unsafe_allow_html=True)

    stages = [
        ("S0", "Data Loading", PAL["muted"],
         "Load SST-2 and IMDb from HuggingFace. Sample 100 positive and 100 negative from each. Cap text at 1,500 characters. Normalize to a consistent schema with sample_id, dataset, text, label, and label_str.",
         "data/sentiment_samples.csv, 400 rows",
         "SST-2 contains short movie review sentences and is a standard NLP benchmark. IMDb has longer, more complex reviews. Together they test whether faithfulness changes with text complexity."),
        ("S1", "VADER Baseline", PAL["teal"],
         "Run VADER, a rule-based lexicon sentiment tool, on all 400 sentences. It requires no model or API calls. It sets a performance floor: DistilBERT must clearly outperform it to justify evaluating its reasoning.",
         "data/vader_baseline.csv",
         "VADER scores 80.2% on SST-2 and 73.5% on IMDb. It also assigns neutral to 37% of SST-2 sentences because they are too short and subtle for lexicon matching. Those are exactly the interesting cases for faithfulness analysis."),
        ("S2", "Two Model Tracks", PAL["purple"],
         "The pipeline splits into two parallel tracks. Track A uses DistilBERT, an open model where we can inspect full internals including attention weights and hidden states. Track B uses Gemini 2.5 Flash, a closed model where we only see API outputs.",
         "Two parallel prediction files",
         "Both tracks run the same counterfactual masking test, but through different access methods. DistilBERT uses internal attention; Gemini is asked explicitly which words drove its classification."),
        ("S3", "Layer Probing", PAL["purple"],
         "Freeze DistilBERT completely. For each of its 6 transformer layers, extract the [CLS] token vector for every sentence. Train a logistic regression on those vectors. Probe accuracy at each layer reveals how much sentiment information is encoded there.",
         "data/probe_results.csv, 6 rows",
         "Result: near-random at layers 1 to 5, sharp jump to 60% at layer 6. Sentiment crystallises only at the final layer. This contrasts with larger models like LLaMA where sentiment peaks at middle layers, suggesting compression changes where knowledge is stored."),
        ("S4", "Attention Extraction and Word Parsing", PAL["amber"],
         "DistilBERT: average attention weights across all heads in the final transformer layer. Take the top-K tokens by attention score from [CLS]. Gemini: send a structured prompt asking which words drove the classification, then parse the response.",
         "Top-K words per sentence per model",
         "Stopwords and short subword fragments are filtered from DistilBERT's top-K. Without this filter, trivial tokens like 'the' and '##ing' dominate and the faithfulness test becomes meaningless."),
        ("S5", "Counterfactual Masking", PAL["coral"],
         "Replace the top-K flagged words with [MASK] tokens and re-run the sentence. Check whether the predicted label flips. Also mask K random words as a control. The faithfulness gap is: top-K flip rate minus random flip rate.",
         "Flip or no-flip boolean per sentence",
         "If the explanation were genuine, removing the cited words should destabilise the prediction. A near-zero gap means masking the supposedly important words has the same effect as masking random words. The explanation is decorative."),
        ("S6", "Linguistic Tagging", PAL["teal"],
         "Classify each flagged word into a category: sentiment word, negation, intensifier, positional (first or last token), or other. Each category gets a quality multiplier that feeds into the LQ component of RTS.",
         "data/tagger_results.csv",
         "Category multipliers: sentiment 1.0, intensifier 0.8, negation 0.7, other 0.5, positional 0.3. Intensifiers are penalised because the model frequently cites them while they have zero causal effect on the prediction."),
        ("S7", "RTS Scoring", PAL["amber"],
         "Compute the Reasoning Trustworthiness Score per sentence: RTS = (FG times LQ times CMA) times confidence. FG is the faithfulness gap averaged across both models. LQ is the weighted category multiplier average. CMA is Jaccard similarity between both models' flagged words.",
         "data/rts_results.csv, one RTS score per sentence",
         "All four components are between 0 and 1, so RTS is too. The multiply structure means all three main components must be non-zero for a high score. One weak component drags the whole result down. This is stricter than any existing single metric."),
        ("S8", "Failure Mode Analysis", PAL["navy"],
         "Group low-RTS sentences and ask: which dataset has more failures, which word category dominates, and did both models fail for the same reason? This produces the three-layered conclusion: which sentences fail, why they fail, and whether both models fail the same way.",
         "data/analysis_results.csv plus 6 figures",
         "Central finding: 91.5% of IMDb failures involve both models simultaneously, dominated by other-category tokens such as subword fragments. The failure is structural and reflects the sentence's lexical complexity, not any single model's limitations."),
    ]

    for n, name, color, desc, output, detail in stages:
        st.markdown(f"""
        <div class="pipeline-step">
            <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;">
                <div style="background:{color};border-radius:50%;width:32px;height:32px;
                display:flex;align-items:center;justify-content:center;
                color:white;font-size:0.7rem;font-weight:bold;flex-shrink:0;">{n}</div>
                <strong style="font-size:1.05rem;color:#0D1B2A;">{name}</strong>
            </div>
            <p style="color:#2A2A2A;margin:0 0 0.5rem 0;line-height:1.55;">{desc}</p>
            <div class="info-box"><p style="color:#1A3A5C;">{detail}</p></div>
            <p style="font-size:0.82rem;color:#6A604E;margin:0.3rem 0 0 0;">Output: <code>{output}</code></p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Results Dashboard":
    st.markdown('<div class="section-eyebrow">Research findings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Results Dashboard</div>', unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Baseline", "Layer Probing", "Faithfulness Gap",
        "RTS Distribution", "Cross-Model Failure", "Word Categories"
    ])

    with t1:
        st.markdown("**VADER baseline vs DistilBERT accuracy**")
        st.markdown('<div class="info-box"><p>VADER is a rule-based tool requiring no model or API. It sets the minimum performance floor. DistilBERT must clearly outperform it to justify evaluating its reasoning chains.</p></div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="VADER", x=["SST-2","IMDb"], y=[0.802,0.735],
            marker_color="#AAAAAA", text=["80.2%","73.5%"], textposition="outside",
            textfont=dict(family="Times New Roman", color="#1A1A1A")))
        fig.add_trace(go.Bar(name="DistilBERT", x=["SST-2","IMDb"], y=[0.985,0.865],
            marker_color=PAL["purple"], text=["98.5%","86.5%"], textposition="outside",
            textfont=dict(family="Times New Roman", color="#1A1A1A")))
        fig.add_hline(y=0.85, line_dash="dash", line_color="#999",
            annotation_text="85% threshold", annotation_font=dict(family="Times New Roman", color="#555"))
        fig.update_layout(**base_fig("Figure 1: VADER vs DistilBERT Accuracy"))
        fig.update_layout(barmode="group", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("DistilBERT on SST-2", "98.5%", "+18.3pp over VADER")
        c2.metric("DistilBERT on IMDb", "86.5%", "+13.0pp over VADER")
        c3.metric("VADER neutral rate SST-2", "37%", "Cannot handle short subtle text")

    with t2:
        st.markdown("**Where does sentiment live inside DistilBERT?**")
        st.markdown('<div class="info-box"><p>A logistic regression is trained on the [CLS] token vector from each of DistilBERT\'s 6 layers. Probe accuracy reveals how much sentiment information is encoded at each layer.</p></div>', unsafe_allow_html=True)
        colors = [PAL["purple"]] * 5 + [PAL["amber"]]
        fig = go.Figure(go.Bar(
            x=probe_df["layer"], y=probe_df["accuracy"],
            marker_color=colors,
            text=[f"{v:.1%}" for v in probe_df["accuracy"]],
            textposition="outside",
            textfont=dict(family="Times New Roman", color="#1A1A1A")
        ))
        fig.add_annotation(x=6, y=0.65, text="Peak", showarrow=True, arrowhead=2,
            arrowcolor=PAL["amber"], font=dict(color=PAL["amber"], family="Times New Roman"))
        fig.update_layout(**base_fig("Figure 2: Probe Accuracy by Layer"))
        fig.update_layout(yaxis_tickformat=".0%",
            xaxis=dict(tickvals=list(range(1,7)), title="Layer",
                tickfont=dict(family="Times New Roman", color="#1A1A1A")))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Finding:** Accuracy jumps from 38.8% at layer 5 to 60% at layer 6. DistilBERT's compression forces sentiment understanding to the final layer, unlike larger models where it peaks in the middle.")

    with t3:
        st.markdown("**Do the explanations hold up under masking?**")
        sst_m = masking_df[masking_df["dataset"]=="sst2"]
        imdb_m = masking_df[masking_df["dataset"]=="imdb"]
        sst_g = gemini_df[gemini_df["dataset"]=="sst2"]
        imdb_g = gemini_df[gemini_df["dataset"]=="imdb"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("SST-2","IMDb"))
        for i, (sub_m, sub_g, label) in enumerate([(sst_m, sst_g,"SST-2"),(imdb_m, imdb_g,"IMDb")]):
            col = i+1
            for name, sub, color in [("DistilBERT top-K", sub_m, PAL["purple"]),
                                      ("Gemini top-K", sub_g, PAL["coral"])]:
                fig.add_trace(go.Bar(name=name, x=[name.split()[0]],
                    y=[sub["topk_flipped"].mean()], marker_color=color,
                    showlegend=(i==0),
                    textfont=dict(family="Times New Roman")), row=1, col=col)
            for name, sub in [("DistilBERT random", sub_m),("Gemini random", sub_g)]:
                fig.add_trace(go.Bar(name=name, x=[name.split()[0]],
                    y=[sub["random_flipped"].mean()], marker_color="#CCCCCC",
                    showlegend=False), row=1, col=col)
        fig.update_layout(**base_fig("Figure 3: Faithfulness Gap, Top-K vs Random Masking", h=420))
        fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis2_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**SST-2:** Gemini gap 0.415, DistilBERT gap 0.250. Explicit stated reasoning outperforms attention on short text.")
        with c2:
            st.markdown("**IMDb:** Both gaps near zero (0.035 to 0.050). Long reviews dilute the signal completely regardless of model.")

    with t4:
        st.markdown("**Distribution of RTS scores across all 400 sentences**")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("SST-2","IMDb"))
        for i, (dataset, color) in enumerate([("sst2", PAL["teal"]),("imdb", PAL["amber"])]):
            sub = analysis_df[analysis_df["dataset"]==dataset]
            fig.add_trace(go.Histogram(x=sub["rts"], nbinsx=30, marker_color=color,
                opacity=0.85, name=dataset.upper(), showlegend=True), row=1, col=i+1)
            fig.add_vline(x=sub["rts"].mean(), line_dash="dash", line_color=PAL["coral"],
                annotation_text=f"Mean={sub['rts'].mean():.4f}",
                annotation_font=dict(family="Times New Roman", color=PAL["coral"]),
                row=1, col=i+1)
        fig.update_layout(**base_fig("Figure 4: RTS Score Distribution", h=400))
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("SST-2 mean RTS", "0.0305", "Long tail of high-scoring sentences")
        c2.metric("IMDb mean RTS", "0.0012", "Nearly all sentences cluster at zero")
        st.markdown("SST-2 is **25 times** more faithfully explainable than IMDb on average.")

    with t5:
        st.markdown("**When both models fail -- and when only one does**")
        fig = make_subplots(rows=1, cols=2, specs=[[{"type":"pie"},{"type":"pie"}]],
                            subplot_titles=("SST-2","IMDb"))
        for i, dataset in enumerate(["sst2","imdb"]):
            sub = analysis_df[analysis_df["dataset"]==dataset]
            both   = sub["both_failed"].sum()
            only_d = sub["only_db_failed"].sum()
            only_g = sub["only_gm_failed"].sum()
            neither = len(sub) - both - only_d - only_g
            fig.add_trace(go.Pie(
                labels=["Both failed","Only DistilBERT","Only Gemini","Neither"],
                values=[both, only_d, only_g, neither],
                marker_colors=[PAL["coral"], PAL["purple"], PAL["amber"], PAL["teal"]],
                textfont=dict(family="Times New Roman"), hole=0.3
            ), row=1, col=i+1)
        fig.update_layout(**base_fig("Figure 5: Cross-Model Failure Analysis", h=430))
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("IMDb: both fail", "91.5%", "Near total failure on long text")
        c2.metric("SST-2: both fail", "44.5%", "More variation on short text")
        c3.metric("SST-2: only DistilBERT fails", "30.5%", "Gemini more robust on short text")
        st.markdown('<div class="info-box"><p>On IMDb the pie is almost entirely red -- 91.5% of sentences cause both models to fail. On SST-2 the distribution is much more varied, with Gemini\'s explicit reasoning outperforming DistilBERT\'s attention in 30.5% of cases.</p></div>', unsafe_allow_html=True)

    with t6:
        st.markdown("**Why does the model classify this way? Faithfulness gap by word category**")
        cat = analysis_df.groupby("dominant_category").agg(
            mean_fg=("fg","mean"), count=("rts","count")
        ).sort_values("mean_fg", ascending=True).reset_index()
        bar_colors = [PAL["coral"] if v < 0.15 else PAL["amber"] if v < 0.25
                      else PAL["teal"] for v in cat["mean_fg"]]
        fig = go.Figure(go.Bar(
            x=cat["mean_fg"], y=cat["dominant_category"],
            orientation="h", marker_color=bar_colors,
            text=[f"n={c}" for c in cat["count"]],
            textposition="outside",
            textfont=dict(family="Times New Roman", color="#1A1A1A")
        ))
        fig.add_vline(x=0.20, line_dash="dash", line_color="#999",
            annotation_text="0.20 threshold",
            annotation_font=dict(family="Times New Roman", color="#555"))
        fig.update_layout(**base_fig("Figure 6: Faithfulness Gap by Word Category", h=380))
        fig.update_layout(xaxis_title="Mean Faithfulness Gap")
        st.plotly_chart(fig, use_container_width=True)
        findings = [
            ("Positional (FG = 0.375)", "teal", "Model attends to first or last token. High flip rate but a structural artifact, not semantic understanding. Faithful but for the wrong reason."),
            ("Negation (FG = 0.313)", "teal", "Words like 'not' and 'never', when flagged, do affect prediction. But models handle negation inconsistently."),
            ("Intensifiers (FG = 0.000)", "coral", "Zero faithfulness gap. Completely decorative. Masking 'very', 'absolutely', 'extremely' changes the prediction zero percent of the time. Direct evidence of post-hoc fabrication."),
        ]
        for label, color, desc in findings:
            css = "b-teal" if color=="teal" else "b-coral"
            st.markdown(f"""
            <div class="sent-card" style="border-left-color:{'#00B4A6' if color=='teal' else '#B03030'}">
            <strong style="color:#0D1B2A;">{label}</strong><br>
            <span style="font-size:0.9rem;color:#2A2A2A;">{desc}</span>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SENTENCE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Sentence Explorer":
    st.markdown('<div class="section-eyebrow">Browse the data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Sentence Explorer</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ds_f = st.selectbox("Dataset", ["All","SST-2","IMDb"])
    with c2:
        cat_f = st.selectbox("Word Category", ["All","sentiment","negation","intensifier","positional","other"])
    with c3:
        fail_f = st.selectbox("Failure Mode", ["All","Both failed","Only DistilBERT","Only Gemini","Neither"])
    with c4:
        sort_f = st.selectbox("Sort by", ["RTS low to high","RTS high to low","Faithfulness Gap","LQ Score"])

    max_rts = float(analysis_df["rts"].max())
    rts_range = st.slider("RTS range", 0.0, max_rts, (0.0, max_rts), step=0.001)

    df = analysis_df.copy()
    if ds_f != "All":
        df = df[df["dataset"]==("sst2" if ds_f=="SST-2" else "imdb")]
    if cat_f != "All":
        df = df[df["dominant_category"]==cat_f]
    if fail_f == "Both failed":
        df = df[df["both_failed"]==1]
    elif fail_f == "Only DistilBERT":
        df = df[df["only_db_failed"]==1]
    elif fail_f == "Only Gemini":
        df = df[df["only_gm_failed"]==1]
    elif fail_f == "Neither":
        df = df[(df["both_failed"]==0)&(df["only_db_failed"]==0)&(df["only_gm_failed"]==0)]
    df = df[(df["rts"]>=rts_range[0])&(df["rts"]<=rts_range[1])]
    if sort_f == "RTS low to high":
        df = df.sort_values("rts")
    elif sort_f == "RTS high to low":
        df = df.sort_values("rts", ascending=False)
    elif sort_f == "Faithfulness Gap":
        df = df.sort_values("fg", ascending=False)
    else:
        df = df.sort_values("lq", ascending=False)

    st.markdown(f"**{len(df)} sentences** match your filters")

    cat_cls = {"sentiment":"b-teal","negation":"b-purple","intensifier":"b-coral",
               "positional":"b-amber","other":"b-gray"}

    for _, row in df.head(25).iterrows():
        rts_pct = min(row["rts"]/0.45*100, 100)
        bar_color = "#00B4A6" if row["rts"]>0.1 else "#B5720A" if row["rts"]>0.02 else "#B03030"
        fail_badge = ("b-coral", "Both failed") if row["both_failed"] else \
                     ("b-purple","Only DistilBERT") if row["only_db_failed"] else \
                     ("b-amber","Only Gemini") if row["only_gm_failed"] else \
                     ("b-teal","Neither")
        try:
            dw = ast.literal_eval(row["distilbert_words"])
            gw = ast.literal_eval(row["gemini_words"])
            db_str = ", ".join(str(w) for w in dw[:3])
            gm_str = ", ".join(str(w) for w in gw[:3]) if isinstance(gw, list) else str(gw)
        except:
            db_str = str(row.get("distilbert_words",""))
            gm_str = str(row.get("gemini_words",""))
        ds_label = "SST-2" if row["dataset"]=="sst2" else "IMDb"
        cc = cat_cls.get(row["dominant_category"],"b-gray")
        st.markdown(f"""
        <div class="sent-card">
          <div class="sent-text">"{row['text'][:200]}{'...' if len(str(row['text']))>200 else ''}"</div>
          <div class="badge-row">
            <span class="badge b-gray">{ds_label}</span>
            <span class="badge {cc}">{row['dominant_category']}</span>
            <span class="badge {fail_badge[0]}">{fail_badge[1]}</span>
            <span class="badge b-teal">RTS {row['rts']:.4f}</span>
            <span class="badge b-gray">FG {row['fg']:.3f}</span>
            <span class="badge b-gray">CMA {row['cma']:.3f}</span>
          </div>
          <div style="font-size:0.78rem;color:#6A604E;margin-top:0.4rem;">
            DistilBERT: <em>{db_str}</em> &nbsp;|&nbsp; Gemini: <em>{gm_str}</em>
          </div>
          <div class="rts-bar-bg">
            <div class="rts-bar-fill" style="width:{rts_pct:.1f}%;background:{bar_color};"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    if len(df) > 25:
        st.markdown(f"Showing first 25 of {len(df)} results. Use filters to narrow down.")

# ═══════════════════════════════════════════════════════════════════════════
# LIVE SCORER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Live Scorer":
    st.markdown('<div class="section-eyebrow">Try it yourself</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Live RTS Scorer</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box"><p>This scorer uses pre-computed lookup from the 400 sentences in the dataset. Enter a sentence (or part of one) and if it matches, you will see the full RTS breakdown. Use the dropdown to load example sentences.</p></div>', unsafe_allow_html=True)

    examples = {
        "High RTS: leaves a lot to be desired":    "leaves a lot to be desired",
        "High RTS: appear foolish and shallow":    "appear foolish and shallow",
        "High RTS: massive cardiac arrest":        "massive cardiac arrest",
        "Low RTS: long and eventful":              "long and eventful",
        "Low RTS: incisive and sensitive":         "incisive and sensitive",
        "IMDb example: higher learning review":    "higher learning is a slap in the face",
    }

    c1, c2 = st.columns([3,1])
    with c1:
        user_input = st.text_input("Search sentence", placeholder="e.g. leaves a lot to be desired",
                                   label_visibility="visible")
    with c2:
        ex = st.selectbox("Load example", ["-- choose --"] + list(examples.keys()),
                          label_visibility="visible")
    if ex != "-- choose --":
        user_input = examples[ex]

    if user_input and len(user_input.strip()) > 3:
        search = user_input.strip().lower()[:30]
        matches = analysis_df[analysis_df["text"].str.lower().str.contains(search, na=False)]
        if len(matches) == 0:
            matches = analysis_df[analysis_df["text"].str.lower().str.startswith(search[:12], na=False)]

        if len(matches) > 0:
            row = matches.iloc[0]
            rts_val = row["rts"]
            rts_color = "#00B4A6" if rts_val > 0.1 else "#B5720A" if rts_val > 0.02 else "#B03030"
            verdict = "Faithful" if rts_val > 0.1 else "Partially faithful" if rts_val > 0.02 else "Unfaithful"

            st.markdown(f"""
            <div class="score-box">
              <div style="font-size:0.75rem;color:#8BAAB8;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">
                Reasoning Trustworthiness Score
              </div>
              <div class="score-number" style="color:{rts_color};">{rts_val:.4f}</div>
              <div class="score-label">{verdict}</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("FG", f"{row['fg']:.4f}", "Faithfulness gap")
            c2.metric("LQ", f"{row['lq']:.4f}", "Linguistic quality")
            c3.metric("CMA", f"{row['cma']:.4f}", "Cross-model agreement")
            c4.metric("Confidence", f"{row['confidence']:.4f}", "Score reliability")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Sentence**")
                st.markdown(f"*\"{str(row['text'])[:300]}\"*")
                st.markdown(f"Dataset: **{'SST-2' if row['dataset']=='sst2' else 'IMDb'}**")
                st.markdown(f"Dominant word category: **{row['dominant_category']}**")
            with c2:
                st.markdown("**What each model flagged**")
                try:
                    dw = ast.literal_eval(row["distilbert_words"])
                    gw = ast.literal_eval(row["gemini_words"])
                    overlap = set(w.lower().strip().replace("##","") for w in dw) & \
                              set(w.lower().strip() for w in gw if isinstance(gw, list))
                except:
                    dw, gw, overlap = [], [], set()
                st.markdown(f"DistilBERT attention: **{', '.join(str(w) for w in dw[:3])}**")
                st.markdown(f"Gemini stated: **{', '.join(str(w) for w in gw[:3]) if isinstance(gw,list) else str(gw)}**")
                st.markdown(f"Overlap: **{', '.join(overlap) if overlap else 'None'}**")

            st.markdown("---")
            st.markdown("**Failure mode**")
            if row["both_failed"]:
                st.error("Both models failed. Neither explanation survives counterfactual removal.")
            elif row["only_db_failed"]:
                st.warning("Only DistilBERT failed. Gemini's stated reasoning held up on this sentence.")
            elif row["only_gm_failed"]:
                st.warning("Only Gemini failed. DistilBERT's attention was more causally grounded here.")
            else:
                st.success("Neither model failed. Both explanations survive the masking test.")

            # radar
            fig = go.Figure(go.Scatterpolar(
                r=[row["fg"], row["lq"], row["cma"], row["confidence"], row["fg"]],
                theta=["FG","LQ","CMA","Confidence","FG"],
                fill="toself", marker_color=PAL["teal"],
                line_color=PAL["teal"], fillcolor=PAL["teal"], opacity=0.3
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0,1],
                    tickfont=dict(family="Times New Roman", color="#1A1A1A")),
                    angularaxis=dict(tickfont=dict(family="Times New Roman", color="#1A1A1A"))),
                paper_bgcolor=PAL["bg"], plot_bgcolor=PAL["bg"],
                font=dict(family="Times New Roman", color="#1A1A1A"),
                height=300, margin=dict(l=40,r=40,t=30,b=30), showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No matching sentence found. Try one of the example sentences from the dropdown.")
            st.markdown("**Top scoring sentences to browse:**")
            for _, r in analysis_df.nlargest(5,"rts").iterrows():
                st.markdown(f"""
                <div class="sent-card">
                  <div class="sent-text">"{str(r['text'])[:150]}"</div>
                  <div class="badge-row">
                    <span class="badge b-gray">{'SST-2' if r['dataset']=='sst2' else 'IMDb'}</span>
                    <span class="badge b-teal">RTS {r['rts']:.4f}</span>
                    <span class="badge b-gray">{r['dominant_category']}</span>
                  </div>
                </div>""", unsafe_allow_html=True)