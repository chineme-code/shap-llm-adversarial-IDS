"""
app.py  -  SOC Alert Explainability Dashboard (Streamlit)

WHAT THIS DOES
--------------
Presents each network alert across three panels so a non-expert analyst can act
on it: (1) the classification with a colour-coded severity badge, (2) a SHAP bar
chart of the top contributing features, and (3) a plain-language analyst summary.

WHY IT EXISTS
-------------
A raw "malicious, 94% confidence" verdict is not actionable for an SME's single
IT generalist. This dashboard is the delivery surface for the three-layer
pipeline (XGBoost detect -> SHAP explain -> Claude translate), turning a verdict
into a reason plus a recommended action.

TWO MODES (sidebar)
-------------------
  - Pre-generated: reads cached Claude explanations from results/. Instant, free.
  - Live API:      calls the Claude API on each selection. Fresh, costs credits.

RUN IT (from the project root, not from inside dashboard/):
    streamlit run dashboard/app.py

PREREQUISITES (produced by notebooks/ or src/run_ids2025.py):
    models/xgboost_ids2025.pkl
    models/class_names.json
    data/X_test_sample.csv
    results/pregenerated_explanations.json   (optional; only for pre-generated mode)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import shap
import joblib
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import anthropic

load_dotenv(".env")

st.set_page_config(page_title="SOC Alert Explainability Dashboard", layout="wide")

# Severity colour per class (green = benign, deepening red = more severe)
ATTACK_COLORS = {
    "Normal": "#1a7f37",
    "PortScan": "#d97706",
    "Web Attack": "#dc2626",
    "Brute Force": "#dc2626",
    "Dos/DDos": "#b91c1c",
    "Botnet ARES": "#7f1d1d",
    "Infiltration": "#7f1d1d",
}


# ---- Loaders (cached so they run once per session) -------------------
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_ids2025.pkl")


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_data():
    return pd.read_csv("data/X_test_sample.csv")


@st.cache_data
def load_class_names():
    with open("models/class_names.json") as f:
        return json.load(f)


@st.cache_data
def load_pregenerated():
    try:
        with open("results/pregenerated_explanations.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def get_client():
    key = os.getenv("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else None


@st.cache_data(show_spinner=False)
def generate_live_explanation(classification, top_features_str):
    """Call Claude live; cached by (class, features) so re-clicks are free."""
    client = get_client()
    if client is None:
        return ("No API key found. Add ANTHROPIC_API_KEY to your .env file, "
                "or switch to Pre-generated mode.")
    prompt = f"""You are a SOC analyst assistant. Given this network traffic alert
classification and its top contributing features, write a SHORT plain-text
explanation for a dashboard card. No markdown, no headers, no bullet points.

Write exactly 3 sentences:
1. Why this alert was classified this way (plain English, reference the top feature)
2. One specific, concrete action the analyst should take right now
3. One sentence of risk context for a small/medium organization

Classification: {classification}
Top contributing features (SHAP values, positive = pushed toward this classification):
{top_features_str}
"""
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# ---- Load everything -------------------------------------------------
model = load_model()
explainer = load_explainer(model)
X_test = load_data()
class_names = load_class_names()
pregenerated = load_pregenerated()

# ---- Sidebar controls ------------------------------------------------
st.sidebar.title("Controls")
mode = st.sidebar.radio(
    "Explanation mode",
    ["Pre-generated (fast, free)", "Live API (fresh, costs credits)"],
)
sample_idx = st.sidebar.selectbox("Select an alert to review", range(len(X_test)))
st.sidebar.markdown("---")
st.sidebar.caption(
    "Pre-generated reads cached Claude output from results/. "
    "Live calls the Claude API on each selection."
)

# ---- Compute for the selected alert ----------------------------------
st.title("SOC Alert Explainability Dashboard")
st.caption("XGBoost detection - SHAP attribution - Claude plain-English translation")

row = X_test.iloc[[sample_idx]]
predicted_class = int(model.predict(row)[0])
class_label = class_names[predicted_class]

shap_values = explainer.shap_values(row)          # shape (1, n_features, n_classes)
sample_shap = shap_values[0, :, predicted_class]
top_idx = np.argsort(np.abs(sample_shap))[::-1][:5]
top_features = [(X_test.columns[i], float(sample_shap[i])) for i in top_idx]
top_features_str = "\n".join([f"- {f}: {v:.4f}" for f, v in top_features])

# ---- Three-panel layout ----------------------------------------------
col1, col2, col3 = st.columns([1, 1.3, 1.6])

with col1:
    st.subheader("Classification")
    color = ATTACK_COLORS.get(class_label, "#dc2626")
    st.markdown(
        f"<div style='background:{color};color:white;padding:14px 18px;"
        f"border-radius:8px;font-size:20px;font-weight:600;text-align:center'>"
        f"{class_label}</div>",
        unsafe_allow_html=True,
    )
    st.metric("Alert index", sample_idx)

with col2:
    st.subheader("Top Contributing Features")
    feats = [f for f, _ in top_features][::-1]
    vals = [v for _, v in top_features][::-1]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    bar_colors = ["#dc2626" if v > 0 else "#2563eb" for v in vals]
    ax.barh(feats, vals, color=bar_colors)
    ax.set_xlabel("SHAP value")
    ax.axvline(0, color="#666", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Red = pushed toward this class, blue = pushed away.")

with col3:
    st.subheader("Analyst Summary")
    if mode.startswith("Pre-generated"):
        entry = pregenerated.get(str(sample_idx))
        if entry:
            st.info(entry["explanation"])
        else:
            st.warning("No pre-generated explanation for this alert. "
                       "Run the batch generation step, or switch to Live API mode.")
    else:
        with st.spinner("Asking Claude..."):
            explanation = generate_live_explanation(class_label, top_features_str)
        st.success(explanation)
