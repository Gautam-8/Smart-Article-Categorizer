import numpy as np
import streamlit as st
from joblib import load
from embedding_models import BERTEmbedder, GloveEmbedder, SBERTEmbedder, NomicEmbedder
import pandas as pd
import os

st.set_page_config(page_title="Smart Article Categorizer", layout="centered")
st.title("📰 Smart Article Categorizer")
st.markdown("Classify your article into one of 6 categories using different embedding models.")

# ========== 🔮 Predictor Loaders ==========
@st.cache_resource
def load_glove():
    return GloveEmbedder(), load("models/glove_classifier.joblib")

@st.cache_resource
def load_bert():
    return BERTEmbedder(), load("models/bert_classifier.joblib")
    
@st.cache_resource
def load_sbert():
    return SBERTEmbedder(), load("models/sbert_classifier.joblib")

@st.cache_resource
def load_nomic():
    return NomicEmbedder(), load("models/nomic_classifier.joblib")

# ========== 🔍 Predictor Functions ==========
def predict_model(text, embedder, clf):
    vector = embedder.embed(text)
    label = clf.predict([vector])[0]
    confs = clf.predict_proba([vector])[0]
    return label, dict(zip(clf.classes_, np.round(confs, 3)))

# ========== 🎨 Streamlit UI ==========
article_text = st.text_area("✍️ Paste your article text here:", height=150)

if st.button("🔍 Classify"):
    if not article_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        st.subheader("🎯 Predictions")

        with st.spinner("Running GloVe..."):
            glove_embed, glove_clf = load_glove()
            g_label, g_conf = predict_model(article_text, glove_embed, glove_clf)

        with st.spinner("Running BERT..."):
            bert_embed, bert_clf = load_bert()
            b_label, b_conf = predict_model(article_text, bert_embed, bert_clf)

        with st.spinner("Running SBERT..."):
            sbert_embed, sbert_clf = load_sbert()
            s_label, s_conf = predict_model(article_text, sbert_embed, sbert_clf)

        with st.spinner("Running Nomic..."):
            nomic_embed, nomic_clf = load_nomic()
            n_label, n_conf = predict_model(article_text, nomic_embed, nomic_clf)

        # 🎯 Summary Table
        st.markdown("## 🧠 Model Predictions")
        st.table({
            "Model": ["GloVe", "BERT", "SBERT", "Nomic"],
            "Prediction": [g_label, b_label, s_label, n_label]
        })

        # 📊 Confidence Visualization
        st.markdown("## 📊 Confidence Scores")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("#### 🔹 GloVe")
            st.bar_chart(g_conf)
        with col2:
            st.markdown("#### 🔹 BERT")
            st.bar_chart(b_conf)
        with col3:
            st.markdown("#### 🔹 SBERT")
            st.bar_chart(s_conf)
        with col4:
            st.markdown("#### 🔹 Nomic")
            st.bar_chart(n_conf)

        # 📈 Model Evaluation Comparison Table
        st.markdown("## 📈 Evaluation Comparison Table")
        eval_path = "metrics/eval_scores.csv"
        if os.path.exists(eval_path):
            eval_df = pd.read_csv(eval_path)
            st.dataframe(eval_df.set_index("Model").style.format("{:.3f}"))
        else:
            st.warning("⚠️ Evaluation metrics not found. Please run `train.py` first.")
