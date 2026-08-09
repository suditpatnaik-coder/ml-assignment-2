from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from utils import calculate_metrics

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "target"

st.set_page_config(page_title="DigitLens ML Classifier", page_icon="🔢", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}
    metrics = pd.read_csv(MODEL_DIR / "metrics.csv").set_index("Model")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    return models, metrics, metadata

@st.cache_data
def load_default_test_data():
    return pd.read_csv(ROOT / "test_data.csv")

models, reference_metrics, metadata = load_artifacts()
feature_names = metadata["feature_names"]

st.title("🔢 DigitLens: Handwritten Digit Classification")
st.caption(
    "Five classical ML classifiers evaluated on 64 pixel-intensity features. "
    "Upload the provided test_data.csv or a compatible CSV to reproduce the evaluation."
)

with st.sidebar:
    st.header("Evaluation controls")
    selected_model = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
    st.caption("Expected columns: 64 pixel features and, for evaluation, a target column named 'target'.")
    st.divider()
    st.write(f"**Dataset rows used:** {metadata['implementation_instances']}")
    st.write(f"**Features:** {metadata['features']}")
    st.write("**Classes:** 0-9")

if uploaded_file is None:
    data = load_default_test_data()
    st.info("Using the bundled test_data.csv. Upload another compatible CSV to evaluate your own test split.")
else:
    data = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(data):,} rows from the uploaded CSV.")

missing_features = [c for c in feature_names if c not in data.columns]
if missing_features:
    st.error(f"The uploaded CSV is missing {len(missing_features)} required feature columns. First missing columns: {missing_features[:8]}")
    st.stop()

X = data[feature_names]
model = models[selected_model]
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

summary_tab, compare_tab, explorer_tab, guide_tab = st.tabs(
    ["Selected model", "Compare models", "Prediction explorer", "Dataset guide"]
)

with summary_tab:
    st.subheader(selected_model)
    if TARGET in data.columns:
        y_true = data[TARGET].astype(int)
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        for start in (0, 3):
            cols = st.columns(3)
            for col, metric_name in zip(cols, metric_names[start:start+3]):
                col.metric(metric_name, f"{metrics[metric_name]:.4f}")

        st.markdown("#### Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{selected_model} - Confusion Matrix")
        st.pyplot(fig, clear_figure=True)

        st.markdown("#### Classification report")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(4), use_container_width=True)
    else:
        st.warning("No target column found, so evaluation metrics cannot be calculated. Predictions are still available below.")

    result_preview = X.copy()
    result_preview["predicted_digit"] = y_pred
    if TARGET in data.columns:
        result_preview["actual_digit"] = data[TARGET].astype(int).to_numpy()
        result_preview["correct"] = result_preview["predicted_digit"] == result_preview["actual_digit"]
    st.markdown("#### Prediction sample")
    st.dataframe(result_preview.head(20), use_container_width=True)

with compare_tab:
    st.subheader("Reference comparison on bundled test split")
    display = reference_metrics[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].copy()
    st.dataframe(display.round(4).sort_values("F1", ascending=False), use_container_width=True)
    best_model = display["F1"].idxmax()
    st.success(f"Overall winner on this split: {best_model} (highest weighted F1).")

    if TARGET in data.columns:
        st.markdown("#### Re-evaluate all models on the currently loaded CSV")
        live_rows = []
        y_true = data[TARGET].astype(int)
        for model_name, candidate in models.items():
            p = candidate.predict(X)
            pp = candidate.predict_proba(X)
            row = {"Model": model_name, **calculate_metrics(y_true, p, pp)}
            live_rows.append(row)
        live_df = pd.DataFrame(live_rows).set_index("Model")
        st.dataframe(live_df.round(4).sort_values("F1", ascending=False), use_container_width=True)

with explorer_tab:
    st.subheader("Inspect one 8 × 8 digit")
    idx = st.number_input("Row index", min_value=0, max_value=max(len(data)-1, 0), value=0, step=1)
    row = X.iloc[int(idx)].to_numpy().reshape(8, 8)
    left, right = st.columns([1, 2])
    with left:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(row, cmap="gray_r", interpolation="nearest")
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)
    with right:
        st.metric("Predicted digit", int(y_pred[int(idx)]))
        if TARGET in data.columns:
            actual = int(data.iloc[int(idx)][TARGET])
            st.metric("Actual digit", actual)
            st.write("✅ Correct" if actual == int(y_pred[int(idx)]) else "❌ Misclassified")
        prob_df = pd.DataFrame({"Digit": list(range(10)), "Probability": y_proba[int(idx)]})
        st.bar_chart(prob_df.set_index("Digit"))

with guide_tab:
    st.subheader("Dataset and metric choices")
    st.write(
        "The implementation uses 1,797 samples from scikit-learn's bundled handwritten-digits data, "
        "which corresponds to the UCI Optical Recognition of Handwritten Digits test portion. "
        "Each sample has 64 integer pixel-intensity features arranged as an 8 × 8 image, with values 0-16."
    )
    st.write(
        "For this multiclass problem, Precision, Recall and F1 are calculated using weighted averaging. "
        "AUC is calculated as weighted one-vs-rest (OvR). MCC is the multiclass Matthews correlation coefficient."
    )
    st.code("Required target column: target\nRequired feature columns: pixel_0_0 ... pixel_7_7", language="text")
