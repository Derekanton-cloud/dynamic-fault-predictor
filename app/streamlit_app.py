import glob
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

st.set_page_config(layout="wide")
st.title("🏢 Software Fault Intelligence Platform")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
METADATA_DIR = os.path.join(BASE_DIR, "artifacts", "metadata")
SCALER_DIR = os.path.join(BASE_DIR, "artifacts", "scalers")


def resolve_artifact_path(relative_path):
    """Convert metadata paths into absolute paths within the project."""

    if not relative_path:
        return None

    normalized = relative_path.replace("\\", os.sep)

    if os.path.isabs(normalized):
        return normalized

    return os.path.join(BASE_DIR, normalized)

# -----------------------------------------------------
# LOAD METADATA FILES
# -----------------------------------------------------
def load_all_metadata():
    files = [f for f in os.listdir(METADATA_DIR) if f.endswith(".json")]
    all_meta = []

    for file in files:
        with open(os.path.join(METADATA_DIR, file)) as f:
            meta = json.load(f)
            meta["file"] = file
            all_meta.append(meta)

    return pd.DataFrame(all_meta)

# -----------------------------------------------------
# DRIFT DETECTION
# -----------------------------------------------------
def detect_drift(train_means, new_means, threshold=0.2):
    drift_features = []
    for col in train_means.index:
        shift = abs(train_means[col] - new_means[col])
        if shift > threshold:
            drift_features.append(col)
    return drift_features

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
st.sidebar.header("⚙ Model Configuration")

model_choice_label = st.sidebar.selectbox(
    "Select Model",
    ["Production (CNN)", "Dynamic (CNN)", "Hybrid (CNN+XGBoost)"]
)

model_choice_map = {
    "Production (CNN)": "Production",
    "Dynamic (CNN)": "Dynamic",
    "Hybrid (CNN+XGBoost)": "Hybrid"
}
model_choice = model_choice_map.get(model_choice_label)

if model_choice is None:
    st.error(f"Unsupported model selection: {model_choice_label}")
    st.stop()

threshold_placeholder = st.sidebar.empty()
threshold_slider_placeholder = st.sidebar.empty()

optimization_goal = st.sidebar.radio(
    "Optimization Strategy",
    ["Max Recall", "Max Precision", "Balanced (F1)"]
)

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

# -----------------------------------------------------
# EXECUTIVE SUMMARY PANEL
# -----------------------------------------------------
st.subheader("📊 Executive Summary")

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    y = np.where(df["defects"] > 0, 1, 0)

    dataset_size = len(df)
    fault_rate = np.mean(y)

    col1, col2 = st.columns(2)
    col1.metric("Dataset Size", dataset_size)
    col2.metric("Fault Rate", f"{fault_rate:.2%}")

    st.markdown("---")

    # -------------------------------------------------
    # MODEL LOADING + INFERENCE
    # -------------------------------------------------
    feature_columns = [col for col in df.columns if col != "defects"]
    if not feature_columns:
        st.error("Dataset must include a 'defects' column plus input features.")
        st.stop()

    X_features = df[feature_columns].copy()
    X_features = X_features.apply(pd.to_numeric, errors="coerce")
    X_features = X_features.fillna(X_features.median())
    y_prob = None
    base_threshold = 0.5
    active_metadata = {}

    if model_choice == "Production":
        results_path = os.path.join(METADATA_DIR, "production_model_results.json")

        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                active_metadata = json.load(f)

        scaler_path = os.path.join(SCALER_DIR, "scaler.pkl")

        production_model_candidates = [
            os.path.join(MODEL_DIR, "production", "cnn_model.keras"),
            os.path.join(MODEL_DIR, "production", "fault_predictor.keras")
        ]
        production_model_path = next(
            (path for path in production_model_candidates if os.path.exists(path)),
            None
        )

        if production_model_path is None or not os.path.exists(scaler_path):
            st.error("Required production CNN artifacts are missing. Train the production CNN model first.")
            st.stop()

        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_features)
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        production_model = load_model(production_model_path)
        y_prob = production_model.predict(X_cnn, verbose=0).ravel()
        base_threshold = active_metadata.get("threshold", 0.5)

    elif model_choice == "Hybrid":
        metadata_path = os.path.join(METADATA_DIR, "production_metadata.json")
        if not os.path.exists(metadata_path):
            st.error("Hybrid metadata not found. Train the hybrid production model first.")
            st.stop()

        with open(metadata_path, "r") as f:
            active_metadata = json.load(f)

        scaler_path = resolve_artifact_path(
            active_metadata.get("scaler_path", os.path.join("artifacts", "scalers", "scaler.pkl"))
        )
        cnn_path = resolve_artifact_path(
            active_metadata.get("cnn_model_path", os.path.join("models", "production", "cnn_model.keras"))
        )
        xgb_path = resolve_artifact_path(
            active_metadata.get("xgb_model_path", os.path.join("models", "production", "xgb_model.pkl"))
        )

        missing_artifact = next(
            (path for path in (scaler_path, cnn_path, xgb_path) if path is None or not os.path.exists(path)),
            None
        )

        if missing_artifact:
            st.error("Required hybrid artifacts are missing. Retrain the hybrid production model.")
            st.stop()

        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_features)
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        cnn_model = load_model(cnn_path)
        xgb_model = joblib.load(xgb_path)

        cnn_prob = cnn_model.predict(X_cnn, verbose=0).ravel()
        xgb_prob = xgb_model.predict_proba(X_scaled)[:, 1]

        cnn_weight = active_metadata.get("cnn_weight", 0.6)
        xgb_weight = active_metadata.get("xgb_weight", 0.4)

        y_prob = cnn_weight * cnn_prob + xgb_weight * xgb_prob
        base_threshold = active_metadata.get("threshold", 0.5)

    else:
        dynamic_meta_files = glob.glob(os.path.join(METADATA_DIR, "dynamic_metadata_*.json"))
        if not dynamic_meta_files:
            st.error("No dynamic model metadata found. Retrain a dynamic model first.")
            st.stop()

        latest_meta_path = max(dynamic_meta_files, key=os.path.getctime)
        with open(latest_meta_path, "r") as f:
            active_metadata = json.load(f)

        timestamp = active_metadata.get("timestamp")
        if not timestamp:
            st.error("Dynamic metadata is missing the timestamp identifier.")
            st.stop()

        model_path = os.path.join(MODEL_DIR, "experiments", f"dynamic_model_{timestamp}.keras")
        scaler_candidates = [
            os.path.join(SCALER_DIR, f"dynamic_scaler_{timestamp}.pkl"),
            os.path.join(SCALER_DIR, f"dynamic_model_{timestamp}.pkl")
        ]
        scaler_path = next((path for path in scaler_candidates if os.path.exists(path)), None)

        if not os.path.exists(model_path) or scaler_path is None:
            st.error("Dynamic model artifacts are incomplete. Retrain the dynamic model.")
            st.stop()

        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_features)
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        dynamic_model = load_model(model_path)
        y_prob = dynamic_model.predict(X_cnn, verbose=0).ravel()
        base_threshold = active_metadata.get("threshold", 0.5)

    if y_prob is None:
        st.error("Unable to compute prediction probabilities for the uploaded dataset.")
        st.stop()

    # -------------------------------------------------
    # THRESHOLD STRATEGY
    # -------------------------------------------------
    if model_choice == "Production":
        best_t = float(base_threshold)
        threshold_placeholder.success(f"Using Fixed Production Threshold: {best_t:.2f}")
    else:
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_t = float(base_threshold)
        best_metric = -1.0

        for t in thresholds:
            y_pred_tmp = (y_prob >= t).astype(int)

            if optimization_goal == "Max Recall":
                metric = recall_score(y, y_pred_tmp)
            elif optimization_goal == "Max Precision":
                metric = precision_score(y, y_pred_tmp, zero_division=0)
            else:
                metric = f1_score(y, y_pred_tmp, zero_division=0)

            if metric > best_metric:
                best_metric = metric
                best_t = t

        threshold_placeholder.success(f"Suggested Threshold: {best_t:.2f}")

    threshold = threshold_slider_placeholder.slider(
        "Adjust Threshold", 0.05, 0.95, float(best_t), 0.01
    )

    y_pred = (y_prob >= threshold).astype(int)

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    acc = accuracy_score(y, y_pred)
    rec = recall_score(y, y_pred)
    prec = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Recall", f"{rec:.4f}")
    col3.metric("Precision", f"{prec:.4f}")
    col4.metric("F1", f"{f1:.4f}")

    # -------------------------------------------------
    # VISUAL TABS
    # -------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Drift Detection"])

    with tab1:
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    with tab2:
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        ax.plot([0,1],[0,1],'--')
        ax.legend()
        st.pyplot(fig)

    with tab3:
        baseline_df = pd.read_csv("data/jm1.csv")
        train_mean = baseline_df.drop(columns=["defects"]).mean()
        new_mean = X_features.mean()
        drift_cols = detect_drift(train_mean, new_mean)

        if drift_cols:
            st.error("⚠ Data Drift Detected!")
            st.write(drift_cols)
        else:
            st.success("✅ No Significant Drift")

    # -------------------------------------------------
    # DOWNLOAD
    # -------------------------------------------------
    df["Prediction"] = y_pred
    df["Probability"] = y_prob

    st.download_button(
        "Download Prediction Report",
        df.to_csv(index=False),
        "enterprise_fault_predictions.csv"
    )

    # =========================================================
    # 🧠 MODEL PERFORMANCE COMPARISON PANEL
    # =========================================================

    st.markdown("## 📊 Model Performance Comparison")

    def load_production_metadata():
        path = os.path.join(METADATA_DIR, "production_model_results.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def load_latest_dynamic_metadata():
        pattern = os.path.join(METADATA_DIR, "dynamic_metadata_*.json")
        dynamic_files = glob.glob(pattern)
        if dynamic_files:
            latest = max(dynamic_files, key=os.path.getctime)
            with open(latest, "r") as f:
                data = json.load(f)
                data.setdefault("model", "Dynamic CNN")
                return data
        return None

    def load_hybrid_metadata():
        path = os.path.join(METADATA_DIR, "hybrid_production_model_results.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                payload = json.load(f)
                payload.setdefault("model", "Hybrid (CNN+XGBoost)")
                return payload
        return None

    prod_meta = load_production_metadata()
    dyn_meta = load_latest_dynamic_metadata()
    hyb_meta = load_hybrid_metadata()

    def extract_metric(meta, *keys):
        for source in (meta.get("results", {}), meta):
            for key in keys:
                if key in source and source[key] is not None:
                    return source[key]
        return 0

    comparison_data = []
    df_compare = None

    if prod_meta:
        comparison_data.append({
            "Model": "Production (CNN)",
            "Accuracy": extract_metric(prod_meta, "accuracy"),
            "Recall": extract_metric(prod_meta, "recall", "recall_faulty"),
            "Precision": extract_metric(prod_meta, "precision", "precision_faulty"),
            "F1": extract_metric(prod_meta, "f1_score", "f1", "f1_faulty")
        })

    if dyn_meta:
        comparison_data.append({
            "Model": "Dynamic CNN",
            "Accuracy": extract_metric(dyn_meta, "accuracy"),
            "Recall": extract_metric(dyn_meta, "recall", "recall_faulty"),
            "Precision": extract_metric(dyn_meta, "precision", "precision_faulty"),
            "F1": extract_metric(dyn_meta, "f1_score", "f1", "f1_faulty")
        })

    if hyb_meta:
        comparison_data.append({
            "Model": "Hybrid (CNN+XGBoost)",
            "Accuracy": extract_metric(hyb_meta, "accuracy"),
            "Recall": extract_metric(hyb_meta, "recall", "recall_faulty"),
            "Precision": extract_metric(hyb_meta, "precision", "precision_faulty"),
            "F1": extract_metric(hyb_meta, "f1_score", "f1", "f1_faulty")
        })

    if comparison_data:
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare.style.highlight_max(axis=0, color="lightgreen"))

        best_accuracy_model = df_compare.loc[df_compare["Accuracy"].idxmax(), "Model"]
        best_recall_model = df_compare.loc[df_compare["Recall"].idxmax(), "Model"]
        best_f1_model = df_compare.loc[df_compare["F1"].idxmax(), "Model"]

        st.markdown("### 🏆 Performance Insights")
        st.success(f"Best Accuracy: **{best_accuracy_model}**")
        st.success(f"Best Recall (Fault Detection): **{best_recall_model}**")
        st.success(f"Best F1 Score: **{best_f1_model}**")
    else:
        st.info("No model metadata found yet. Train models first.")

    # =========================================================
    # 🤖 AUTO MODEL SELECTION ENGINE
    # =========================================================

    st.markdown("## 🤖 AI Model Intelligence")

    best_model_auto = None
    selection_objective = "Maximize Recall"

    if df_compare is not None:
        selection_objective = st.sidebar.selectbox(
            "🎯 Optimization Goal",
            ["Maximize Recall", "Maximize Accuracy", "Maximize F1"]
        )

        if selection_objective == "Maximize Recall":
            best_model_auto = df_compare.loc[df_compare["Recall"].idxmax(), "Model"]
        elif selection_objective == "Maximize Accuracy":
            best_model_auto = df_compare.loc[df_compare["Accuracy"].idxmax(), "Model"]
        else:
            best_model_auto = df_compare.loc[df_compare["F1"].idxmax(), "Model"]

        st.success(f"🏆 Recommended Model Based on Goal: **{best_model_auto}**")

    # =========================================================
    # 📊 PERFORMANCE GAP ANALYSIS
    # =========================================================

    st.markdown("## 📊 Performance Gap Analysis")

    acc_gap = 0
    recall_gap = 0

    if df_compare is not None:
        max_acc = df_compare["Accuracy"].max()
        min_acc = df_compare["Accuracy"].min()
        max_recall = df_compare["Recall"].max()
        min_recall = df_compare["Recall"].min()

        acc_gap = round(max_acc - min_acc, 4)
        recall_gap = round(max_recall - min_recall, 4)

        st.info(f"🔎 Accuracy Gap Between Best & Worst Model: {acc_gap}")
        st.info(f"🔎 Recall Gap Between Best & Worst Model: {recall_gap}")

    # =========================================================
    # ⚠️ DATA DRIFT DETECTION
    # =========================================================

    st.markdown("## ⚠️ Data Drift Monitor")

    def compute_dataset_drift(upload_df, training_df_sample):
        drift_scores = {}
        for col in upload_df.columns:
            if col in training_df_sample.columns:
                drift = abs(upload_df[col].mean() - training_df_sample[col].mean())
                drift_scores[col] = drift

        avg_drift = np.mean(list(drift_scores.values())) if drift_scores else 0
        return avg_drift

    if uploaded_file is not None:
        try:
            sample_train = pd.read_csv("data/jm1.csv").drop(columns=["defects"])
            sample_upload = df.drop(columns=["defects"])

            drift_score = compute_dataset_drift(sample_upload, sample_train)

            if drift_score > 0.5:
                st.error(f"⚠️ High Dataset Drift Detected (Score: {drift_score:.3f})")
            else:
                st.success(f"✅ Dataset Drift Within Safe Range (Score: {drift_score:.3f})")

        except Exception:
            st.warning("Drift check unavailable for this dataset.")

    # =========================================================
    # 🧠 AI RECOMMENDATION ENGINE
    # =========================================================

    st.markdown("## 🧠 AI Recommendation Engine")

    if df_compare is not None and best_model_auto:
        if best_model_auto == "Hybrid (CNN+XGBoost)":
            recommendation_text = (
                """
        Hybrid model is recommended due to balanced generalization and improved recall.
        Suitable for enterprise environments where missing faulty modules is costly.
        """
            )
        elif best_model_auto == "Dynamic CNN":
            recommendation_text = (
                """
        Dynamic model adapts per dataset.
        Recommended when datasets vary significantly across projects.
        """
            )
        else:
            recommendation_text = (
                """
        Production model is stable and tested.
        Recommended for consistent environments with controlled data distribution.
        """
            )

        st.write(recommendation_text)

    # =========================================================
    # 📄 EXECUTIVE SUMMARY PANEL
    # =========================================================

    st.markdown("## 🧾 AI Executive Summary")

    if df_compare is not None and best_model_auto:
        summary = f"""
    ✔ Selected Objective: {selection_objective}

    ✔ Recommended Model: {best_model_auto}

    ✔ Highest Accuracy Achieved: {round(df_compare['Accuracy'].max(), 4)}

    ✔ Highest Recall Achieved: {round(df_compare['Recall'].max(), 4)}

    ✔ Performance Spread (Accuracy Gap): {acc_gap}

    ✔ System Ready for Production Evaluation.
    """

        st.text_area("Executive Summary Report", summary, height=220)

# -----------------------------------------------------
# EXPERIMENT VIEWER
# -----------------------------------------------------
st.markdown("---")
st.subheader("📂 Experiment History")

if os.path.exists(METADATA_DIR):
    metadata_df = load_all_metadata()
    st.dataframe(metadata_df)
