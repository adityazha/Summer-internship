import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "diabetes_model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"

SMOKING_OPTIONS = {
    "Never": "never",
    "Current": "current",
    "Former": "former",
    "Ever": "ever",
    "Not Current": "not current",
    "No Info": "No Info",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Trained model not found. Run `python train_model.py` first, then reload the app."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def risk_label(probability: float) -> tuple[str, str]:
    if probability < 0.35:
        return "Low Risk", "success"
    if probability < 0.65:
        return "Moderate Risk", "warning"
    return "High Risk", "error"


def build_input_frame(
    gender: str,
    age: float,
    hypertension: str,
    heart_disease: str,
    smoking_history: str,
    bmi: float,
    hba1c_level: float,
    blood_glucose_level: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gender": gender,
                "age": age,
                "hypertension": 1 if hypertension == "Yes" else 0,
                "heart_disease": 1 if heart_disease == "Yes" else 0,
                "smoking_history": SMOKING_OPTIONS[smoking_history],
                "bmi": bmi,
                "HbA1c_level": hba1c_level,
                "blood_glucose_level": blood_glucose_level,
            }
        ]
    )


st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon=":hospital:",
    layout="wide",
)

model = load_model()
metrics = load_metrics()

st.title("Diabetes Prediction System")
st.caption(
    "College ML project | Predict diabetes risk using patient health indicators"
)

if metrics:
    best_metrics = metrics["selected_model_metrics"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", metrics["best_model"])
    col2.metric("Accuracy", f"{best_metrics['accuracy'] * 100:.2f}%")
    col3.metric("F1 Score", f"{best_metrics['f1_score'] * 100:.2f}%")
    col4.metric("ROC-AUC", f"{best_metrics['roc_auc'] * 100:.2f}%")

tab_predict, tab_models, tab_about = st.tabs(
    ["Predict", "Model Analysis", "About Project"]
)

with tab_predict:
    st.subheader("Enter Patient Details")

    left, right = st.columns(2, gap="large")

    with left:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

    with right:
        smoking_history = st.selectbox(
            "Smoking History",
            list(SMOKING_OPTIONS.keys()),
        )
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        hba1c_level = st.number_input(
            "HbA1c Level (%)",
            min_value=3.0,
            max_value=15.0,
            value=5.5,
            step=0.1,
        )
        blood_glucose_level = st.number_input(
            "Blood Glucose Level (mg/dL)",
            min_value=50,
            max_value=400,
            value=120,
        )

    predict = st.button("Predict Diabetes Risk", type="primary", use_container_width=True)

    if predict:
        user_data = build_input_frame(
            gender,
            age,
            hypertension,
            heart_disease,
            smoking_history,
            bmi,
            hba1c_level,
            blood_glucose_level,
        )

        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0][1]
        risk_text, alert_type = risk_label(probability)

        st.divider()
        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            if prediction == 0:
                st.success("Prediction: No Diabetes Detected")
            else:
                st.error("Prediction: Diabetes Detected")
                st.info("Please consult a doctor for proper medical advice.")

        with result_col2:
            st.metric("Diabetes Probability", f"{probability * 100:.2f}%")
            st.progress(min(max(probability, 0.0), 1.0))
            if alert_type == "success":
                st.success(f"Risk Level: {risk_text}")
            elif alert_type == "warning":
                st.warning(f"Risk Level: {risk_text}")
            else:
                st.error(f"Risk Level: {risk_text}")

with tab_models:
    if not metrics:
        st.warning("Run `python train_model.py` to generate model analysis metrics.")
    else:
        st.subheader("Model Comparison")
        comparison_df = pd.DataFrame(metrics["model_comparison"])[
            ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_f1_mean"]
        ]
        comparison_df.columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "CV F1 (Mean)",
        ]
        st.dataframe(
            comparison_df.style.format(
                {
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1 Score": "{:.2%}",
                    "ROC-AUC": "{:.2%}",
                    "CV F1 (Mean)": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        st.subheader("Confusion Matrix (Best Model)")
        matrix = np.array(metrics["selected_model_metrics"]["confusion_matrix"])
        st.write("Rows: Actual | Columns: Predicted")
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual: No Diabetes", "Actual: Diabetes"],
                columns=["Predicted: No", "Predicted: Yes"],
            ),
            use_container_width=True,
        )

        if metrics["feature_importance"]:
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame(metrics["feature_importance"])
            st.bar_chart(importance_df.set_index("feature")["importance"])

with tab_about:
    st.markdown(
        """
        ### Project Overview
        This application predicts whether a person is likely to have diabetes using
        machine learning on health and lifestyle features.

        ### Input Features
        - Gender, Age
        - Hypertension and Heart Disease history
        - Smoking history
        - BMI, HbA1c level, Blood glucose level

        ### ML Pipeline Improvements
        - **Stratified train-test split** for balanced evaluation
        - **Multiple model comparison** (Random Forest, Gradient Boosting, Logistic Regression)
        - **Class imbalance handling** using balanced weights
        - **Cross-validation** for reliable performance
        - **Saved pipeline** with preprocessing + model for consistent predictions

        ### How to Run
        1. Install dependencies: `pip install -r requirements.txt`
        2. Train model: `python train_model.py`
        3. Launch app: `streamlit run app.py`

        ### Disclaimer
        This tool is for educational purposes only and is not a substitute for
        professional medical diagnosis.
        """
    )
