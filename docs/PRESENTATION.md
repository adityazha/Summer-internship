# Diabetes Prediction using Machine Learning
## College Project Presentation Content

Copy each slide section into PowerPoint / Google Slides. Add screenshots from the Streamlit app and EDA notebook charts.

---

## Slide 1: Title Slide

**Title:** Diabetes Prediction using Machine Learning

**Subtitle:** A Data-Driven Approach for Early Risk Detection

**Presented By:** [Your Name]

**Department:** [Your Department]

**College:** [Your College Name]

**Guide:** [Faculty Name]

**Date:** [Presentation Date]

---

## Slide 2: Introduction

**What is Diabetes?**
- A chronic disease where blood glucose (sugar) levels are too high
- Early detection helps prevent serious complications

**Problem Statement**
- Manual screening is time-consuming
- Need a fast, data-driven tool to estimate diabetes risk

**Project Goal**
- Build a Machine Learning model to predict diabetes
- Deploy an easy-to-use web application for predictions

---

## Slide 3: Objectives

1. Collect and analyze patient health data
2. Perform Exploratory Data Analysis (EDA)
3. Train and compare multiple ML algorithms
4. Select the best model based on evaluation metrics
5. Build a Streamlit web app for real-time prediction
6. Show probability-based risk assessment

---

## Slide 4: Dataset Description

**Source:** Kaggle - Diabetes Prediction Dataset

**Total Records:** 1,00,000 patients

**Features Used:**
| Feature | Description |
|---------|-------------|
| Gender | Male / Female / Other |
| Age | Patient age |
| Hypertension | 0 = No, 1 = Yes |
| Heart Disease | 0 = No, 1 = Yes |
| Smoking History | never, current, former, etc. |
| BMI | Body Mass Index |
| HbA1c Level | Average blood sugar (%) |
| Blood Glucose Level | mg/dL |

**Target:** Diabetes (0 = No, 1 = Yes)

---

## Slide 5: EDA - Key Findings

**Dataset Overview**
- 1,00,000 rows, 9 columns
- No missing values in the dataset

**Class Imbalance**
- Non-Diabetic: ~91.5%
- Diabetic: ~8.5%

**Important Insight**
- HbA1c and Blood Glucose are the strongest indicators of diabetes
- Patients with higher HbA1c and glucose levels show significantly higher diabetes rates

*(Insert chart: Target distribution + Boxplot of HbA1c vs Diabetes)*

---

## Slide 6: Data Preprocessing

**Steps Performed:**
1. **Categorical Encoding** - Gender & Smoking History converted using Ordinal Encoder
2. **Feature Scaling** - Numeric features scaled using StandardScaler
3. **Train-Test Split** - 80% training, 20% testing (stratified)
4. **Class Imbalance Handling** - Balanced class weights in models

**Why Stratified Split?**
- Ensures both diabetic and non-diabetic cases are represented in train and test sets

---

## Slide 7: Machine Learning Models Compared

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 94.04% | 70.64% | 97.54% |
| **Gradient Boosting** | **97.23%** | **80.98%** | **97.96%** |
| Logistic Regression | 88.75% | 57.29% | - |

**Selected Model:** Gradient Boosting (best F1 score)

**Why F1 Score?**
- Accuracy alone is misleading on imbalanced data
- F1 balances Precision and Recall

---

## Slide 8: Model Evaluation Metrics

**Best Model: Gradient Boosting**

| Metric | Value |
|--------|-------|
| Accuracy | 97.23% |
| Precision | 97.28% |
| Recall | 69.35% |
| F1 Score | 80.98% |
| ROC-AUC | 97.96% |
| Cross-Validation F1 | 80.45% |

**Confusion Matrix (Test Set)**
|  | Predicted: No | Predicted: Yes |
|--|---------------|----------------|
| **Actual: No** | 18,267 | 33 |
| **Actual: Yes** | 521 | 1,179 |

*(Insert confusion matrix chart from app)*

---

## Slide 9: Feature Importance

**Top Predictive Features:**
1. **HbA1c Level** - 62.9% importance
2. **Blood Glucose Level** - 32.2% importance
3. BMI - 1.9%
4. Age - 1.9%

**Conclusion:**
- Clinical blood markers dominate prediction
- Lifestyle factors (BMI, age) provide additional context

*(Insert feature importance bar chart)*

---

## Slide 10: System Architecture

```
Dataset (CSV)
    ↓
EDA Notebook (Analysis)
    ↓
train_model.py (Training + Evaluation)
    ↓
Saved Model (artifacts/diabetes_model.pkl)
    ↓
Streamlit App (app.py) → User Prediction
```

**Tech Stack:**
- Python, Pandas, NumPy
- Scikit-learn, Joblib
- Streamlit (Web UI)
- Matplotlib, Seaborn (Visualization)

---

## Slide 11: Streamlit Application Demo

**Features:**
- Patient data input form with validation
- Instant diabetes prediction (Yes / No)
- **Probability score** (e.g., 72.5% risk)
- **Risk level:** Low / Moderate / High
- Model comparison dashboard
- Confusion matrix & feature importance view

**How to Run:**
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

*(Insert screenshot of Streamlit app)*

---

## Slide 12: Sample Prediction Flow

**Example Input:**
- Male, Age 55, Hypertension: Yes, Heart Disease: No
- Smoking: Former, BMI: 32, HbA1c: 7.2, Glucose: 180

**Output:**
- Prediction: Diabetes Detected
- Probability: ~85%
- Risk Level: High
- Recommendation: Consult a doctor

---

## Slide 13: Advantages

- Fast and automated risk screening
- Multiple models compared for best accuracy
- Probability-based output (not just binary)
- User-friendly web interface
- Reproducible pipeline (train → save → deploy)
- Suitable for educational and demo purposes

---

## Slide 14: Limitations & Future Scope

**Limitations**
- Dataset is synthetic/simulated, not real hospital data
- Not a substitute for medical diagnosis
- Recall for diabetic class can be improved further

**Future Scope**
- Integrate real hospital EMR data
- Add SHAP explainability for each prediction
- Deploy on cloud (Streamlit Cloud / AWS)
- Mobile app integration
- Doctor dashboard with patient history

---

## Slide 15: Conclusion

- Successfully built an ML-based diabetes prediction system
- Achieved **97.23% accuracy** and **97.96% ROC-AUC**
- Gradient Boosting outperformed Random Forest and Logistic Regression
- Deployed interactive Streamlit application for end users
- Project demonstrates full ML lifecycle: EDA → Training → Evaluation → Deployment

---

## Slide 16: Thank You

**Questions?**

**Project Repository:** [GitHub Link]

**Demo:** `streamlit run app.py`

**Contact:** [Your Email]

---

## Viva / Q&A Preparation

**Q: Why is accuracy not enough?**
A: Dataset has 91.5% non-diabetic cases. A model predicting "No" always would still get ~91% accuracy. F1 and ROC-AUC handle imbalance better.

**Q: What is HbA1c?**
A: Hemoglobin A1c measures average blood sugar over 2-3 months. Normal is below 5.7%, prediabetes 5.7-6.4%, diabetes 6.5%+.

**Q: Why Gradient Boosting?**
A: It sequentially corrects errors from previous trees, giving strong performance on tabular healthcare data.

**Q: What is cross-validation?**
A: Model is trained and tested on 5 different data splits to ensure stable, reliable performance metrics.

**Q: Is this medically approved?**
A: No. This is an educational college project, not a certified medical device.
