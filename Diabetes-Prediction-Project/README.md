# Diabetes Prediction using Machine Learning

A complete college ML project that predicts diabetes risk from patient health indicators and deploys the model through an interactive Streamlit web application.

## Introduction

This project uses machine learning to classify whether a person is likely to have diabetes based on demographic and clinical features such as age, gender, BMI, hypertension, heart disease, smoking history, HbA1c level, and blood glucose level.

The system includes:
- Exploratory Data Analysis (EDA) notebook
- Automated model training with multiple algorithm comparison
- Saved ML pipeline for consistent predictions
- Streamlit web app with probability-based risk assessment

## Project Structure

```
Diabetes-Prediction-using-Machine-Learning/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training & evaluation script
├── requirements.txt        # Python dependencies
├── notebooks/
│   └── diabetes_eda.ipynb  # EDA notebook with visualizations
├── docs/
│   └── PRESENTATION.md     # College presentation slide content
├── artifacts/
│   ├── diabetes_model.pkl  # Saved best model pipeline
│   └── model_metrics.json  # Training metrics & comparisons
└── .streamlit/
    └── config.toml         # Streamlit theme config
```

## Key Technologies

- Python 3.10+
- Pandas, NumPy
- Scikit-learn, Joblib
- Streamlit
- Matplotlib, Seaborn (EDA)

## Installation

```bash
git clone https://github.com/gopiashokan/Diabetes-Prediction-using-Machine-Learning.git
cd Diabetes-Prediction-using-Machine-Learning
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python train_model.py
```

This compares Random Forest, Gradient Boosting, and Logistic Regression, then saves the best model to `artifacts/`.

### 2. Run the Web App

```bash
streamlit run app.py
```

Open in browser: `http://localhost:8501`

### 3. Run EDA Notebook (Optional)

```bash
jupyter notebook notebooks/diabetes_eda.ipynb
```

## Model Performance

**Best Model:** Gradient Boosting

| Metric | Score |
|--------|-------|
| Accuracy | 97.23% |
| Precision | 97.28% |
| Recall | 69.35% |
| F1 Score | 80.98% |
| ROC-AUC | 97.96% |
| CV F1 (5-fold) | 80.45% |

## Features

### Machine Learning Pipeline
- Stratified train-test split (80/20)
- Ordinal encoding for categorical features
- Standard scaling for numeric features
- Class imbalance handling with balanced weights
- 5-fold cross-validation
- Automatic best model selection by F1 score

### Streamlit Application
- **Predict Tab** - Input patient details and get instant prediction
- **Model Analysis Tab** - Model comparison, confusion matrix, feature importance
- **About Tab** - Project overview and setup instructions
- Probability percentage and risk level (Low / Moderate / High)

### Top Predictive Features
1. HbA1c Level (~63% importance)
2. Blood Glucose Level (~32% importance)
3. BMI, Age, Hypertension, Heart Disease

## Dataset

**Source:** [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iamahmadjutt/diabetes-prediction-dataset) (Kaggle)

- **Records:** 100,000
- **Features:** 8 input features + 1 target (diabetes)
- **Class Distribution:** ~91.5% non-diabetic, ~8.5% diabetic

## College Presentation

Ready-made slide content is available in [`docs/PRESENTATION.md`](docs/PRESENTATION.md) including:
- Introduction & objectives
- EDA findings
- Model comparison tables
- Architecture diagram
- Viva Q&A preparation

## Disclaimer

This project is for **educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified doctor for health decisions.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## Contact

- Email: gopiashokankiot@gmail.com
- LinkedIn: [linkedin.com/in/gopiashokan](https://www.linkedin.com/in/gopiashokan)
