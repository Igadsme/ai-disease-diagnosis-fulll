# 🏥 AI Disease Diagnosis System

A production-ready Machine Learning system for predicting heart disease using patient health metrics. Built with explainable AI principles using SHAP (SHapley Additive exPlanations) for model transparency and trust.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.26.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Project Overview

This project demonstrates a complete end-to-end machine learning pipeline for medical diagnosis:

- **Problem**: Predict whether a patient has heart disease based on clinical parameters
- **Solution**: Logistic Regression model with SHAP explainability
- **Deployment**: Interactive Streamlit web application for real-time predictions

### Key Features

✅ **Comprehensive EDA** - Correlation analysis, distribution plots, and statistical insights  
✅ **Robust Preprocessing** - Missing value handling, feature scaling, and train-test split  
✅ **High Performance** - Achieves ~85% accuracy with strong AUC-ROC scores  
✅ **Explainable AI** - SHAP values show which features drive each prediction  
✅ **User-Friendly Interface** - Streamlit app for non-technical users  
✅ **Production Ready** - Clean code, proper documentation, and modular design  

---

## 📊 Dataset

**Source**: UCI Heart Disease Dataset / Kaggle (johnsmith88/heart-disease-dataset)

**Features** (13 clinical parameters):
- `age`: Age in years
- `sex`: Sex (1 = male, 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `restecg`: Resting ECG results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment (0-2)
- `ca`: Number of major vessels colored by fluoroscopy (0-3)
- `thal`: Thalassemia (1-3)

**Target**: `target` (1 = heart disease, 0 = no heart disease)

---

## 🛠️ Tools & Technologies

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Explainability** | SHAP |
| **Web Framework** | Streamlit |
| **Notebook** | Jupyter |

---

## 📈 Model Performance

### Logistic Regression Classifier

| Metric | Score |
|--------|-------|
| **Accuracy** | ~85% |
| **Precision** | ~84% |
| **Recall** | ~87% |
| **F1-Score** | ~85% |
| **AUC-ROC** | ~0.90 |

**Why Logistic Regression?**
- Highly interpretable (critical for healthcare)
- Fast training and inference
- Strong baseline performance
- Works well with SHAP for explainability
- No black-box complexity

---

🧠 Explainable AI with SHAP

This project implements **SHAP (SHapley Additive exPlanations)** to provide transparency:

### What SHAP Shows:
- **Feature Importance**: Which clinical parameters matter most?
- **Individual Predictions**: Why did the model predict heart disease for this patient?
- **Decision Logic**: How do different feature values push predictions toward/away from disease?

### Ethical Considerations:
⚠️ **Transparency**: Doctors and patients can understand WHY a prediction was made  
⚠️ **Trust**: Explainability builds confidence in AI-assisted diagnosis  
⚠️ **Accountability**: Clear reasoning helps identify model biases or errors  

> **Note**: This is a demonstration project. Real medical AI systems require extensive validation, regulatory approval (FDA, CE marking), and should NEVER replace professional medical judgment.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- (For Kaggle auto-download) a Kaggle account with API credentials in `~/.kaggle/kaggle.json`

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-disease-diagnosis.git
cd ai-disease-diagnosis
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Notebook or App**
- Notebook (full analysis): `jupyter notebook disease_prediction.ipynb`
- App (UI): `streamlit run app.py`

---

## 📁 Project Structure
```
ai-disease-diagnosis/
├── README.md
├── requirements.txt
├── disease_prediction.ipynb
├── app.py
├── data/
│   └── heart.csv
└── models/
    └── logistic_model.pkl
```

---

## 🔬 Methodology
1. Data Loading → EDA → Preprocessing → Train/Test Split  
2. Train Logistic Regression → Cross-Validation  
3. Evaluate (Accuracy, F1, AUC, ROC)  
4. Explain with SHAP (global + per-patient)  
5. Save artifacts and serve via Streamlit

---

## 🔮 Future Enhancements
- [ ] Ensemble models (RF, XGBoost)
- [ ] Hyperparameter tuning & CV
- [ ] Cloud deployment (Streamlit Cloud/AWS)
- [ ] Multi-disease support

---

## ⚠️ Disclaimer
**Educational project — not for medical use.** Not FDA approved or clinically validated.

---

## 👨‍💻 Author
**Imani L Gad**

---

## 📚 References
- UCI Heart Disease Dataset
- scikit-learn docs
- SHAP docs
