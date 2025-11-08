"""
AI Disease Diagnosis System - Streamlit Web Application
Author: Imani Gad
Date: 2024

A user-friendly web interface for heart disease prediction using Machine Learning.
Features real-time predictions with SHAP explainability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="AI Disease Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model, scaler, and feature names
@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, and feature names"""
    model = joblib.load('models/logistic_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, feature_names

# Feature descriptions for user guidance
FEATURE_INFO = {
    'age': ('Age', 'years', 20, 100, 50, 'Patient age in years'),
    'sex': ('Sex', '', 0, 1, 1, '0 = Female, 1 = Male'),
    'cp': ('Chest Pain Type', '', 0, 3, 0, '0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic'),
    'trestbps': ('Resting Blood Pressure', 'mm Hg', 80, 200, 120, 'Blood pressure at rest'),
    'chol': ('Cholesterol', 'mg/dl', 100, 600, 200, 'Serum cholesterol level'),
    'fbs': ('Fasting Blood Sugar > 120', '', 0, 1, 0, '0 = No, 1 = Yes'),
    'restecg': ('Resting ECG', '', 0, 2, 0, '0=Normal, 1=ST-T abnormality, 2=LV hypertrophy'),
    'thalach': ('Max Heart Rate', 'bpm', 60, 220, 150, 'Maximum heart rate achieved'),
    'exang': ('Exercise Induced Angina', '', 0, 1, 0, '0 = No, 1 = Yes'),
    'oldpeak': ('ST Depression', 'mm', 0.0, 6.5, 0.0, 'ST depression induced by exercise'),
    'slope': ('ST Slope', '', 0, 2, 0, '0=Upsloping, 1=Flat, 2=Downsloping'),
    'ca': ('Major Vessels', '', 0, 3, 0, 'Number of major vessels colored by fluoroscopy'),
    'thal': ('Thalassemia', '', 1, 3, 2, '1=Normal, 2=Fixed defect, 3=Reversible defect')
}

def sidebar_inputs(feature_names):
    st.sidebar.header("📋 Patient Information")
    st.sidebar.markdown("Enter patient clinical parameters below:")
    user_input = {}
    for feature in feature_names:
        if feature in FEATURE_INFO:
            name, unit, min_val, max_val, default, help_text = FEATURE_INFO[feature]
            label = f"{name} ({unit})" if unit else name
            # categorical (small range ints)
            if isinstance(min_val, int) and isinstance(max_val, int) and (max_val - min_val) <= 10:
                options = list(range(int(min_val), int(max_val)+1))
                default_index = options.index(int(default)) if int(default) in options else 0
                user_input[feature] = st.sidebar.selectbox(label, options=options, index=default_index, help=help_text)
            # float inputs
            elif isinstance(min_val, float) or isinstance(default, float):
                user_input[feature] = st.sidebar.number_input(label, min_value=float(min_val), max_value=float(max_val),
                                                              value=float(default), step=0.1, help=help_text)
            # sliders for wider int ranges
            else:
                user_input[feature] = st.sidebar.slider(label, min_value=int(min_val), max_value=int(max_val),
                                                        value=int(default), help=help_text)
    return user_input

def generate_clinical_interpretation(prediction, prediction_proba, contributions):
    """Generate human-readable clinical interpretation"""
    top_risk_factors = contributions[contributions['SHAP Value'] > 0].head(3)
    top_protective_factors = contributions[contributions['SHAP Value'] < 0].head(3)
    interpretation = ""
    if prediction == 1:
        interpretation += "**⚠️ High Risk Assessment:**\n\n"
        interpretation += f"The model predicts a {prediction_proba[1]:.1%} probability of heart disease. "
        interpretation += "Key risk factors identified:\n\n"
        for _, row in top_risk_factors.iterrows():
            interpretation += f"- **{FEATURE_INFO[row['Feature']][0]}**: Value of {row['Value']} increases risk\n"
        if len(top_protective_factors) > 0:
            interpretation += "\n**Protective factors present:**\n\n"
            for _, row in top_protective_factors.iterrows():
                interpretation += f"- **{FEATURE_INFO[row['Feature']][0]}**: Value of {row['Value']} reduces risk\n"
        interpretation += "\n**Recommended Actions:**\n"
        interpretation += "- Schedule comprehensive cardiac evaluation\n"
        interpretation += "- Consider stress test or cardiac imaging\n"
        interpretation += "- Review and optimize risk factors (cholesterol, blood pressure, etc.)\n"
    else:
        interpretation += "**✅ Low Risk Assessment:**\n\n"
        interpretation += f"The model predicts a {prediction_proba[0]:.1%} probability of NO heart disease. "
        interpretation += "Key protective factors:\n\n"
        for _, row in top_protective_factors.iterrows():
            interpretation += f"- **{FEATURE_INFO[row['Feature']][0]}**: Value of {row['Value']} is favorable\n"
        if len(top_risk_factors) > 0:
            interpretation += "\n**Risk factors to monitor:**\n\n"
            for _, row in top_risk_factors.iterrows():
                interpretation += f"- **{FEATURE_INFO[row['Feature']][0]}**: Value of {row['Value']} warrants monitoring\n"
        interpretation += "\n**Recommended Actions:**\n"
        interpretation += "- Continue regular health checkups\n"
        interpretation += "- Maintain heart-healthy lifestyle\n"
        interpretation += "- Monitor risk factors regularly\n"
    return interpretation

def main():
    st.title("🏥 AI Disease Diagnosis System")
    st.caption("Heart Disease Risk Prediction with Explainable AI (SHAP) — Author: Imani Gad")

    # Load artifacts
    try:
        model, scaler, feature_names = load_model_artifacts()
    except Exception as e:
        st.error("❌ Model files not found. Please run the notebook to train and save artifacts first.")
        st.stop()

    # Inputs
    user_input = sidebar_inputs(feature_names)

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("📊 Patient Data Summary")
        patient_df = pd.DataFrame([user_input])
        st.dataframe(patient_df, use_container_width=True)
    with col2:
        st.subheader("🔍 Prediction")
        if st.button("🔬 Predict Heart Disease Risk"):
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            st.session_state.prediction = prediction
            st.session_state.prediction_proba = prediction_proba
            st.session_state.input_scaled = input_scaled
            st.session_state.input_df = input_df

    if "prediction" in st.session_state:
        st.markdown("---")
        prediction = st.session_state.prediction
        prediction_proba = st.session_state.prediction_proba

        if prediction == 1:
            st.error(f"⚠️ HIGH RISK — Heart Disease Probability: {prediction_proba[1]:.1%}")
        else:
            st.success(f"✅ LOW RISK — No Disease Probability: {prediction_proba[0]:.1%}")

        # SHAP analysis
        st.subheader("🧠 Model Explanation (SHAP)")
        try:
            explainer = shap.LinearExplainer(model, scaler.transform(
                pd.DataFrame([[50, 1, 0, 120, 200, 0, 0, 150, 0, 0.0, 0, 0, 2]], columns=feature_names)
            ))
            shap_values = explainer.shap_values(st.session_state.input_scaled)
            contributions = pd.DataFrame({
                'Feature': feature_names,
                'Value': st.session_state.input_df.iloc[0].values,
                'SHAP Value': shap_values[0],
                'Absolute Impact': np.abs(shap_values[0])
            }).sort_values('Absolute Impact', ascending=False)
            st.write(contributions[['Feature','Value','SHAP Value']])

            # Waterfall-style bar plot
            fig, ax = plt.subplots(figsize=(8,6))
            sorted_contrib = contributions.sort_values('SHAP Value')
            colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in sorted_contrib['SHAP Value']]
            ax.barh(sorted_contrib['Feature'], sorted_contrib['SHAP Value'], color=colors, alpha=0.8)
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            ax.set_xlabel("SHAP value (impact on model output)")
            ax.set_title("Feature Impact on Prediction")
            st.pyplot(fig)

            st.markdown("### 🩺 Clinical Interpretation")
            st.markdown(generate_clinical_interpretation(prediction, prediction_proba, contributions))
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")

    st.markdown("---")
    st.caption("Educational demo — not for medical use.")

if __name__ == "__main__":
    main()
