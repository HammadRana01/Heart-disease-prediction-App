# app.py - Enhanced Heart Disease Prediction App
# ====================  IMPORTS ==========================
import streamlit as st
import pandas as pd


# Import our custom modules
from model import (load_data, split_data, train_model, evaluate_model,
                   select_best_features, extract_features, genetic_feature_selection)
from visuals import (plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
                     plot_correlation_heatmap, plot_model_comparison, plot_data_distribution,
                     create_confidence_gauge, plot_risk_factors)

# ==================== PAGE CONFIGURATION ==========================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ==========================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }

    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }

    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }

    .risk-low {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 2px solid #dee2e6;
        border-radius: 8px;
    }

    .feature-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ==========================
@st.cache_data
def load_and_process_data():
    """Load and preprocess the data (cached for performance)"""
    df, X, y = load_data()
    df_enhanced = extract_features(df)
    return df, df_enhanced, X, y


from fpdf import FPDF
from datetime import datetime

def generate_pdf_report(user_data, prediction, probability, model_name, model_accuracy, selected_features):
    """Generate a comprehensive PDF report"""
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.cell(200, 15, txt="Heart Disease Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # Header info
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Report Summary", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Basic info
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 8, txt=f"Model Used: {model_name}", ln=True)
    pdf.cell(200, 8, txt=f"Model Accuracy: {model_accuracy:.2%}", ln=True)
    pdf.ln(5)

    # Prediction result
    pdf.set_font("Arial", "B", 14)
    result_text = "Heart Disease Risk Detected" if prediction else "Low Heart Disease Risk"
    pdf.cell(200, 10, txt=f"Prediction: {result_text}", ln=True)
    pdf.cell(200, 8, txt=f"Confidence Level: {probability:.2%}", ln=True)
    pdf.ln(10)

    # Patient information
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Patient Information", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    for key, value in user_data.items():
        display_key = key.replace('_', ' ').title()
        pdf.cell(200, 6, txt=f"{display_key}: {value}", ln=True)

    pdf.ln(10)

    # Selected features
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Key Features Used in Prediction", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    for feature in selected_features[:10]:  # Show top 10 features
        pdf.cell(200, 6, txt=f"- {feature.replace('_', ' ').title()}", ln=True)

    # Recommendations
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Recommendations", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    if prediction:
        recommendations = [
            "- Consult with a cardiologist for further evaluation",
            "- Monitor blood pressure and cholesterol regularly",
            "- Maintain a heart-healthy diet low in saturated fats",
            "- Engage in regular physical exercise (as approved by doctor)",
            "- Consider stress management techniques"
        ]
    else:
        recommendations = [
            "- Continue maintaining healthy lifestyle habits",
            "- Regular check-ups with healthcare provider",
            "- Keep monitoring cardiovascular risk factors",
            "- Maintain healthy diet and exercise routine"
        ]

    for rec in recommendations:
        pdf.cell(200, 6, txt=rec, ln=True)

    output_path = "heart_disease_report.pdf"
    pdf.output(output_path)
    return output_path

# ==================== LOAD DATA ==========================
df, df_enhanced, X, y = load_and_process_data()

# ==================== MAIN HEADER ==========================
st.markdown("""
<div class="main-header">
    <h1>💓 Advanced Heart Disease Prediction System</h1>
    Using Machine Learning and Genetic Algorithm Optimization
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR: PATIENT INPUT ==========================
st.sidebar.markdown("## 🏥 Patient Information & Settings")
st.sidebar.markdown("---")

# Model selection with info
st.sidebar.markdown("### 🤖 AI Model Selection")
model_name = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Random Forest", "Logistic Regression"],
    help="Random Forest: More complex, higher accuracy. Logistic Regression: Simpler, faster."
)

# Test size
test_size = st.sidebar.slider(
    "Model Testing Data (%)",
    10, 50, 20,
    help="Percentage of data used for testing the model"
) / 100

# Feature selection method
feature_method = st.sidebar.selectbox(
    "Feature Selection Method",
    ["Genetic Algorithm", "Statistical Selection", "All Features"],
    help="Method to select the most important features"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Patient Details")

# Basic patient information
col1, col2 = st.sidebar.columns(2)
with col1:
    age = st.number_input("Age", 29, 77, 54, help="Patient's age in years")
with col2:
    sex = st.selectbox("Gender", ["Male", "Female"])
    sex_encoded = 1 if sex == "Male" else 0

# Chest pain type with descriptions
st.sidebar.markdown("#### 💔 Chest Pain Symptoms")
cp_options = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-Anginal Pain",
    3: "Asymptomatic"
}
cp = st.sidebar.selectbox("Chest Pain Type", list(cp_options.keys()),
                          format_func=lambda x: f"{x}: {cp_options[x]}")

# Vital signs
st.sidebar.markdown("#### 🩺 Vital Signs")
col1, col2 = st.sidebar.columns(2)
with col1:
    trestbps = st.number_input("Blood Pressure", 90, 200, 130,
                               help="Resting blood pressure (mm Hg)")
with col2:
    chol = st.number_input("Cholesterol", 120, 600, 245,
                           help="Serum cholesterol (mg/dL)")

# Heart measurements
st.sidebar.markdown("#### ❤️ Heart Measurements")
col1, col2 = st.sidebar.columns(2)
with col1:
    thalach = st.number_input("Max Heart Rate", 70, 210, 150,
                              help="Maximum heart rate achieved")
with col2:
    oldpeak = st.number_input("ST Depression", 0.0, 6.2, 1.0,
                              help="Exercise-induced ST depression")

# Medical tests
st.sidebar.markdown("#### 🔬 Medical Test Results")
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                           format_func=lambda x: "Yes" if x else "No")

restecg_options = {0: "Normal", 1: "Abnormal", 2: "Hypertrophy"}
restecg = st.sidebar.selectbox("ECG Result", list(restecg_options.keys()),
                               format_func=lambda x: restecg_options[x])

exang = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1],
                             format_func=lambda x: "Yes" if x else "No")

slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
slope = st.sidebar.selectbox("ST Segment Slope", list(slope_options.keys()),
                             format_func=lambda x: slope_options[x])

ca = st.sidebar.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4],
                          help="Number of major vessels colored by fluoroscopy")

thal_options = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}
thal = st.sidebar.selectbox("Thalassemia", list(thal_options.keys()),
                            format_func=lambda x: thal_options[x])

# Compile user input
user_input = {
    "age": age, "sex": sex_encoded, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}

# ==================== FEATURE PROCESSING ==========================
st.markdown("## 🔬 AI Processing Pipeline")

# Create columns for processing steps
proc_col1, proc_col2, proc_col3 = st.columns(3)

with proc_col1:
    st.markdown("""
    <div class="feature-info">
        <h4>📊 Feature Engineering</h4>
        <p>Creating enhanced features from raw data</p>
    </div>
    """, unsafe_allow_html=True)

with proc_col2:
    st.markdown("""
    <div class="feature-info">
        <h4>🧬 Feature Selection</h4>
        <p>Using AI to find the most important features</p>
    </div>
    """, unsafe_allow_html=True)

with proc_col3:
    st.markdown("""
    <div class="feature-info">
        <h4>🤖 Model Training</h4>
        <p>Training AI model for prediction</p>
    </div>
    """, unsafe_allow_html=True)

# Process features based on selection
with st.spinner("🔄 Processing data and optimizing features..."):
    if feature_method == "Genetic Algorithm":
        X_processed, selected_features, ga_score = genetic_feature_selection(X, y)
        st.success(f"✅ Genetic Algorithm found {len(selected_features)} optimal features (Score: {ga_score:.3f})")
    elif feature_method == "Statistical Selection":
        X_processed, selected_features, selector = select_best_features(X, y, k=10)
        st.success(f"✅ Statistical method selected {len(selected_features)} best features")
    else:
        X_processed = X
        selected_features = X.columns.tolist()
        st.info("ℹ️ Using all available features")

# ==================== MODEL TRAINING ==========================
# Split data and train model
X_train, X_test, y_train, y_test = split_data(X_processed, y, test_size)
model = train_model(model_name, X_train, y_train)
model_accuracy, predictions = evaluate_model(model, X_test, y_test)

# ==================== PREDICTION ==========================
# Prepare user input for prediction
user_df = pd.DataFrame([user_input])
if feature_method != "All Features":
    # Select only the features that were chosen by the selection method
    available_features = [f for f in selected_features if f in user_df.columns]
    user_df_processed = user_df[available_features]
else:
    user_df_processed = user_df

# Make prediction
prediction = model.predict(user_df_processed)[0]
if hasattr(model, 'predict_proba'):
    probability = model.predict_proba(user_df_processed)[0][1]
else:
    probability = 0.5  # Default probability for models without predict_proba

# ==================== PREDICTION RESULTS ==========================
st.markdown("## 🎯 Prediction Results")

# Create prediction display
risk_class = "risk-high" if prediction == 1 else "risk-low"
risk_text = "HIGH RISK - Heart Disease Detected" if prediction == 1 else "LOW RISK - No Heart Disease Detected"
risk_emoji = "🚨" if prediction == 1 else "✅"

st.markdown(f"""
<div class="prediction-box {risk_class}">
    <h2>{risk_emoji} {risk_text}</h2>
    <h3>Confidence Level: {probability:.1%}</h3>
    <p>Based on {model_name} model with {len(selected_features)} selected features</p>
</div>
""", unsafe_allow_html=True)

# Model performance metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Model Accuracy", f"{model_accuracy:.2%}")
with col2:
    st.metric("🔧 Features Used", len(selected_features))
with col3:
    st.metric("📊 Training Samples", len(X_train))
with col4:
    st.metric("🧪 Test Samples", len(X_test))

# ==================== DETAILED ANALYSIS ==========================
st.markdown("## 📈 Detailed Analysis")

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🔍 Feature Analysis", "📋 Data Insights", "⚠️ Risk Assessment"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_test, predictions)
        st.pyplot(fig_cm)

    with col2:
        st.subheader("📈 ROC Curve")
        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test)[:, 1]
            fig_roc = plot_roc_curve(y_test, y_scores)
            st.pyplot(fig_roc)
        else:
            st.info("ROC curve not available for this model type")

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Selected Features")
        if len(selected_features) <= 15:  # Show all if not too many
            for i, feature in enumerate(selected_features, 1):
                st.write(f"{i}. **{feature.replace('_', ' ').title()}**")
        else:  # Show top 10 if too many
            st.write("**Top 10 Most Important Features:**")
            for i, feature in enumerate(selected_features[:10], 1):
                st.write(f"{i}. **{feature.replace('_', ' ').title()}**")

    with col2:
        st.subheader("📊 Feature Importance")
        if hasattr(model, 'feature_importances_'):
            fig_imp = plot_feature_importance(model, selected_features)
            if fig_imp:
                st.pyplot(fig_imp)
        else:
            st.info("Feature importance not available for this model type")

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Correlation Analysis")
        fig_corr = plot_correlation_heatmap(df)
        st.pyplot(fig_corr)

    with col2:
        st.subheader("📊 Data Distribution")
        fig_dist = plot_data_distribution(df)
        st.pyplot(fig_dist)

with tab4:
    st.subheader("⚠️ Personal Risk Assessment")

    # Risk factors analysis
    risk_factors = []
    if user_input['age'] > 60:
        risk_factors.append("Age > 60 years")
    if user_input['chol'] > 240:
        risk_factors.append("High cholesterol (>240 mg/dL)")
    if user_input['trestbps'] > 140:
        risk_factors.append("High blood pressure (>140 mmHg)")
    if user_input['fbs'] == 1:
        risk_factors.append("High fasting blood sugar")
    if user_input['exang'] == 1:
        risk_factors.append("Exercise-induced chest pain")

    if risk_factors:
        st.warning("**Identified Risk Factors:**")
        for factor in risk_factors:
            st.write(f"• {factor}")
    else:
        st.success("✅ No major risk factors identified in the input data")

    # Risk comparison chart
    fig_risk = plot_risk_factors(user_input, {})
    if fig_risk:
        st.pyplot(fig_risk)

# ==================== EXPORT SECTION ==========================
st.markdown("## 📄 Export Results")

col1, col2 = st.columns(2)

with col1:
    if st.button("📋 Generate Detailed Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            report_path = generate_pdf_report(
                user_input, prediction, probability, model_name,
                model_accuracy, selected_features
            )
            st.success("✅ Report generated successfully!")

            # Provide download
            with open(report_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=file,
                    file_name=f"heart_disease_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )

with col2:
    if st.button("📊 Export Prediction Data"):
        # Create summary data
        summary_data = {
            'Patient_Age': [user_input['age']],
            'Gender': [sex],
            'Prediction': ['High Risk' if prediction else 'Low Risk'],
            'Confidence': [f"{probability:.2%}"],
            'Model_Used': [model_name],
            'Model_Accuracy': [f"{model_accuracy:.2%}"],
            'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }

        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False)

        st.download_button(
            label="⬇️ Download CSV Data",
            data=csv,
            file_name=f"prediction_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# ==================== FOOTER ==========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
   
    <strong>Developed for Academic Project | Galgotias University</strong>
</div>
""", unsafe_allow_html=True)