#Utility Functions for Heart Disease Prediction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# ===== Data Validation Functions =====
def validate_input_data(user_data):
    """
    Validates user input data to ensure it's within acceptable ranges.
    Returns True if valid, False otherwise with error messages.
    """
    errors = []

    # Define acceptable ranges for each feature
    valid_ranges = {
        'age': (1, 120),
        'sex': (0, 1),
        'cp': (0, 3),
        'trestbps': (50, 250),
        'chol': (50, 800),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'thalach': (50, 250),
        'exang': (0, 1),
        'oldpeak': (0, 10),
        'slope': (0, 2),
        'ca': (0, 4),
        'thal': (0, 2)
    }

    # Check each field
    for field, (min_val, max_val) in valid_ranges.items():
        if field in user_data:
            value = user_data[field]
            if not (min_val <= value <= max_val):
                errors.append(f"❌ {field.title()}: {value} is outside valid range ({min_val}-{max_val})")

    return len(errors) == 0, errors


def clean_dataset(df):
    """
    Cleans the dataset by handling missing values and outliers.
    Returns cleaned dataframe.
    """
    df_clean = df.copy()

    # Handle missing values
    if df_clean.isnull().sum().any():
        # Fill numeric columns with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    # Remove obvious outliers (values that are clearly data entry errors)
    # Age outliers
    df_clean = df_clean[(df_clean['age'] >= 20) & (df_clean['age'] <= 100)]

    # Blood pressure outliers
    df_clean = df_clean[(df_clean['trestbps'] >= 80) & (df_clean['trestbps'] <= 220)]

    # Cholesterol outliers (0 values are likely missing data)
    df_clean = df_clean[df_clean['chol'] > 0]

    return df_clean


# ===== Model Evaluation Functions =====
def comprehensive_model_evaluation(model, X_test, y_test, model_name="Model"):
    """
    Performs comprehensive evaluation of a trained model.
    Returns dictionary with all evaluation metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    # Get probability scores if available
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics['probability_scores'] = y_prob

    # Classification report
    metrics['classification_report'] = classification_report(y_test, y_pred)

    return metrics


def compare_models(models_dict, X_test, y_test):
    """
    Compares multiple models and returns performance comparison.
    models_dict: {'Model Name': trained_model}
    """
    comparison_results = {}

    for name, model in models_dict.items():
        metrics = comprehensive_model_evaluation(model, X_test, y_test, name)
        comparison_results[name] = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        }

    # Convert to DataFrame for easy comparison
    comparison_df = pd.DataFrame(comparison_results).T
    return comparison_df


# ===== Data Preprocessing Functions =====
def preprocess_data(df, target_column='target'):
    """
    Preprocesses the dataset for machine learning.
    Includes scaling and encoding if necessary.
    """
    df_processed = df.copy()

    # Separate features and target
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]

    # Standard scaling for numerical features
    scaler = StandardScaler()
    numerical_features = X.select_dtypes(include=[np.number]).columns

    if len(numerical_features) > 0:
        X_scaled = X.copy()
        X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])
        return X_scaled, y, scaler

    return X, y, None


# ===== Feature Analysis Functions =====
def analyze_feature_correlation(df, target_col='target'):
    """
    Analyzes correlation between features and target variable.
    Returns sorted correlation values.
    """
    # Calculate correlations with target
    correlations = df.corr()[target_col].drop(target_col)

    # Sort by absolute correlation value
    correlations_sorted = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )

    return correlations_sorted


def get_feature_statistics(df):
    """
    Gets comprehensive statistics for all features in the dataset.
    """
    stats = {}

    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            stats[column] = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'quartile_25': df[column].quantile(0.25),
                'quartile_75': df[column].quantile(0.75)
            }
        else:
            stats[column] = {
                'unique_values': df[column].nunique(),
                'most_common': df[column].mode().iloc[0] if not df[column].mode().empty else None,
                'value_counts': df[column].value_counts().to_dict()
            }

    return stats


# ===== Risk Assessment Functions =====
def calculate_risk_score(user_data):
    """
    Calculates a simple risk score based on known cardiovascular risk factors.
    Higher score indicates higher risk.
    """
    risk_score = 0

    # Age factor (higher age = higher risk)
    if user_data.get('age', 0) > 65:
        risk_score += 3
    elif user_data.get('age', 0) > 55:
        risk_score += 2
    elif user_data.get('age', 0) > 45:
        risk_score += 1

    # Gender factor (males typically higher risk)
    if user_data.get('sex', 0) == 1:  # Male
        risk_score += 1

    # Chest pain factor
    cp_value = user_data.get('cp', 0)
    if cp_value == 0:  # Typical angina
        risk_score += 3
    elif cp_value == 1:  # Atypical angina
        risk_score += 2
    elif cp_value == 2:  # Non-anginal pain
        risk_score += 1

    # Blood pressure factor
    bp = user_data.get('trestbps', 120)
    if bp > 160:
        risk_score += 3
    elif bp > 140:
        risk_score += 2
    elif bp > 130:
        risk_score += 1

    # Cholesterol factor
    chol = user_data.get('chol', 200)
    if chol > 280:
        risk_score += 3
    elif chol > 240:
        risk_score += 2
    elif chol > 200:
        risk_score += 1

    # Other risk factors
    if user_data.get('fbs', 0) == 1:  # High fasting blood sugar
        risk_score += 1

    if user_data.get('exang', 0) == 1:  # Exercise-induced angina
        risk_score += 2

    # Maximum heart rate (lower can be concerning)
    max_hr = user_data.get('thalach', 150)
    expected_max_hr = 220 - user_data.get('age', 50)
    if max_hr < expected_max_hr * 0.6:
        risk_score += 2
    elif max_hr < expected_max_hr * 0.75:
        risk_score += 1

    return risk_score


def interpret_risk_score(risk_score):
    """
    Interprets the calculated risk score and provides recommendations.
    """
    if risk_score <= 3:
        risk_level = "Low"
        recommendations = [
            "Continue maintaining healthy lifestyle habits",
            "Regular exercise and balanced diet",
            "Annual health check-ups"
        ]
    elif risk_score <= 7:
        risk_level = "Moderate"
        recommendations = [
            "Monitor cardiovascular risk factors regularly",
            "Consider lifestyle modifications",
            "Consult healthcare provider for assessment",
            "Regular blood pressure and cholesterol checks"
        ]
    else:
        risk_level = "High"
        recommendations = [
            "Immediate consultation with healthcare provider recommended",
            "Consider comprehensive cardiovascular evaluation",
            "Lifestyle modifications may be necessary",
            "Regular monitoring of all risk factors",
            "Follow medical advice strictly"
        ]

    return risk_level, recommendations


# ===== Data Export Functions =====
def create_summary_report(user_data, prediction, probability, model_metrics, selected_features):
    """
    Creates a comprehensive summary report as a dictionary.
    """
    risk_score = calculate_risk_score(user_data)
    risk_level, recommendations = interpret_risk_score(risk_score)

    report = {
        'patient_info': user_data,
        'prediction_result': {
            'prediction': 'Heart Disease Risk' if prediction else 'No Heart Disease Risk',
            'confidence': f"{probability:.2%}",
            'risk_score': risk_score,
            'risk_level': risk_level
        },
        'model_performance': model_metrics,
        'selected_features': selected_features,
        'recommendations': recommendations,
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return report


# ===== Helper Functions =====
def format_feature_names(feature_list):
    """
    Converts technical feature names to user-friendly names.
    """
    name_mapping = {
        'age': 'Age',
        'sex': 'Gender',
        'cp': 'Chest Pain Type',
        'trestbps': 'Resting Blood Pressure',
        'chol': 'Cholesterol Level',
        'fbs': 'Fasting Blood Sugar',
        'restecg': 'ECG Results',
        'thalach': 'Maximum Heart Rate',
        'exang': 'Exercise-Induced Angina',
        'oldpeak': 'ST Depression',
        'slope': 'ST Segment Slope',
        'ca': 'Major Vessels',
        'thal': 'Thalassemia',
        'target': 'Heart Disease'
    }

    return [name_mapping.get(feature, feature.title()) for feature in feature_list]


def get_feature_descriptions():
    """
    Returns descriptions for each feature to help users understand them.
    """
    descriptions = {
        'age': 'Patient age in years',
        'sex': 'Gender (1 = male, 0 = female)',
        'cp': 'Chest pain type (0-3, where 0 = typical angina)',
        'trestbps': 'Resting blood pressure in mm Hg',
        'chol': 'Serum cholesterol in mg/dL',
        'fbs': 'Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)',
        'restecg': 'Resting ECG results (0 = normal, 1 = abnormal, 2 = hypertrophy)',
        'thalach': 'Maximum heart rate achieved during exercise',
        'exang': 'Exercise-induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-4)',
        'thal': 'Thalassemia type (0 = normal, 1 = fixed defect, 2 = reversible defect)'
    }

    return descriptions