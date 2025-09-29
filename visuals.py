# visuals.py - Enhanced Visualization Functions
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# Set nice style for matplotlib plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ===== Confusion Matrix Plot =====
def plot_confusion_matrix(y_true, y_pred):
    """
    Creates a beautiful confusion matrix plot.
    Shows how well our model predicts each class.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                square=True, cbar_kws={'shrink': 0.8}, ax=ax)

    # Add labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix\n(How well did our model predict?)',
                 fontsize=14, fontweight='bold', pad=20)

    # Set tick labels
    ax.set_xticklabels(['No Disease', 'Heart Disease'])
    ax.set_yticklabels(['No Disease', 'Heart Disease'])

    plt.tight_layout()
    return fig


# ===== ROC Curve Plot =====
def plot_roc_curve(y_true, y_scores):
    """
    Creates ROC curve plot showing model performance.
    ROC curve shows the trade-off between sensitivity and specificity.
    """
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=3,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')

    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            alpha=0.8, label='Random Classifier')

    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Model Performance\n(Higher curve = Better model)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ===== Feature Importance Plot =====
def plot_feature_importance(model, feature_names):
    """
    Shows which features are most important for predictions.
    Only works with tree-based models like Random Forest.
    """
    if not hasattr(model, 'feature_importances_'):
        return None

    # Get feature importances
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(feature_imp_df['feature'], feature_imp_df['importance'],
                   color='skyblue', edgecolor='darkblue', linewidth=1)

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}', ha='left', va='center', fontweight='bold')

    # Styling
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance\n(Which factors matter most for prediction?)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


# ===== Interactive Correlation Heatmap =====
def plot_correlation_heatmap(df):
    """
    Creates an enhanced correlation heatmap showing relationships between features.
    """
    # Calculate correlation matrix
    corr = df.corr(numeric_only=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Generate heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlBu_r', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": .5}, ax=ax)

    # Styling
    ax.set_title('Correlation Matrix\n(How features relate to each other)',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


# ===== Model Comparison Chart =====
def plot_model_comparison(model_results):
    """
    Compares performance of different models.
    model_results should be a dict like: {'Random Forest': 0.85, 'Logistic Regression': 0.80}
    """
    models = list(model_results.keys())
    scores = list(model_results.values())

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars with gradient colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(models, scores, color=colors[:len(models)],
                  edgecolor='darkblue', linewidth=2, alpha=0.8)

    # Add value labels on top of bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=12)

    # Styling
    ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison\n(Higher is better)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


# ===== Data Distribution Plots =====
def plot_data_distribution(df, target_column='target'):
    """
    Shows the distribution of key features by target class.
    """
    # Select key numeric features
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    available_features = [col for col in numeric_features if col in df.columns]

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(available_features):
        if i < len(axes):
            # Create violin plot
            sns.violinplot(data=df, x=target_column, y=feature, ax=axes[i])
            axes[i].set_title(f'{feature.title()} Distribution', fontweight='bold')
            axes[i].set_xlabel('Heart Disease (0=No, 1=Yes)')

    # Remove empty subplots
    for i in range(len(available_features), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle('Feature Distributions by Target Class\n(Compare patterns between groups)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# ===== Prediction Confidence Gauge =====
def create_confidence_gauge(confidence_score):
    """
    Creates a gauge chart showing prediction confidence using Plotly.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Confidence (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=400)
    return fig


# ===== Risk Factors Chart =====
def plot_risk_factors(user_data, feature_ranges):
    """
    Shows user's values compared to normal ranges for each risk factor.
    """
    # Define normal ranges for key features
    normal_ranges = {
        'age': (30, 60),
        'trestbps': (90, 120),
        'chol': (150, 200),
        'thalach': (120, 180)
    }

    # Filter user data for available features
    available_features = [k for k in normal_ranges.keys() if k in user_data.keys()]

    if not available_features:
        return None

    # Create the comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(available_features))
    user_values = [user_data[feat] for feat in available_features]
    normal_mins = [normal_ranges[feat][0] for feat in available_features]
    normal_maxs = [normal_ranges[feat][1] for feat in available_features]

    # Plot normal ranges as bars
    ax.bar(x_pos, normal_maxs, alpha=0.3, color='green',
           label='Normal Range', width=0.6)

    # Plot user values as points
    ax.scatter(x_pos, user_values, color='red', s=100,
               label='Your Values', zorder=5, marker='o')

    # Connect points with lines for better visibility
    ax.plot(x_pos, user_values, 'r--', alpha=0.7, linewidth=2)

    # Styling
    ax.set_xlabel('Risk Factors', fontsize=12, fontweight='bold')
    ax.set_ylabel('Values', fontsize=12, fontweight='bold')
    ax.set_title('Your Risk Factors vs Normal Ranges\n(Red dots show your values)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([feat.title() for feat in available_features])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig