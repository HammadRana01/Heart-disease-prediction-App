❤️ Advanced Heart Disease Prediction System

An interactive machine learning web app to predict the risk of heart disease.
Built with Scikit-learn and Streamlit, the system combines predictive modeling with optimization techniques to deliver accurate and user-friendly results.

⸻

🚀 Features
	•	Machine Learning Models
	•	Random Forest
	•	Logistic Regression
	•	Feature Selection & Optimization
	•	Genetic Algorithm for selecting the most important features
	•	Meta-heuristic optimization for improved accuracy
	•	Interactive Streamlit UI
	•	Enter patient details (age, gender, chest pain type, cholesterol, ECG results, etc.)
	•	Visual AI pipeline (feature engineering → feature selection → model training)
	•	Real-time prediction results with confidence level
	•	Reports & Analysis
	•	Confusion Matrix & ROC Curve visualization
	•	Model accuracy and feature importance insights
	•	Exportable PDF report of results

  
📂 Project Structure
├── App.py          # Main Streamlit application (UI)
├── Model.py        # Machine learning models (Random Forest, Logistic Regression)
├── Utility.py      # Helper functions (data preprocessing, PDF export, etc.)
├── Visuals.py      # Visualization functions (confusion matrix, ROC curve)
├── Dataset.csv     # Heart disease dataset (from Kaggle)
└── requirements.txt # Dependencies


⸻

📊 Example Output
	•	Risk prediction: High / Low Risk with confidence level
	•	Model performance: accuracy, training/testing data split
	•	Visual insights: ROC curve & confusion matrix

⸻

🛠️ Tech Stack
	•	Python
	•	Scikit-Learn → ML models (Random Forest & Logistic Regression)
	•	Streamlit → Web app UI
	•	Genetic Algorithm → Feature selection & optimization

⸻

📥 Installation
# Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

📤 Export Options
	•	Prediction Report (downloadable as PDF)
	•	Detailed Analysis (ROC curve, confusion matrix)

🔮 Future Improvements
	•	Add support for more ML/DL models
	•	Deploy as a cloud service (e.g., Hugging Face Spaces, Heroku, or Streamlit Cloud)
	•	Improve UI/UX with advanced medical data visualizations

⸻

⚠️ Disclaimer

This tool is built for educational and research purposes only.
It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
