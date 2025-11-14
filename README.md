# Student_Performance_Prediction
🎓 Student Performance Prediction
This project uses machine learning techniques to predict student academic performance based on various socio-economic and academic features. It includes both regression (predicting scores) and classification (predicting pass/fail or grade categories) models.

📁 Dataset
The dataset contains student-related attributes such as:
- Demographics (gender, age, parental education)
- Academic background (study time, failures, absences)
- Lifestyle and support (internet access, family support, health)
Target Variables:
- For regression: final grade (G3)
- For classification: pass/fail or grade category (e.g., A/B/C)

🧠 ML Models Used
🔢 Regression
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
🧮 Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

🛠️ Project Structure  
student-performance-prediction/
<img width="681" height="232" alt="image" src="https://github.com/user-attachments/assets/2fb7dac2-355b-4fcc-bfa7-72bb3932cdd4" />




📊 Evaluation Metrics
- Regression: MAE, MSE, RMSE, R² Score
- Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

🚀 How to Run
- Clone the repository:
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
- Install dependencies:
pip install -r requirements.txt
- Run preprocessing:
python src/preprocessing.py
- Train models:
python src/train_regression.py
python src/train_classification.py



📌 Future Improvements
- Hyperparameter tuning with GridSearchCV
- Model interpretability using SHAP
- Deployment via Flask or Streamlit

📚 References
- UCI Machine Learning Repository: Student Performance Data Set
- Scikit-learn documentation
- XGBoost and SHAP libraries
