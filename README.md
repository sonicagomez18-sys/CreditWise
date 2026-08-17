CreditWise — Loan Approval Prediction System

CreditWise is a machine learning-based system that predicts whether a loan application is likely to be Approved or Not Approved based on applicant financial and demographic information.

🚀 Live Demo

https://credit-wise2.streamlit.app/

📌 Project Overview

The project follows an end-to-end machine learning workflow:

* Data preprocessing and exploratory data analysis
* Feature engineering and categorical encoding
* Feature scaling
* Training and comparison of Logistic Regression, KNN, and Naive Bayes
* Model evaluation using Precision, Recall, and F1-Score
* Selection of Naive Bayes as the final model
* Deployment using Streamlit

🛠️ Technologies

Python · Pandas · NumPy · Scikit-learn · Streamlit · Joblib

🤖 Machine Learning

The model analyzes factors such as:

* Applicant & Co-applicant Income
* Credit Score
* Savings
* Loan Amount & Term
* Age & Dependents
* Employment Status
* Education Level
* Collateral Value
* Debt-to-Income Ratio

The models were compared with emphasis on precision, with the goal of reducing incorrect approval predictions for potentially high-risk applicants.

📂 Project Structure

CreditWise/
├── app.py
├── credit_wise.ipynb
├── naive_bayes_model.pkl
├── scaler.pkl
├── onehot_encoder.pkl
├── education_encoder.pkl
└── README.md

💡 Key Skills Demonstrated

Machine Learning · Classification · Data Analysis · Feature Engineering · Model Evaluation · Python · Scikit-learn · Streamlit · Model Deployment
