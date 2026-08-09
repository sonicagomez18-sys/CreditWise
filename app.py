import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# LOAD MODEL AND PREPROCESSING FILES
# =========================================================

model = joblib.load("naive_bayes_model.pkl")
scaler = joblib.load("scaler.pkl")
onehot_encoder = joblib.load("onehot_encoder.pkl")
education_encoder = joblib.load("education_encoder.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CreditWise",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("💳 CreditWise")

st.subheader("Smart Loan Approval Prediction System")

st.write(
    """
    **CreditWise** is an AI-powered loan assessment system designed to
    provide a quick and data-driven prediction of loan approval.

    Instead of manually evaluating multiple financial and applicant
    characteristics, CreditWise analyzes the information provided by the
    applicant and uses a trained **Naive Bayes machine learning model**
    to estimate the likelihood of loan approval.

    The system considers factors such as income, age, dependents,
    existing loans, savings, collateral, loan amount, credit score,
    employment status, education, marital status, loan purpose and
    property area.

    CreditWise automatically performs the required data preprocessing
    and feature engineering behind the scenes before sending the
    information to the machine learning model.
    """
)

st.info(
    "💡 CreditWise is designed as a decision-support tool. "
    "The prediction is based on the trained machine learning model "
    "and should not be considered a final lending decision."
)

st.write(
    "Enter the applicant details below to predict loan approval."
)


# =========================================================
# APPLICANT INFORMATION
# =========================================================

st.header("Applicant Information")

st.write(
    "Provide the applicant's basic financial and personal information "
    "to generate a loan approval prediction."
)

col1, col2 = st.columns(2)

with col1:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=50000.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0.0,
        value=0.0
    )

    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=10000.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0.0,
        max_value=900.0,
        value=650.0
    )


with col2:

    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=0.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=100000.0
    )

    loan_term = st.number_input(
        "Loan Term",
        min_value=1,
        value=360
    )

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        value=0.30
    )


# =========================================================
# CATEGORICAL INPUTS
# =========================================================

st.header("Applicant Profile")

col1, col2, col3 = st.columns(3)

with col1:

    education_level = st.selectbox(
        "Education Level",
        [
            "Graduate",
            "Not Graduate"
        ]
    )

    employment_status = st.selectbox(
        "Employment Status",
        [
            "Salaried",
            "Self-employed",
            "Unemployed"
        ]
    )


with col2:

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Married",
            "Single"
        ]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Car",
            "Education",
            "Home",
            "Personal"
        ]
    )


with col3:

    property_area = st.selectbox(
        "Property Area",
        [
            "Rural",
            "Semiurban",
            "Urban"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )

    employer_category = st.selectbox(
        "Employer Category",
        [
            "Government",
            "MNC",
            "Private",
            "Unemployed"
        ]
    )


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔍 Predict Loan Approval"):

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    credit_score_sq = credit_score ** 2

    dti_ratio_sq = dti_ratio ** 2

    applicant_income_log = np.log1p(applicant_income)


    # =====================================================
    # EDUCATION ENCODING
    # =====================================================

    education_encoded = education_encoder.transform(
        [education_level]
    )


    # =====================================================
    # ONE-HOT ENCODING
    # =====================================================

    categorical_data = pd.DataFrame({

        "Employment_Status": [
            employment_status
        ],

        "Marital_Status": [
            marital_status
        ],

        "Loan_Purpose": [
            loan_purpose
        ],

        "Property_Area": [
            property_area
        ],

        "Gender": [
            gender
        ],

        "Employer_Category": [
            employer_category
        ]
    })


    encoded_data = onehot_encoder.transform(
        categorical_data
    )


    encoded_df = pd.DataFrame(
        encoded_data,
        columns=onehot_encoder.get_feature_names_out(
            categorical_data.columns
        )
    )


    # =====================================================
    # CREATE NUMERICAL DATA
    # =====================================================

    input_data = pd.DataFrame({

        "Coapplicant_Income": [
            coapplicant_income
        ],

        "Age": [
            age
        ],

        "Dependents": [
            dependents
        ],

        "Existing_Loans": [
            existing_loans
        ],

        "Savings": [
            savings
        ],

        "Collateral_Value": [
            collateral_value
        ],

        "Loan_Amount": [
            loan_amount
        ],

        "Loan_Term": [
            loan_term
        ],

        "Education_Level": [
            education_encoded[0]
        ],

        "DTI_ratio_sq": [
            dti_ratio_sq
        ],

        "Credit_score_sq": [
            credit_score_sq
        ],

        "Applicant_Income_log": [
            applicant_income_log
        ]
    })


    # =====================================================
    # ADD ONE-HOT FEATURES
    # =====================================================

    input_data = pd.concat(
        [
            input_data,
            encoded_df
        ],
        axis=1
    )


    # =====================================================
    # EXACT 27 FEATURE ORDER FROM TRAINING
    # =====================================================

    expected_columns = [

        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Existing_Loans",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",

        "Education_Level",

        "Employment_Status_Salaried",
        "Employment_Status_Self-employed",
        "Employment_Status_Unemployed",

        "Marital_Status_Single",

        "Loan_Purpose_Car",
        "Loan_Purpose_Education",
        "Loan_Purpose_Home",
        "Loan_Purpose_Personal",

        "Property_Area_Semiurban",
        "Property_Area_Urban",

        "Gender_Male",

        "Employer_Category_Government",
        "Employer_Category_MNC",
        "Employer_Category_Private",
        "Employer_Category_Unemployed",

        "DTI_ratio_sq",
        "Credit_score_sq",
        "Applicant_Income_log"
    ]


    # Reorder columns exactly like training
    input_data = input_data.reindex(
        columns=expected_columns,
        fill_value=0
    )


    # =====================================================
    # SCALE INPUT
    # =====================================================

    input_scaled = scaler.transform(
        input_data
    )


    # =====================================================
    # NAIVE BAYES PREDICTION
    # =====================================================

    prediction = model.predict(
        input_scaled
    )

    probability = model.predict_proba(
        input_scaled
    )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.header("Prediction Result")

    if prediction[0] == 1:

        st.success("✅ Loan Approved")

    else:

        st.error("❌ Loan Not Approved")


    # =====================================================
    # APPROVAL PROBABILITY
    # =====================================================

    approval_probability = probability[0][1] * 100

    st.metric(
        "Loan Approval Probability",
        f"{approval_probability:.2f}%"
    )