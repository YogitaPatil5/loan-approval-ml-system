import streamlit as st
import joblib
import os
import pandas as pd


# =============================
# Page Config
# =============================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Loan Approval Prediction System")
st.write("AI-powered system to evaluate loan eligibility and recommend optimal loan value.")


# =============================
# Load Models (Cached)
# =============================

@st.cache_resource
def load_models():
    model_path = os.path.join(os.getcwd(), "models")

    clf = joblib.load(os.path.join(model_path, "stage_1_rf_classifier_pipeline.pkl"))
    reg = joblib.load(os.path.join(model_path, "stage_2_rf_regression_pipeline.pkl"))

    return clf, reg


clf, reg = load_models()


# =============================
# Sidebar Inputs
# =============================

st.sidebar.header("Applicant Details")

no_of_dependents = st.sidebar.slider("No. of Dependents", 0, 10, 2)

education = st.sidebar.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.sidebar.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.sidebar.number_input(
    "Annual Income",
    min_value=0.0,
    value=500000.0,
    step=50000.0
)

loan_amount = st.sidebar.number_input(
    "Loan Amount Requested",
    min_value=0.0,
    value=100000.0,
    step=50000.0
)

loan_term = st.sidebar.slider(
    "Loan Term (months)",
    6,
    360,
    60
)

cibil_score = st.sidebar.slider(
    "CIBIL Score",
    300,
    900,
    650
)

residential_assets_value = st.sidebar.number_input(
    "Residential Assets Value",
    0.0,
    100000000.0,
    500000.0
)

commercial_assets_value = st.sidebar.number_input(
    "Commercial Assets Value",
    0.0,
    100000000.0,
    200000.0
)

luxury_assets_value = st.sidebar.number_input(
    "Luxury Assets Value",
    0.0,
    100000000.0,
    100000.0
)

bank_asset_value = st.sidebar.number_input(
    "Bank Asset Value",
    0.0,
    100000000.0,
    300000.0
)


# =============================
# Prepare Input Data
# =============================

input_data = pd.DataFrame([{
    "no_of_dependents": no_of_dependents,
    "education": education,
    "self_employed": self_employed,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": residential_assets_value,
    "commercial_assets_value": commercial_assets_value,
    "luxury_assets_value": luxury_assets_value,
    "bank_asset_value": bank_asset_value
}])


# Align column order
input_data = input_data[clf.feature_names_in_]


# =============================
# Prediction Button
# =============================

if st.button("Predict Loan Approval", use_container_width=True):

    approve = clf.predict(input_data)[0]

    st.subheader("Prediction Result")

    if approve == 1:

        st.success("✅ Loan Approved")

        input_reg = input_data.copy()
        input_reg["loan_status"] = "Approve"

        predicted_value = reg.predict(input_reg)[0]

        st.metric(
            label="Recommended Loan Value",
            value=f"₹ {predicted_value:,.2f}"
        )

    else:
        st.error("❌ Loan Rejected")
        st.warning("Applicant is classified as high-risk based on financial indicators.")
