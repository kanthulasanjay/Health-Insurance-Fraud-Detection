import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import tensorflow as tf
from datetime import date


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Health Insurance Fraud Detection",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# LOAD MODEL / PREPROCESSOR / THRESHOLD
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "health_insurance_fraud_model.keras",
        compile=False
    )


@st.cache_resource
def load_preprocessor():
    with open("health_claims_preprocessor.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_threshold():
    with open("health_claims_best_threshold.json", "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return float(data.get("threshold", 0.54))

    return float(data)


model = load_model()
preprocessor = load_preprocessor()
THRESHOLD = load_threshold()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(df):

    df = df.copy()

    # Convert dates
    date_cols = [
        "Claim_Date",
        "Service_Date",
        "Policy_Expiration_Date"
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

    # Feature engineering used during training
    df["Claim_Delay_Days"] = (
        df["Claim_Date"] - df["Service_Date"]
    ).dt.days

    df["Days_To_Policy_Expiration"] = (
        df["Policy_Expiration_Date"] - df["Claim_Date"]
    ).dt.days

    df["Claim_Amount_Per_Procedure"] = (
        df["Claim_Amount"] /
        (df["Number_of_Procedures"] + 1)
    )

    df["Patient_Claim_Risk"] = (
        df["Number_of_Previous_Claims_Patient"] /
        (df["Patient_Age"] + 1)
    )

    df["Provider_Claim_Risk"] = (
        df["Number_of_Previous_Claims_Provider"] /
        (df["Number_of_Previous_Claims_Provider"] + 1)
    )

    df["Total_Previous_Claims"] = (
        df["Number_of_Previous_Claims_Patient"] +
        df["Number_of_Previous_Claims_Provider"]
    )

    df["Patient_Out_of_Pocket"] = (
        df["Deductible_Amount"] +
        df["CoPay_Amount"]
    )

    df["Claim_to_Patient_Cost_Ratio"] = (
        df["Claim_Amount"] /
        (df["Patient_Out_of_Pocket"] + 1)
    )

    df["Claim_Amount_Per_Stay_Day"] = (
        df["Claim_Amount"] /
        (df["Length_of_Stay_Days"] + 1)
    )

    df["Procedures_Per_Stay_Day"] = (
        df["Number_of_Procedures"] /
        (df["Length_of_Stay_Days"] + 1)
    )

    # Date features
    df["Claim_Year"] = df["Claim_Date"].dt.year
    df["Claim_Month"] = df["Claim_Date"].dt.month
    df["Claim_DayOfWeek"] = df["Claim_Date"].dt.dayofweek

    df["Service_Year"] = df["Service_Date"].dt.year
    df["Service_Month"] = df["Service_Date"].dt.month

    # Drop date columns
    df.drop(
        columns=[
            "Claim_Date",
            "Service_Date",
            "Policy_Expiration_Date"
        ],
        inplace=True
    )

    # Drop ID columns exactly as training
    df.drop(
        columns=[
            "Patient_ID",
            "Policy_Number",
            "Claim_ID",
            "Hospital_ID"
        ],
        inplace=True
    )

    return df


# =========================================================
# PREDICTION
# =========================================================

def predict_fraud(input_df):

    # Feature engineering
    processed_df = create_features(input_df)

    # Preprocessing
    transformed = preprocessor.transform(processed_df)

    # Convert sparse matrix if required
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(
        transformed,
        dtype=np.float32
    )

    # Model prediction
    probability = float(
        model.predict(
            transformed,
            verbose=0
        )[0][0]
    )

    prediction = int(
        probability > THRESHOLD
    )

    return probability, prediction


# =========================================================
# TITLE
# =========================================================

st.title("🏥 Health Insurance Claim Fraud Detection")

st.markdown(
    """
    ### Artificial Neural Network based Fraud Detection

    Enter the health insurance claim details below and the trained
    ANN model will estimate the probability that the claim is fraudulent.
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Model:** Artificial Neural Network")
    st.write("**Input Features:** 37")
    st.write("**Decision Threshold:** 0.54")
    st.write("**Test Accuracy:** 85.65%")

    st.divider()

    st.info(
        "The prediction uses the same feature engineering "
        "and preprocessing pipeline used during model training."
    )


# =========================================================
# INPUT FORM
# =========================================================

with st.form("fraud_detection_form"):

    st.subheader("👤 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        patient_id = st.number_input(
            "Patient ID",
            min_value=0,
            value=10000001,
            step=1
        )

    with col2:
        patient_age = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=40
        )

    with col3:
        patient_gender = st.selectbox(
            "Patient Gender",
            ["Male", "Female", "Other"]
        )

    col1, col2 = st.columns(2)

    with col1:
        patient_city = st.text_input(
            "Patient City",
            value="New York"
        )

    with col2:
        patient_state = st.text_input(
            "Patient State",
            value="NY"
        )


    # =====================================================
    # CLAIM INFORMATION
    # =====================================================

    st.subheader("📋 Claim Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        claim_id = st.number_input(
            "Claim ID",
            min_value=0,
            value=10001,
            step=1
        )

    with col2:
        claim_amount = st.number_input(
            "Claim Amount",
            min_value=0.0,
            value=100000.0,
            step=1000.0
        )

    with col3:
        number_of_procedures = st.number_input(
            "Number of Procedures",
            min_value=0,
            value=2,
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        claim_date = st.date_input(
            "Claim Date",
            value=date.today()
        )

    with col2:
        service_date = st.date_input(
            "Service Date",
            value=date.today()
        )

    with col3:
        policy_expiration_date = st.date_input(
            "Policy Expiration Date",
            value=date(
                date.today().year + 2,
                date.today().month,
                date.today().day
            )
        )


    # =====================================================
    # PROVIDER INFORMATION
    # =====================================================

    st.subheader("🏥 Provider Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        hospital_id = st.number_input(
            "Hospital ID",
            min_value=0,
            value=1001,
            step=1
        )

    with col2:
        provider_type = st.selectbox(
            "Provider Type",
            [
                "Hospital",
                "Clinic",
                "Laboratory",
                "Specialist Office"
            ]
        )

    with col3:
        provider_specialty = st.text_input(
            "Provider Specialty",
            value="Cardiology"
        )

    col1, col2 = st.columns(2)

    with col1:
        provider_city = st.text_input(
            "Provider City",
            value="New York"
        )

    with col2:
        provider_state = st.text_input(
            "Provider State",
            value="NY"
        )


    # =====================================================
    # MEDICAL INFORMATION
    # =====================================================

    st.subheader("🩺 Medical Information")

    col1, col2 = st.columns(2)

    with col1:
        diagnosis_code = st.text_input(
            "Diagnosis Code",
            value="I10"
        )

    with col2:
        procedure_code = st.number_input(
            "Procedure Code",
            min_value=0,
            value=1001,
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        admission_type = st.selectbox(
            "Admission Type",
            [
                "Emergency",
                "Elective",
                "Urgent"
            ]
        )

    with col2:
        discharge_type = st.selectbox(
            "Discharge Type",
            [
                "Home",
                "Deceased",
                "Against Medical Advice",
                "Transfer to another facility",
                "Rehab/Skilled Nursing"
            ]
        )

    with col3:
        service_type = st.selectbox(
            "Service Type",
            [
                "Inpatient",
                "Outpatient",
                "Pharmacy",
                "Emergency Room",
                "Laboratory"
            ]
        )


    # =====================================================
    # FINANCIAL INFORMATION
    # =====================================================

    st.subheader("💰 Financial Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        length_of_stay = st.number_input(
            "Length of Stay (Days)",
            min_value=0,
            value=5,
            step=1
        )

    with col2:
        deductible_amount = st.number_input(
            "Deductible Amount",
            min_value=0.0,
            value=2000.0,
            step=100.0
        )

    with col3:
        copay_amount = st.number_input(
            "CoPay Amount",
            min_value=0.0,
            value=500.0,
            step=50.0
        )


    # =====================================================
    # HISTORY
    # =====================================================

    st.subheader("📊 Claim History")

    col1, col2, col3 = st.columns(3)

    with col1:
        previous_patient_claims = st.number_input(
            "Previous Patient Claims",
            min_value=0,
            value=2,
            step=1
        )

    with col2:
        previous_provider_claims = st.number_input(
            "Previous Provider Claims",
            min_value=0,
            value=5,
            step=1
        )

    with col3:
        provider_patient_distance = st.number_input(
            "Provider-Patient Distance (Miles)",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

    claim_submitted_late = st.checkbox(
        "Claim Submitted Late"
    )


    # =====================================================
    # SUBMIT
    # =====================================================

    submitted = st.form_submit_button(
        "🔍 Predict Fraud",
        use_container_width=True
    )


# =========================================================
# PREDICTION RESULT
# =========================================================

if submitted:

    try:

        input_data = pd.DataFrame([{
            "Patient_ID": patient_id,
            "Policy_Number": "WEB_POLICY_001",
            "Claim_ID": claim_id,

            "Claim_Date": str(claim_date),
            "Service_Date": str(service_date),
            "Policy_Expiration_Date": str(
                policy_expiration_date
            ),

            "Claim_Amount": claim_amount,
            "Patient_Age": patient_age,
            "Patient_Gender": patient_gender,
            "Patient_City": patient_city,
            "Patient_State": patient_state,

            "Hospital_ID": hospital_id,

            "Provider_Type": provider_type,
            "Provider_Specialty": provider_specialty,
            "Provider_City": provider_city,
            "Provider_State": provider_state,

            "Diagnosis_Code": diagnosis_code,
            "Procedure_Code": procedure_code,
            "Number_of_Procedures": number_of_procedures,

            "Admission_Type": admission_type,
            "Discharge_Type": discharge_type,
            "Length_of_Stay_Days": length_of_stay,

            "Service_Type": service_type,

            "Deductible_Amount": deductible_amount,
            "CoPay_Amount": copay_amount,

            "Number_of_Previous_Claims_Patient":
                previous_patient_claims,

            "Number_of_Previous_Claims_Provider":
                previous_provider_claims,

            "Provider_Patient_Distance_Miles":
                provider_patient_distance,

            "Claim_Submitted_Late":
                claim_submitted_late
        }])


        # Prediction
        probability, prediction = predict_fraud(
            input_data
        )

        probability_percent = probability * 100


        st.divider()

        st.subheader("📊 Prediction Result")


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Fraud Probability",
                f"{probability_percent:.2f}%"
            )

        with col2:

            st.metric(
                "Decision Threshold",
                f"{THRESHOLD:.2f}"
            )


        # Result
        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT CLAIM"
            )

            st.warning(
                f"The model estimates a fraud probability "
                f"of **{probability_percent:.2f}%**, which is "
                f"above the decision threshold of **{THRESHOLD:.2f}**."
            )

        else:

            st.success(
                "✅ NOT FRAUDULENT"
            )

            st.info(
                f"The model estimates a fraud probability "
                f"of **{probability_percent:.2f}%**, which is "
                f"below the decision threshold of **{THRESHOLD:.2f}**."
            )


        # Probability bar
        st.subheader("Fraud Probability")

        st.progress(
            min(max(probability, 0.0), 1.0)
        )


        # Show input
        with st.expander("View Submitted Claim Data"):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)