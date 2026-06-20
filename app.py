import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIG

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

# CUSTOM CSS

st.markdown("""
<style>

.block-container{
    padding-top:1rem;
    padding-bottom:0rem;
}

section[data-testid="stSidebar"]{
    width:380px !important;
}

</style>
""", unsafe_allow_html=True)


# LOAD MODEL

model = joblib.load("extra_trees_model.pkl")

encoders = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in [
        "Sex",
        "Housing",
        "Saving accounts",
        "Checking account"
    ]
}


# SESSION STATE

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "proba" not in st.session_state:
    st.session_state.proba = None


# SIDEBAR INPUTS

st.sidebar.header("📋 Applicant Information")

age = st.sidebar.number_input(
    "Age",
    min_value=19,
    max_value=80,
    value=30
)

sex = st.sidebar.selectbox(
    "Sex",
    ["male", "female"]
)

job = st.sidebar.selectbox(
    "Job Level",
    [0, 1, 2, 3]
)

housing = st.sidebar.selectbox(
    "Housing",
    ["own", "rent", "free"]
)

saving_accounts = st.sidebar.selectbox(
    "Saving Account",
    ["little", "moderate", "quite rich", "rich"]
)

checking_account = st.sidebar.selectbox(
    "Checking Account",
    ["little", "moderate", "rich"]
)

credit_amount = st.sidebar.number_input(
    "Credit Amount",
    min_value=250,
    max_value=50000,
    value=1000
)

duration = st.sidebar.number_input(
    "Duration (Months)",
    min_value=1,
    max_value=72,
    value=12
)


# TITLE

st.title("🏦 Credit Risk Prediction Dashboard")

st.write(
    "Predict whether an applicant is likely to be a Good or Bad Credit Risk."
)


# DASHBOARD LAYOUT

left_col, right_col = st.columns([1.5, 1])


# LEFT PANEL

with left_col:

    st.subheader("👤 Applicant Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Age", age)
        st.metric("Job Level", job)
        st.metric("Duration", f"{duration} Months")

    with col2:
        st.metric("Credit Amount", f"£{credit_amount:,}")
        st.metric("Housing", housing.title())
        st.metric("Sex", sex.title())


# RIGHT PANEL

with right_col:

    st.subheader("📈 Risk Probability")

    if st.session_state.proba is not None:

        good_prob = st.session_state.proba[1] * 100
        bad_prob = st.session_state.proba[0] * 100

        st.metric(
            "Good Risk Probability",
            f"{good_prob:.1f}%"
        )

        st.metric(
            "Bad Risk Probability",
            f"{bad_prob:.1f}%"
        )

    else:
        st.info("Run a prediction to view probabilities.")

    st.subheader("ℹ️ Model Information")

    st.info(
        """
Model: Extra Trees Classifier

Features:
• Age
• Sex
• Job
• Housing
• Saving Account
• Checking Account
• Credit Amount
• Duration
"""
    )


# BUTTON

st.markdown("---")

c1, c2, c3 = st.columns([2, 1, 2])

with c2:
    predict_clicked = st.button(
        "🔍 Predict Risk",
        use_container_width=True
    )


# PREDICTION

if predict_clicked:

    with st.spinner("Analyzing applicant profile..."):

        sex_encoded = encoders["Sex"].transform(
            pd.DataFrame({"Sex": [sex]})
        )[0][0]

        housing_encoded = encoders["Housing"].transform(
            pd.DataFrame({"Housing": [housing]})
        )[0][0]

        saving_encoded = encoders["Saving accounts"].transform(
            pd.DataFrame({"Saving accounts": [saving_accounts]})
        )[0][0]

        checking_encoded = encoders["Checking account"].transform(
            pd.DataFrame({"Checking account": [checking_account]})
        )[0][0]

        df = pd.DataFrame({
            "Age": [age],
            "Sex": [sex_encoded],
            "Job": [job],
            "Housing": [housing_encoded],
            "Saving accounts": [saving_encoded],
            "Checking account": [checking_encoded],
            "Credit amount": [credit_amount],
            "Duration": [duration]
        })

        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        st.session_state.prediction = pred
        st.session_state.proba = proba

    st.rerun()

# RESULT SECTION

st.subheader("📊 Prediction Result")

if st.session_state.prediction is None:

    st.info(
        "Enter applicant details and click Predict Risk."
    )

elif st.session_state.prediction == 1:

    st.success(
        "The predicted credit risk is: **Good**"
    )

else:

    st.error(
        "The predicted credit risk is: **Bad**"
    )