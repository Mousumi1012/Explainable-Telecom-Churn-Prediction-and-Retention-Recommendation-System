import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telecom Churn Prediction App")
st.caption("Predict churn risk, identify top reasons, and suggest retention actions.")

pipeline = joblib.load("models/churn_pipeline.pkl")
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


def recommend_action(row, prob):
    actions = []

    if prob < 0.40:
        actions.append("Monitor only")

    if prob >= 0.40 and row["Contract"] == "Month-to-month":
        actions.append("Offer 1-year or 2-year contract discount")

    if prob >= 0.40 and row["MonthlyCharges"] > 80:
        actions.append("Offer pricing review or bundle discount")

    if prob >= 0.40 and row["tenure"] < 12:
        actions.append("Schedule onboarding or retention support call")

    if prob >= 0.40 and row["TechSupport"] == "No" and row["InternetService"] != "No":
        actions.append("Offer free tech support trial")

    if not actions:
        actions.append("General customer loyalty follow-up")

    return actions[:3]


def clean_feature_name(name):
    name = name.replace("cat__", "")
    name = name.replace("num__", "")
    name = name.replace("_", " ")
    return name


def get_top_shap_reasons(input_df, preprocessor, model, top_n=3):
    transformed = preprocessor.transform(input_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(transformed)

    shap_row = shap_values.values[0]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_row
    })

    shap_df = shap_df.sort_values("shap_value", ascending=False)
    top_positive = shap_df[shap_df["shap_value"] > 0].head(top_n)

    return top_positive


with st.form("churn_form"):
    st.subheader("Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{prob:.2%}")

    with col2:
        st.metric("Predicted Label", "Churn" if pred == 1 else "No Churn")

    if prob >= 0.70:
        risk = "High"
    elif prob >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    with col3:
        st.metric("Risk Level", risk)

    if risk == "High":
        st.error("High churn risk detected. Immediate retention action is recommended.")
    elif risk == "Medium":
        st.warning("Medium churn risk detected. Monitor and consider intervention.")
    else:
        st.success("Low churn risk detected. No urgent action required.")

    st.subheader("Business Summary")
    summary = (
        f"This customer has a {prob:.2%} predicted probability of churn "
        f"and is classified as {risk.lower()} risk."
    )
    st.write(summary)

    st.subheader("Top Churn Reasons")
    top_reasons = get_top_shap_reasons(input_df, preprocessor, model, top_n=3)

    if len(top_reasons) == 0:
        st.write("No strong positive churn drivers were detected.")
    else:
        for _, row in top_reasons.iterrows():
            readable_name = clean_feature_name(row["feature"])
            st.write(f"- {readable_name} increased churn risk")

    st.subheader("Recommended Retention Actions")
    actions = recommend_action(input_df.iloc[0], prob)

    for action in actions:
        st.write(f"- {action}")