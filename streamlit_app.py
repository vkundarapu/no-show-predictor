import streamlit as st
import pandas as pd

from src.predict import predict_no_show

st.set_page_config(page_title="No-Show Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Medical Appointment No-Show Risk Predictor")
st.markdown(
    """
This tool estimates the probability that a patient will **miss** their scheduled appointment,
based on historical behavior patterns in a public Brazilian hospital dataset.

> ⚠️ **Disclaimer:** This is a student project for learning & demo purposes only.  
> It is *not* intended for real clinical decision-making.
"""
)

st.header("Input Appointment Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Patient Age", min_value=0, max_value=100, value=40)
    waiting_days = st.slider(
        "Days Between Scheduling and Appointment",
        min_value=-1,
        max_value=60,
        value=3,
        help="0 = same-day; negative means scheduled after the appointment date (data quirks).",
    )
    gender = st.selectbox("Gender", options=["F", "M"])
    appt_weekday = st.selectbox(
        "Appointment Weekday",
        options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=1,
    )

with col2:
    scholarship = st.checkbox("On Scholarship (Government assistance)?", value=False)
    hipertension = st.checkbox("Hypertension", value=False)
    diabetes = st.checkbox("Diabetes", value=False)
    alcoholism = st.checkbox("Alcoholism", value=False)
    handcap = st.checkbox("Any Handicap", value=False)
    sms_received = st.checkbox("SMS Reminder Sent", value=True)

st.markdown("---")

if st.button("Predict No-Show Risk"):
    features = {
        "Age": age,
        "WaitingDays": waiting_days,
        "Scholarship": int(scholarship),
        "Hipertension": int(hipertension),
        "Diabetes": int(diabetes),
        "Alcoholism": int(alcoholism),
        "Handcap": int(handcap),
        "SMS_received": int(sms_received),
        "Gender": gender,
        "ApptWeekday": appt_weekday,
    }

    proba = predict_no_show(features)
    percent = proba * 100

    if proba < 0.25:
        risk_label = "Low"
        color = "🟢"
        msg = "Patient is unlikely to miss the appointment."
    elif proba < 0.5:
        risk_label = "Moderate"
        color = "🟡"
        msg = "Consider a reminder, but risk is not extreme."
    else:
        risk_label = "High"
        color = "🔴"
        msg = "Patient has elevated no-show risk. Prioritize outreach."

    st.subheader("Prediction")
    st.metric(label="Estimated No-Show Probability", value=f"{percent:.1f}%")
    st.markdown(f"**Risk Level:** {color} **{risk_label}**")
    st.write(msg)

    with st.expander("View raw feature payload"):
        st.json(features)
else:
    st.info("Fill in the appointment details and click **Predict No-Show Risk**.")
