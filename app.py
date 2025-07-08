
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Academic Performance Predictor")

with st.sidebar:
    st.info("Enter student academic details to predict Pass or Fail using a trained ML model.", icon="📘")

st.subheader("📥 Enter Student Details")
attendance = st.slider("📅 Attendance (%)", 0, 100, 75)
internal_marks = st.slider("📝 Internal Marks (out of 100)", 0, 100, 60)
study_hours = st.slider("📚 Study Hours per Week", 0, 20, 10)
previous_grade = st.slider("📈 Previous Grade (%)", 0, 100, 65)
ses = st.selectbox("🏠 Socioeconomic Status", ["Low", "Medium", "High"])
ses_encoded = {"Low": 0, "Medium": 1, "High": 2}[ses]

model = joblib.load("student_model_py313.pkl")
if st.button("🔍 Predict Result"):
    input_data = np.array([[attendance, internal_marks, study_hours, previous_grade, ses_encoded]])
    prediction = model.predict(input_data)[0]
    result = "✅ PASS" if prediction == 1 else "❌ FAIL"
    st.success(f"🎯 Predicted Result: {result}")
