import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Predictor", layout="centered")
st.title("🎓 Student Academic Performance Predictor")

# Sidebar info
with st.sidebar:
    st.info("Enter student academic details to predict Pass or Fail using an ML model.", icon="📘")

# Sample data to simulate real training (inside the app)
data = pd.DataFrame({
    'attendance': [85, 60, 90, 40, 70, 95],
    'internal_marks': [75, 50, 88, 35, 65, 90],
    'study_hours': [10, 5, 12, 2, 8, 15],
    'previous_grade': [80, 55, 93, 30, 68, 95],
    'ses': [1, 0, 2, 0, 1, 2],  # 0 = Low, 1 = Medium, 2 = High
    'performance': [1, 0, 1, 0, 1, 1]  # 1 = Pass, 0 = Fail
})

# Train model on-the-fly (no .pkl needed)
X = data.drop("performance", axis=1)
y = data["performance"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Input form
st.subheader("📥 Enter Student Details")
attendance = st.slider("📅 Attendance (%)", 0, 100, 75)
internal_marks = st.slider("📝 Internal Marks", 0, 100, 60)
study_hours = st.slider("📚 Study Hours per Week", 0, 20, 10)
previous_grade = st.slider("📈 Previous Grade (%)", 0, 100, 65)
ses = st.selectbox("🏠 Socioeconomic Status", ["Low", "Medium", "High"])
ses_encoded = {"Low": 0, "Medium": 1, "High": 2}[ses]

if st.button("🔍 Predict Result"):
    input_data = np.array([[attendance, internal_marks, study_hours, previous_grade, ses_encoded]])
    prediction = model.predict(input_data)[0]
    st.success("✅ PASS" if prediction == 1 else "❌ FAIL")

