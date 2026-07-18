import streamlit as st
import joblib
import pandas as pd
import os

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    chart_available = True
except:
    chart_available = False

# Page Settings
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load Model
model = joblib.load("model.pkl")

# Title
st.title("🎓 Student Exam Score Predictor")
st.write("### Enter Student Details")

# Inputs
student_name = st.text_input("👤 Student Name")
subject = st.text_input("📚 Subject Name")

study_hours = st.slider("📖 Study Hours", 0, 12, 5)
attendance = st.slider("📅 Attendance (%)", 0, 100, 80)
previous_marks = st.slider("📝 Previous Marks", 0, 100, 60)

# Predict Button
if st.button("🚀 Predict Score"):

    prediction = model.predict([[study_hours, attendance, previous_marks]])
    score = float(prediction[0])

    # Limit Score
    if score < 0:
        score = 0
    elif score > 100:
        score = 100

    # Grade
    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    # Pass / Fail
    if score >= 40:
        status = "PASS ✅"
    else:
        status = "FAIL ❌"

    # Display Results
    st.success(f"👤 Student Name : {student_name}")
    st.success(f"📚 Subject : {subject}")
    st.success(f"📊 Predicted Score : {score:.2f}")
    st.success(f"🏆 Grade : {grade}")
    st.success(f"📌 Result : {status}")

    # Progress Bar
    st.subheader("Prediction Progress")
    st.progress(int(score))

    # Chart
    if chart_available:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(
            ["Study Hours", "Attendance", "Previous Marks", "Predicted Score"],
            [study_hours, attendance, previous_marks, score]
        )
        ax.set_ylabel("Values")
        ax.set_title("Student Performance")
        st.pyplot(fig)
    else:
        st.warning("Matplotlib is not installed. Chart cannot be displayed.")

    # Save History
    history = pd.DataFrame({
        "Student":[student_name],
        "Subject":[subject],
        "Study Hours":[study_hours],
        "Attendance":[attendance],
        "Previous Marks":[previous_marks],
        "Predicted Score":[round(score,2)],
        "Grade":[grade],
        "Status":[status]
    })

    if os.path.exists("history.csv"):
        history.to_csv("history.csv", mode="a", header=False, index=False)
    else:
        history.to_csv("history.csv", index=False)

    # Show Table
    st.subheader("Prediction Summary")
    st.dataframe(history)

    # Download Button
    csv = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Result",
        data=csv,
        file_name="student_result.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Made with ❤️ using Python, Streamlit & Machine Learning")
