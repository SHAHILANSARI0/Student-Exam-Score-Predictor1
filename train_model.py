import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("student_scores.csv")

# Features
X = data[["StudyHours", "Attendance", "PreviousMarks"]]

# Target
y = data["FinalScore"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")
