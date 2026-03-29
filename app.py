import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("student_data.csv")

# Train model
X = df[["hours_studied", "attendance", "sleep_hours"]]
y = df["marks"]

model = LinearRegression()
model.fit(X, y)

# ---------------- UI ---------------- #

# Title
st.title("Student Performance Predictor")

# Description (ADD HERE)
st.write("This app predicts student marks based on study habits.")

# Input sliders
hours = st.slider("Study Hours", 0, 12)
attendance = st.slider("Attendance (%)", 0, 100)
sleep = st.slider("Sleep Hours", 0, 12)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[hours, attendance, sleep]],
                             columns=["hours_studied", "attendance", "sleep_hours"])
    
    prediction = model.predict(input_data)
    st.success(f"Predicted Marks: {prediction[0]:.2f}")

# ---------------- EXTRA FEATURES ---------------- #

# Model insights (ADD HERE)
st.subheader("Model Insights")
st.write("Feature Importance:", model.coef_)

# Graph (ADD HERE)
fig, ax = plt.subplots()
ax.scatter(df["hours_studied"], df["marks"])
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Marks")
st.pyplot(fig)