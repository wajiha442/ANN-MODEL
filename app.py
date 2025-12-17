 import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("ann_jobs_model.h5")

preprocessor = joblib.load("preprocessor.pkl")

st.title("AI Job Automation Prediction (2030)")
st.write("Enter feature values to predict the automation probability of a job in 2030.")

feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

input_data = np.array([[feature1, feature2, feature3, feature4]])

input_processed = preprocessor.transform(input_data)

if st.button("Predict"):
    pred = model.predict(input_processed)[0][0]
    pred_class = "High Automation Risk" if pred > 0.5 else "Low Automation Risk"
    st.success(f"Predicted Automation Risk: {pred_class} ({pred:.2f})")

