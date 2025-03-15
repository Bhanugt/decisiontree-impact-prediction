import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ✅ Load the saved model, scaler & encoders
model = joblib.load("decision_tree_impact.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ✅ Streamlit Web App
st.title("Incident Impact Prediction Using Decision Tree")

# ✅ User Inputs
feature_inputs = {}

# Dynamically create input fields based on model features
for feature in scaler.feature_names_in_:
    feature_inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

# ✅ Make Prediction
if st.button("Predict"):
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([feature_inputs])

    # Scale numerical inputs
    input_scaled = scaler.transform(input_df)

    # Predict using the model
    prediction = model.predict(input_scaled)

    # Show result
    st.write(f"🔮 Predicted Impact Level: {prediction[0]}")
