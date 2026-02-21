# app.py

import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("house_price_model.pkl")

st.title("California Housing Price Prediction")
st.write("Enter the details below to predict the house price.")

# Input fields
MedInc = st.number_input("Median Income", min_value=0.0, max_value=20.0, value=5.0)
HouseAge = st.number_input("House Age", min_value=0.0, max_value=100.0, value=20.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0, max_value=20.0, value=6.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, max_value=10.0, value=1.0)
Population = st.number_input("Population", min_value=0.0, max_value=50000.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0, max_value=10.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=30.0, max_value=50.0, value=34.2)
Longitude = st.number_input("Longitude", min_value=-130.0, max_value=-100.0, value=-118.3)

# Predict button
if st.button("Predict Price"):
    X_new = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(X_new)[0]
    st.success(f"Predicted Price: {prediction:.2f} (×100,000 USD)")
