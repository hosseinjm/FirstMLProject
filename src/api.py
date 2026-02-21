# api.py

from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model once (efficient)
model = joblib.load("house_price_model.pkl")

@app.get("/")
def home():
    return {"message": "California Housing Price Prediction API is running"}

@app.post("/predict")
def predict_price(
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    AveBedrms: float,
    Population: float,
    AveOccup: float,
    Latitude: float,
    Longitude: float
):
    X_new = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(X_new)[0]
    return {"predicted_price": float(prediction)}
