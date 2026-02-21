# predict.py

import joblib
import numpy as np

def predict_price(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
    model = joblib.load("house_price_model.pkl")
    X_new = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    return model.predict(X_new)[0]

# Example
price = predict_price(5, 20, 6, 1, 1000, 3, 34.2, -118.3)
print("Predicted Price:", price)
