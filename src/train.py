# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# 1) Load dataset
df = pd.read_csv("../data/housing.csv")

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 2) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.2,
        max_depth=4,
        random_state=42
    ))
])

# 4) Train model
pipeline.fit(X_train, y_train)

# 5) Save model
joblib.dump(pipeline, "house_price_model.pkl")

print("Model saved as house_price_model.pkl")
