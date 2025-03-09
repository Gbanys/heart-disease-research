import bentoml
from bentoml.io import JSON
import numpy as np
import pandas as pd
import joblib

scaler = joblib.load("data/scaler.pkl")
encoder = joblib.load("data/onehotencoder.pkl")

columns_to_scale = ["age", "trestbps", "thalach", "oldpeak", "thal", "ca"]
columns_to_encode = ["cp", "exang", "slope"]

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    http={
    "cors": {
        "enabled": True,
        "access_control_allow_origins": ["*"],
        "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
        "access_control_allow_credentials": True,
        "access_control_allow_headers": ["*"],
        "access_control_max_age": 1200,
        "access_control_expose_headers": ["Content-Length"]
    }
    }
)
class HeartDiseaseClassifier:
    bento_model = bentoml.models.get("heart_disease_predictor:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self, input_data) -> np.ndarray:

        # Convert input to a DataFrame
        input_df = pd.DataFrame([input_data], columns=[
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ])

        # Scale only the selected columns
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

        # Apply one-hot encoding to the selected columns
        encoded_features = encoder.transform(input_df[columns_to_encode])

        # Combine the scaled columns and the encoded features
        input_encoded = np.concatenate([input_df.drop(columns=columns_to_encode).values, encoded_features], axis=1)

        # Make prediction
        probabilities = self.model.predict(input_encoded)

        return {"probabilities": probabilities.tolist()}