import bentoml
import mlflow

mlflow.set_tracking_uri("http://mlflow_server:5000")

# Load the registered MLflow model
bento_model = bentoml.mlflow.import_model(
    "heart_disease_predictor",
    model_uri="models:/HeartDiseasePredictor/latest",
)

print(f"Model imported with tag: {bento_model.tag}")
