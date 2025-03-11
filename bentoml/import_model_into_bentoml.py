import bentoml
import mlflow
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_versions = client.get_latest_versions("HeartDiseasePredictor")
if not model_versions:
    raise ValueError("No registered model versions found!")

bento_model = bentoml.mlflow.import_model(
    "heart_disease_predictor",
    model_uri=f"models:/HeartDiseasePredictor/{model_versions[0].version}",
)

print(f"Model imported with tag: {bento_model.tag}")
