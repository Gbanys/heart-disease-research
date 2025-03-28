from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from azure.storage.blob import BlobServiceClient
from io import StringIO
import numpy as np
import os

STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")
CONTAINER_NAME = os.getenv("DATA_CONTAINER_NAME")
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
ENVIRONMENT = os.getenv("ENVIRONMENT")

# Set MLflow Tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def download_from_azure(blob_name):
    """Downloads a CSV file from Azure Blob Storage and returns it as a DataFrame"""
    try:
        # Initialize Azure Blob Client
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

        # Download CSV data
        stream = blob_client.download_blob().readall()
        csv_data = stream.decode("utf-8")

        # Convert CSV data into a Pandas DataFrame
        return pd.read_csv(StringIO(csv_data))
    
    except Exception as e:
        print(f"Error downloading {blob_name}: {str(e)}")
        return None

# Data Loading Function
def load_data():
    """Loads training and testing datasets from Azure Storage"""
    if ENVIRONMENT != "local":
        train_df = download_from_azure("train_data.csv")
        test_df = download_from_azure("test_data.csv")
    else:
        train_df = pd.read_csv("/opt/airflow/data/train_data.csv", index_col=0)
        test_df = pd.read_csv("/opt/airflow/data/test_data.csv", index_col=0)

    train_df.to_csv("/tmp/train_data.csv", index=False)
    test_df.to_csv("/tmp/test_data.csv", index=False)

    return train_df, test_df

def cross_validation():
    train_df = pd.read_csv("/tmp/train_data.csv", index_col=0)
    X_train = train_df.drop(columns=['num'])
    y_train = train_df['num']

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_f1_scores = []

    with mlflow.start_run(run_name="random_forest_kfold_cross_validation"):
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]

            model = RandomForestClassifier(
                max_depth=26,
                max_features=None,
                min_samples_leaf=7,
                min_samples_split=19,
                n_estimators=439,
                random_state=42
            )
            model.fit(X_train_fold, y_train_fold)
            val_preds = model.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, val_preds)
            f1 = f1_score(y_val_fold, val_preds, average='weighted')

            fold_accuracies.append(acc)
            fold_f1_scores.append(f1)
            # Log metrics for this fold
            mlflow.log_metric(f"fold_{fold+1}_accuracy", acc)
            mlflow.log_metric(f"fold_{fold+1}_f1_weighted", f1)
            print(f"Fold {fold+1}: Accuracy = {acc:.4f}, Weighted F1 = {f1:.4f}")

        avg_accuracy = np.mean(fold_accuracies)
        avg_f1 = np.mean(fold_f1_scores)
        # Log average metrics
        mlflow.log_metric("cv_accuracy_mean", avg_accuracy)
        mlflow.log_metric("cv_f1_weighted_mean", avg_f1)
        print(f"Cross-Validation Results - Average Accuracy: {avg_accuracy:.4f}, Average Weighted F1: {avg_f1:.4f}")

# Model Training Function
def final_train_model():
    train_df = pd.read_csv("/tmp/train_data.csv", index_col=0)
    test_df = pd.read_csv("/tmp/test_data.csv", index_col=0)
    print(train_df.columns)

    X_train = train_df.drop(columns=['num'])
    X_test = test_df.drop(columns=['num'])
    y_train = train_df['num']
    y_test = test_df['num']

    # Start MLflow Experiment
    with mlflow.start_run(run_name="random_forest_final_training"):
        model = RandomForestClassifier(
            max_depth=26,
            max_features=None,
            min_samples_leaf=7,
            min_samples_split=19,
            n_estimators=439,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Make Predictions
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1_weighted = f1_score(y_test, predictions, average='weighted')
        f1_macro = f1_score(y_test, predictions, average='weighted')

        # Log Model and Metrics to MLflow
        mlflow.log_param("n_estimators", 439)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("weighted F1 score", f1_weighted)
        mlflow.log_metric("macro F1 score", f1_macro)
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Save Model Locally
        joblib.dump(model, "/tmp/random_forest_model.pkl")

        print(f"Model trained with accuracy: {accuracy}")



# Define Airflow DAG
default_args = {
    "start_date": datetime(2024, 3, 1),
    "catchup": False
}

dag = DAG(
    "heart_disease_ml_training",
    default_args=default_args,
    schedule_interval=None
)

# Define Airflow Tasks
load_task = PythonOperator(task_id="load_data", python_callable=load_data, dag=dag)
cv_kfold_task = PythonOperator(task_id="kfold_cross_validation", python_callable=cross_validation, dag=dag)
final_train_task = PythonOperator(task_id="final_train_model", python_callable=final_train_model, dag=dag)

# Define Task Dependencies
load_task >> cv_kfold_task >> final_train_task