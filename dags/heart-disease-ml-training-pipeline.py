from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# MLflow Tracking URI (replace with your MLflow server URI if remote)
MLFLOW_TRACKING_URI = "http://mlflow_server:5000"  # Change this if MLflow runs on a different host

# Set MLflow Tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Data Loading Function
def load_data():
    train_df = pd.read_csv("/opt/airflow/data/train_data.csv")
    test_df = pd.read_csv("/opt/airflow/data/test_data.csv")
    train_df.to_csv("/tmp/train_data.csv", index=False)
    test_df.to_csv("/tmp/test_data.csv", index=False)

# Model Training Function
def train_model():
    train_df = pd.read_csv("/tmp/train_data.csv", index_col=0)
    test_df = pd.read_csv("/tmp/test_data.csv", index_col=0)
    print(train_df.columns)

    X_train = train_df.drop(columns=['num'])
    X_test = test_df.drop(columns=['num'])
    y_train = train_df['num']
    y_test = test_df['num']

    # Start MLflow Experiment
    with mlflow.start_run(run_name="random_forest_training"):
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
    "ml_training_pipeline_with_mlflow",
    default_args=default_args,
    schedule_interval=None
)

# Define Airflow Tasks
load_task = PythonOperator(task_id="load_data", python_callable=load_data, dag=dag)
train_task = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)

# Define Task Dependencies
load_task >> train_task