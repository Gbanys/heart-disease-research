from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

# Import your pipeline functions (make sure these functions are accessible)
from data_pipeline.data_pipeline import (
    fetch_heart_disease_dataset_from_ucimlrepo,
    remove_missing_values_in_dataset,
    split_the_dataset_into_train_and_test_sets,
    standardize_the_features_of_the_dataset,
    one_hot_encode_features_of_the_dataset,
    export_preprocessed_dataset
)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'schedule_interval': None
}

# Instantiate the DAG
dag = DAG(
    'heart_disease_data_pipeline',
    default_args=default_args,
    description='Process the UCI Heart Disease dataset',
    schedule_interval=None,  # or set to a cron schedule if desired
    catchup=False
)

# Task 1: Fetch the dataset and save it to disk
def fetch_task(**kwargs):
    df = fetch_heart_disease_dataset_from_ucimlrepo()
    df.to_csv('/tmp/heart_disease.csv', index=False)

fetch_dataset = PythonOperator(
    task_id='fetch_dataset',
    python_callable=fetch_task,
    dag=dag
)

# Task 2: Remove missing values
def remove_missing_task(**kwargs):
    df = pd.read_csv('/tmp/heart_disease.csv')
    df_cleaned = remove_missing_values_in_dataset(df)
    df_cleaned.to_csv('/tmp/heart_disease_cleaned.csv', index=False)

remove_missing = PythonOperator(
    task_id='remove_missing',
    python_callable=remove_missing_task,
    dag=dag
)

# Task 3: Split the dataset into train and test sets
def split_task(**kwargs):
    df_cleaned = pd.read_csv('/tmp/heart_disease_cleaned.csv')
    features_train, features_test, num_train, num_test = split_the_dataset_into_train_and_test_sets(df_cleaned)
    features_train.to_csv('/tmp/features_train.csv', index=False)
    features_test.to_csv('/tmp/features_test.csv', index=False)
    num_train.to_csv('/tmp/num_train.csv', index=False)
    num_test.to_csv('/tmp/num_test.csv', index=False)


split_dataset = PythonOperator(
    task_id='split_dataset',
    python_callable=split_task,
    dag=dag
)

# Task 4: Standardize the features
def standardize_task(**kwargs):
    features_train = pd.read_csv('/tmp/features_train.csv')
    features_test = pd.read_csv('/tmp/features_test.csv')
    features_train_std, features_test_std = standardize_the_features_of_the_dataset(features_train, features_test)
    features_train_std.to_csv('/tmp/features_train_standardized.csv', index=False)
    features_test_std.to_csv('/tmp/features_test_standardized.csv', index=False)

standardize = PythonOperator(
    task_id='standardize_features',
    python_callable=standardize_task,
    dag=dag
)

# Task 5: One-Hot Encode features
def encode_task(**kwargs):
    features_train = pd.read_csv('/tmp/features_train_standardized.csv')
    features_test = pd.read_csv('/tmp/features_test_standardized.csv')
    features_train_enc, features_test_enc = one_hot_encode_features_of_the_dataset(features_train, features_test)
    features_train_enc.to_csv('/tmp/features_train_encoded.csv', index=False)
    features_test_enc.to_csv('/tmp/features_test_encoded.csv', index=False)

one_hot_encode = PythonOperator(
    task_id='one_hot_encode_features',
    python_callable=encode_task,
    dag=dag
)

def export_data_task(**kwargs):
    features_train_enc = pd.read_csv('/tmp/features_train_encoded.csv')
    features_test_enc = pd.read_csv('/tmp/features_test_encoded.csv')
    num_train = pd.read_csv('/tmp/num_train.csv')
    num_test = pd.read_csv('/tmp/num_test.csv')
    export_preprocessed_dataset(features_train_enc, features_test_enc, num_train, num_test)

export_data = PythonOperator(
    task_id='export_data',
    python_callable=export_data_task,
    dag=dag
)

fetch_dataset >> remove_missing >> split_dataset >> standardize >> one_hot_encode >> export_data
