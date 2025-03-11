from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo 
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
import os
from azure.storage.blob import BlobServiceClient

STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")


def upload_to_azure(blob_name, data):
    """Uploads a CSV string to Azure Blob Storage"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        blob_client.upload_blob(data, overwrite=True)
        print(f"Successfully uploaded {blob_name} to Azure Storage")
    
    except Exception as e:
        print(f"Error uploading {blob_name}: {str(e)}")

def fetch_heart_disease_dataset_from_ucimlrepo():
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
    
    # data (as pandas dataframes) 
    features = heart_disease.data.features 
    target = heart_disease.data.targets 

    heart_disease_dataframe = features.join(target)

    return heart_disease_dataframe


def remove_missing_values_in_dataset(heart_disease_dataframe: pd.DataFrame):
    heart_disease_dataframe.dropna(inplace=True)
    return heart_disease_dataframe


def split_the_dataset_into_train_and_test_sets(heart_disease_dataframe: pd.DataFrame):
    features = heart_disease_dataframe.drop(columns='num')
    num = heart_disease_dataframe['num'].apply(lambda num: num if num == 0 else 1)

    smote = SMOTE(random_state=42)
    features, num = smote.fit_resample(features, num)

    features_train, features_test, num_train, num_test = train_test_split(
        features, 
        num, 
        train_size=0.8, 
        stratify=num,
        random_state=42
    )

    return features_train, features_test, num_train, num_test


def standardize_the_features_of_the_dataset(
    features_train,
    features_test,
):
    columns_to_scale = ["age", "trestbps", "thalach", "oldpeak", "thal", "ca"]

    scaler = StandardScaler()

    scaler.fit(features_train[columns_to_scale])

    temp_file_path = "/tmp/scaler.pkl"
    joblib.dump(scaler, temp_file_path)

    if os.getenv("ENVIRONMENT") == "local":
        joblib.dump(scaler, '/opt/airflow/data/scaler.pkl')
    else:
        try:
            with open(temp_file_path, "rb") as data:
                upload_to_azure("scaler.pkl", data)

        except Exception as e:
            print(f"Error uploading scaler to Azure Blob Storage: {e}")

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    features_train[columns_to_scale] = scaler.transform(features_train[columns_to_scale])
    features_test[columns_to_scale] = scaler.transform(features_test[columns_to_scale])

    return features_train, features_test


def one_hot_encode_features_of_the_dataset(
    features_train,
    features_test
):
    columns_to_encode = ["cp", "exang", "slope"]

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    encoder.fit(features_train[columns_to_encode])

    temp_file_path = "/tmp/onehotencoder.pkl"
    joblib.dump(encoder, temp_file_path)

    if os.getenv("ENVIRONMENT") == "local":
        joblib.dump(encoder, '/opt/airflow/data/onehotencoder.pkl')
    else:
        try:
            with open(temp_file_path, "rb") as data:
                upload_to_azure("onehotencoder.pkl", data)

        except Exception as e:
            print(f"Error uploading scaler to Azure Blob Storage: {e}")

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    features_train_encoded_array = encoder.transform(features_train[columns_to_encode])
    encoded_features_train = pd.DataFrame(
        features_train_encoded_array, 
        columns=encoder.get_feature_names_out(columns_to_encode)
    )

    features_test_encoded_array = encoder.transform(features_test[columns_to_encode])
    encoded_features_test = pd.DataFrame(
        features_test_encoded_array, 
        columns=encoder.get_feature_names_out(columns_to_encode)
    )

    features_train = features_train.reset_index(drop=True).join(
        encoded_features_train
    ).drop(columns=columns_to_encode)

    features_test = features_test.reset_index(drop=True).join(
        encoded_features_test
    ).drop(columns=columns_to_encode)

    return features_train, features_test


def export_preprocessed_dataset(
    features_train,
    features_test,
    num_train,
    num_test
):
    train_dataframe = features_train.join(num_train)
    test_dataframe = features_test.join(num_test)

    if os.getenv("ENVIRONMENT") == "local":
        train_dataframe.to_csv("/opt/airflow/data/train_data.csv")
        test_dataframe.to_csv("/opt/airflow/data/test_data.csv")
    else:
        train_csv = train_dataframe.to_csv(index=False)
        test_csv = test_dataframe.to_csv(index=False)

        upload_to_azure("train_data.csv", train_csv)
        upload_to_azure("test_data.csv", test_csv)



def main():
    heart_disease_dataset = fetch_heart_disease_dataset_from_ucimlrepo()

    cleaned_heart_disease_dataset = remove_missing_values_in_dataset(heart_disease_dataset)

    features_train, features_test, num_train, num_test = split_the_dataset_into_train_and_test_sets(cleaned_heart_disease_dataset)

    features_train, features_test = standardize_the_features_of_the_dataset(features_train, features_test)
    features_train, features_test = one_hot_encode_features_of_the_dataset(features_train, features_test)

    export_preprocessed_dataset(features_train, features_test, num_train, num_test)


if __name__ == "__main__":
    main()


