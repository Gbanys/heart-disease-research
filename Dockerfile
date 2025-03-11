FROM bitnami/airflow:2.10.5-debian-12-r6

USER root

RUN mkdir /mlflow && chown -R 1001:1001 /mlflow

USER 1001

COPY dags/ /opt/bitnami/airflow/dags/
COPY data/ /opt/bitnami/airflow/data/
COPY data_pipeline/ /opt/bitnami/airflow/data_pipeline/