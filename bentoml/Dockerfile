FROM python:3.12-bullseye

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
RUN mkdir /mlflow

COPY . /home/bentoml
WORKDIR /home/bentoml

EXPOSE 3000

CMD ["sh", "-c", "python import_model_into_bentoml.py && bentoml serve service.py --port 3000"]


