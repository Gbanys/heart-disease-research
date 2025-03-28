# heart-disease-research

Type in the terminal in the root of the project dir ```docker compose build``` to build all the images

One thing to watch out for is directory permissions. If a directory has root:root ownership then airflow will not be able to access it. You can use the below two commands to execute the container
as root and change any file permissions

```docker exec -it --user root airflow-scheduler bash```

```docker exec -it --user root airflow-worker bash```

Make sure that the PostgreSQL database is initialized before firing up all the containers:

```docker compose run --rm airflow-scheduler airflow db init```

Then run all the containers:

```docker compose up -d```

Finally, create an admin user to access Airflow UI:

```
docker exec -it airflow_webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

In production:

Get the airflow UI password (username is user):

```kubectl get secret --namespace airflow airflow -o jsonpath="{.data.airflow-password}" | base64 --decode``` 
