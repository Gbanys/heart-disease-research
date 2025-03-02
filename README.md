# heart-disease-research

One thing to watch out for is directory permissions. If a directory has root:root ownership then airflow will not be able to access it. You can use the below two commands to execute the container
as root and change any file permissions

```docker exec -it --user root airflow-scheduler bash```

```docker exec -it --user root airflow-worker bash```