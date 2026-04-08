# app/bootstrap_airflow.py

from dishka import make_async_container

from app.ioc import get_worker_providers


def create_worker_container():
    return make_async_container(*get_worker_providers())
