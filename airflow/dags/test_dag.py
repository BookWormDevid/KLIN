from datetime import datetime

from airflow.operators.bash import BashOperator  # type: ignore

from airflow import DAG


with DAG(
    dag_id="test_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # type: ignore
    catchup=False,
) as dag:
    t1 = BashOperator(task_id="hello", bash_command="echo Hello Airflow 3!")
