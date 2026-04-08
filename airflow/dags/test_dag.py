from datetime import datetime

from airflow.models.dag import DAG
from airflow.providers.standard.operators.bash import BashOperator


with DAG(
    dag_id="test_dag",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    t1 = BashOperator(task_id="hello", bash_command="echo Hello Airflow 3!")
