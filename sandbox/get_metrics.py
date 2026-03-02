import json
import pathlib

from mlflow.tracking import MlflowClient

import mlflow

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
mlflow_db_path = BASE_DIR / "mlflow" / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path.as_posix()}")
print("Tracking URI:", mlflow.get_tracking_uri())

client = MlflowClient()
RUN_ID = "2fd3f5fed4a54019a050ad1b769b11bb"

run = client.get_run(RUN_ID)

metrics_dict = run.data.metrics  # dict: metric_name → last_value
params_dict = run.data.params  # dict: param_name → value
tags = run.data.tags

print("\nMetrics (last values):")
for k, v in metrics_dict.items():
    print(f"{k}: {v}")

print("\nParams:")
for k, v in params_dict.items():
    print(f"{k}: {v}")

# ───────────────────────────────────────────────
#          Подключение к PostgreSQL
# ───────────────────────────────────────────────

DB_CONFIG = {
    "dbname": "ml",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432",
}

# Преобразуем словари в JSON-строки
metrics_json = json.dumps(metrics_dict, ensure_ascii=False)
params_json = json.dumps(params_dict, ensure_ascii=False)

# добавить просчёт precision, recall, PSI между train_probs и prod_probs
'''
# Подключение и вставка
cur = conn.cursor()

cur.execute(
    """
    INSERT INTO metrics (
        metrics,
        params,
        created_at
    ) VALUES (
        %(metrics)s,
        %(params)s,
        %(created_at)s
    )
""",
    {"metrics": metrics_json, "params": params_json, "created_at": datetime.utcnow()},
)

#conn.commit()
cur.close()
# conn.close()

print("Запись выполнена → один ряд с полными словарями metrics и params в JSON")
'''
