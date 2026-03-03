import json
import os
import pathlib

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

import mlflow


load_dotenv()

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
mlflow_db_path = BASE_DIR / "mlflow" / "mlflow.db"

mlflow_tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{mlflow_db_path.as_posix()}",
)
mlflow.set_tracking_uri(mlflow_tracking_uri)
print("Tracking URI:", mlflow.get_tracking_uri())

client = MlflowClient()
RUN_ID = os.getenv("MLFLOW_RUN_ID", "2fd3f5fed4a54019a050ad1b769b11bb")

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
    "dbname": os.getenv("ML_METRICS_DB_NAME", "ml"),
    "user": os.getenv("ML_METRICS_DB_USER", os.getenv("POSTGRES_USER", "")),
    "password": os.getenv(
        "ML_METRICS_DB_PASSWORD",
        os.getenv("POSTGRES_PASSWORD", ""),
    ),
    "host": os.getenv("ML_METRICS_DB_HOST", "localhost"),
    "port": os.getenv("ML_METRICS_DB_PORT", "5432"),
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
