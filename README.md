# KLIN (Klin Logical Inference Negation)

Сервис для детекции агрессии на видео:
- API на Litestar
- воркер на FastStream + RabbitMQ
- PostgreSQL
- мониторинг (Prometheus, Grafana, Alertmanager)
- эксперименты и метрики через MLflow

## Требования

- `git`
- `python` 3.10+
- `uv`
- `docker` + Docker Compose plugin
- `make` (опционально, для удобных команд)

## 1) Установка uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2) Установка зависимостей проекта

```bash
git clone https://github.com/BookWormDevid/KLIN.git
cd KLIN

uv venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

uv sync
uv sync --dev # Если планируете менять код
```

## 3) Настройка `.env`

```bash
cp example.env .env
# Windows: copy example.env .env
```

Что обязательно сделать:
- заменить все значения вида `*_change_me`
- убедиться, что `DATABASE_URL` и `RABBIT_URL` указывают на рабочие сервисы
- при обучении настроить `MLFLOW_TRACKING_URI` (или оставить локальный SQLite из примера)

Минимум для запуска API/воркера:
- `DATABASE_URL`
- `RABBIT_URL`

Для полного запуска `docker-compose.infra.yml` заполните все переменные из `example.env`.

## 4) Запуск в Docker (рекомендуется)

Сеть `web` в compose-файлах объявлена как `external`, поэтому ее нужно создать один раз:

```bash
docker network create web
```

Поднять инфраструктуру:

```bash
docker compose -f docker-compose.infra.yml up --build -d
```

Поднять API и воркер:

```bash
docker compose -f docker-compose.yml up --build -d
```

## 5) Локальный запуск приложения (без Docker для API/воркера)

Поднимите зависимости (PostgreSQL и RabbitMQ), затем запускайте процессы из локального окружения:

```bash
docker compose -f docker-compose.infra.yml up -d postgresql rabbitmq

uv run -m uvicorn --host 0.0.0.0 --port 8000 app.presentation.litestar.run:app
uv run -m faststream run app.presentation.faststream.app:app
```

## 6) Полезные URL

При запуске через Docker + Traefik:
- API docs: `http://localhost/api/docs`
- Health: `http://localhost/api/v1/Klin/health/live`
- RabbitMQ UI: `http://localhost:15672`
- Prometheus: `http://prometheus.localhost`
- Grafana: `http://grafana.localhost`
- Alertmanager: `http://alertmanager.localhost`
- PgAdmin: `http://pgadmin.localhost`

При локальном запуске API без Traefik:
- API docs: `http://localhost:8000/api/docs`

## 7) Команды для разработки

Линтеры/статический анализ:

```bash
make
```

Тесты:

```bash
uv run pytest
```

Pre-commit hooks:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

uv run pre-commit run --all-files
uv run pre-commit run --hook-stage pre-push --all-files
```

## 8) EDA и обучение VideoMAE

```bash
# EDA по action splits (13 классов, без Normal)
uv run python sandbox/src/eda_action_splits.py

# EDA c классом Normal (14 классов)
uv run python sandbox/src/eda_action_splits.py --include-normal

# Обучение VideoMAE + логирование в MLflow
uv run python sandbox/src/train.py
```

## 9) Остановка сервисов

```bash
docker compose -f docker-compose.yml down # или stop
docker compose -f docker-compose.infra.yml down # или stop, что удобнее
```

## Документация

- ML system design: `docs/ML_System_Design_Doc.md`
