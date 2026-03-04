# KLIN (Klin Logical Inference Negation)

KLIN - сервис асинхронной обработки видео для детекции потенциально опасных событий.

Проект включает:
- API на Litestar (прием видео, получение статуса обработки)
- воркер на FastStream + RabbitMQ
- хранение результатов в PostgreSQL
- ML-инференс через X3D -> VideoMAE -> YOLO
- инфраструктуру мониторинга (Prometheus, Grafana, Alertmanager)
- MLflow для экспериментов
- репозиторий ONNX-моделей для Triton (`model_repository/`)

## Архитектура потока

```text
Клиент
  -> POST /api/v1/Klin/upload (multipart)
  -> API сохраняет задачу в PostgreSQL (state=PENDING)
  -> API публикует сообщение в RabbitMQ
  -> Worker читает очередь и запускает инференс (VideoMAE + YOLO)
  -> Worker обновляет запись в PostgreSQL (FINISHED/ERROR)
  -> Worker отправляет callback на response_url
  -> Клиент читает статус по GET /api/v1/Klin/{id}
```

## Структура репозитория

```text
.
├── app/                  # Основной backend (API, worker, сервисы, DI, БД)
├── data/                 # Данные
├── docs/                 # Документация и архитектурные материалы
├── helpers/              # Вспомогательные скрипты (экспорт, анализ)
├── model_repository/     # Triton model repository
├── models/               # Локальные веса/каталоги моделей
├── monitoring/           # Конфиги Prometheus/Grafana/Alertmanager
├── sandbox/              # EDA/обучение/эксперименты
└── tests/                # Unit и интеграционные тесты
```


## Требования

- `git`
- `python >= 3.10`
- `uv`
- `docker` + Docker Compose plugin
- `make` (опционально)

## Быстрый старт (Docker)

### 1) Клонирование и окружение

```bash
git clone https://github.com/BookWormDevid/KLIN.git
cd KLIN
cp example.env .env
```

Заполните в `.env` значения вместо `*_change_me`.

Минимум для API/воркера:
- `DATABASE_URL`
- `RABBIT_URL`

Если запускаете полный `docker-compose.infra.yml`, заполните также переменные для PostgreSQL, RabbitMQ, Grafana, PgAdmin и Alertmanager.

Для Triton доступны опциональные переменные:
- `TRITON_IMAGE_TAG` (по умолчанию `24.01-py3`)
- `TRITON_EXIT_ON_ERROR` (по умолчанию `false`)

Если у вас RTX 50xx (например RTX 5080), задайте более новый `TRITON_IMAGE_TAG` (например `25.xx-py3`), иначе контейнер `24.01` может вывести `No supported GPU(s) detected`.

### 2) Создать внешнюю Docker-сеть

```bash
docker network create web
```

### 3) Поднять инфраструктуру

Минимально (достаточно для работы API/воркера):

```bash
docker compose -f docker-compose.infra.yml up -d postgresql rabbitmq triton
```

Полный стек (monitoring + traefik + sysadmin UIs):

```bash
docker compose -f docker-compose.infra.yml up --build -d
```

### 4) Поднять API и worker

```bash
docker compose -f docker-compose.yml up --build -d
```

### 5) Применить миграции (один раз на БД)

```bash
docker compose -f docker-compose.yml exec klin_api_development /code/.venv/bin/alembic upgrade head
```

### 6) Проверка

- Swagger: `http://localhost/api/docs`
- Live health: `http://localhost/api/v1/Klin/health/live`

## Локальный запуск (без Docker для API/worker)

### 1) Установка зависимостей

```bash
uv venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

uv sync
uv sync --dev
```

### 2) Поднять зависимости (PostgreSQL + RabbitMQ)

```bash
docker compose -f docker-compose.infra.yml up -d postgresql rabbitmq
```

### 3) Миграции

```bash
uv run alembic upgrade head
```

### 4) Запуск процессов

В отдельных терминалах:

```bash
uv run -m uvicorn --host 0.0.0.0 --port 8000 app.presentation.litestar.run:app
uv run -m faststream run app.presentation.faststream.app:app
```

Или через `Makefile`:

```bash
make start-api
make start-queue
```

## API

Базовый префикс: `/api/v1/Klin`

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/upload` | Загрузить видео на обработку |
| `GET` | `/{klin_id}` | Получить статус и результат по ID |
| `GET` | `/` | Получить последние записи |
| `GET` | `/health/live` | Live-check API |
| `GET` | `/health/ready` | Readiness-check сервиса |

### Пример загрузки видео

```bash
curl -X POST 'http://localhost/api/v1/Klin/upload' \
  -F 'data=@tests/videos/test.mp4' \
  -F 'response_url=https://webhook.site/your-id'
```

Пример ответа:

```json
{
  "id": "0f5d5f5a-5bf4-4afa-8ef1-1b0be4b7ce4a",
  "mae": null,
  "yolo": null,
  "objects": null,
  "all_classes": null,
  "state": "PENDING"
}
```

Проверка статуса:

```bash
curl 'http://localhost/api/v1/Klin/<klin_id>'
```

## Полезные URL (при полном Docker-запуске)

- API docs: `http://localhost/api/docs`
- RabbitMQ UI: `http://localhost:15672`
- Traefik: `http://traefik.localhost`
- Prometheus: `http://prometheus.localhost`
- Grafana: `http://grafana.localhost`
- Alertmanager: `http://alertmanager.localhost`
- PgAdmin: `http://pgadmin.localhost`

## Разработка

Линтинг и статический анализ:

```bash
make
```

Тесты:

```bash
uv run pytest
```

Pre-commit:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
uv run pre-commit run --all-files
uv run pre-commit run --hook-stage pre-push --all-files
```

## EDA и обучение VideoMAE

```bash
# EDA по action splits (13 классов, без Normal)
uv run python sandbox/src/eda_action_splits.py

# EDA c классом Normal (14 классов)
uv run python sandbox/src/eda_action_splits.py --include-normal

# Обучение VideoMAE + логирование в MLflow
uv run python sandbox/src/train.py
```

## Экспорт моделей в ONNX для Triton

Утилита: `helpers/model_exporters.py`

Примеры:

```bash
# VideoMAE (HuggingFace directory -> ONNX)
uv run python helpers/model_exporters.py videomae \
  --model-dir models/videomae-UCF-crime \
  --target-ir-version 9

# X3D checkpoint (.pt/.pth -> ONNX)
uv run python helpers/model_exporters.py x3d \
  --checkpoint /path/to/x3d_checkpoint.pt \
  --target-ir-version 9

# YOLO (.pt -> ONNX)
uv run python helpers/model_exporters.py yolo \
  --weights models/yolov8x.pt \
  --target-ir-version 9
```

Путь назначения по умолчанию - `model_repository/<model_name>/1/model.onnx`.

`--target-ir-version 9` нужен для совместимости с Triton `24.01` (ORT в этом образе поддерживает IR <= 9).

Подробности по структуре Triton-репозитория: `model_repository/README.md`.


## Остановка сервисов

```bash
docker compose -f docker-compose.yml down
docker compose -f docker-compose.infra.yml down
```

## Документация

- ML system design: `docs/ML_System_Design_Doc.md`

## License

Проект распространяется под лицензией, указанной в `LICENSE`.
