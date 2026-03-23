<!-- markdownlint-disable-file MD033 -->

# KLIN (Klin Logical Inference Negation)

KLIN - сервис асинхронной обработки видео для детекции потенциально опасных событий.

## Проект включает

- API на Litestar (прием видео, получение статуса обработки)
- воркер на FastStream + RabbitMQ
- хранение результатов в PostgreSQL
- ML-инференс через X3D -> VideoMAE -> YOLO
- инфраструктуру мониторинга (Prometheus, Grafana, Alertmanager)
- MLflow для экспериментов
- репозиторий ONNX-моделей для Triton (`model_repository/`)

### Архитектура потока

```text
Клиент
  -> POST /api/v1/Klin/upload (multipart)
  -> API uploads the source video to external S3 storage and persists its URI
  -> API сохраняет задачу в PostgreSQL (state=PENDING)
  -> API публикует сообщение в RabbitMQ
  -> Worker downloads the video from S3 for inference
  -> Worker читает очередь и запускает инференс (x3d -> VideoMAE -> YOLO)
  -> Worker обновляет запись в PostgreSQL (FINISHED/ERROR)
  -> Worker deletes the S3 object after processing
  -> Worker отправляет callback на response_url
  -> Клиент читает статус по GET /api/v1/Klin/{id}
```

### Структура репозитория

```text
.
├── app/                  # Основной backend (API, worker, сервисы, DI, БД)
├── data/                 # Данные
├── docker/               # Dockerfiles, compose-файлы и docker-специфичные скрипты
├── docs/                 # Документация и архитектурные материалы
├── helpers/              # Вспомогательные скрипты (экспорт, анализ)
├── model_repository/     # Triton model repository
├── models/               # Локальные веса/каталоги моделей
├── monitoring/           # Конфиги Prometheus/Grafana/Alertmanager
├── sandbox/              # EDA/обучение/эксперименты
└── tests/                # Unit и интеграционные тесты
```

<details>
<summary><h1> Как запустить проект</summary>

## Требования

- `git`
- `python >= 3.10`
- `uv`
- `docker` + Docker Compose plugin
- `make`

### 0) Help | Man | Помощь

```bash
make   # Help по командам из Makefile
```

### 1) Клонирование и окружение

```bash
git clone https://github.com/BookWormDevid/KLIN.git
cd KLIN
make init-env
```

Заполните в `.env` значения вместо `*_change_me`.

Для Triton доступны опциональные переменные:

- `TRITON_IMAGE_TAG` (по умолчанию `24.01-py3`)
- `TRITON_EXIT_ON_ERROR` (по умолчанию `false`)

### 2) Поднять инфраструктуру

```bash
make infra-up
```

### 3) Применить миграции (один раз на БД)

```bash
make migration
```

### 4) Поднять API и worker (docker или локально)

#### docker

```bash
make app-up
```

#### local (В основном для дебага)

```bash
make uv-dev
```

```bash
make start-api-local  # В отдельном терминале
```

```bash
make start-queue-local  # В отдельном терминале
```

### 5) Проверка

- Swagger: `http://localhost/api/docs`
- Live health: `http:localhost/api/v1/klin/health/live`
- Web UI: `http://localhost/frontend`

- Swagger: `http://localhost:8008/api/docs`                    (if local)
- Live health: `http://localhost:8008/api/v1/Klin/health/live` (if local)
- Web UI: `http://localhost:8008/frontend`                     (if local)

#### API

Базовый префикс: `/api/v1/Klin`

| Метод  | Путь            |  Назначение                       |
|--------|-----------------|-----------------------------------|
| `POST` | `/upload`       | Загрузить видео на обработку      |
| `GET`  | `/{klin_id}`    | Получить статус и результат по ID |
| `GET`  | `/`             | Получить последние записи         |
| `GET`  | `/health/live`  | Live-check API                    |
| `GET`  | `/health/ready` | Readiness-check сервиса           |

#### Полезные URL (при полном Docker-запуске)

- API docs: `http://localhost/api/docs`
- RabbitMQ UI: `http://localhost:15672`
- External S3 settings: `S3_ENDPOINT_URL`, `S3_BUCKET_NAME`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`
- Traefik: `http://traefik.localhost`
- Prometheus: `http://prometheus.localhost`
- Grafana: `http://grafana.localhost`
- Alertmanager: `http://alertmanager.localhost`
- PgAdmin: `http://pgadmin.localhost`

### Остановка сервисов

```bash
make docker-stop
make infra-down
make app-down
```

</details>

<details>
<summary><h3> Линтинг и статический анализ</h2></summary>

```bash
make lint
```

Тесты:

```bash
make test
```

Pre-commit:

```bash
make pre-commit
```

</details>

<details>
<summary><h2>Data science и ONNX </h2></summary>

```bash
# EDA по action splits (13 классов, без Normal)
uv run python sandbox/src/eda_action_splits.py

# EDA c классом Normal (14 классов)
uv run python sandbox/src/eda_action_splits.py --include-normal

# Обучение VideoMAE + логирование в MLflow
uv run python sandbox/src/train.py
```

### Экспорт моделей в ONNX для Triton

- Скачайте базовые модели MAE, x3d и yolo в папку models/

```zsh
make uv-dev # установите все зависимости dev (необходимо для экспорта)
```

Утилита: `helpers/*modelname*_to_onnx.py`

Примеры:

```bash
# models/*
# VideoMAE (local folder  -> ONNX)
uv run helpers/videomae_to_onnx.py

# X3D checkpoint (.pt/.pth -> ONNX)
uv run helpers/x3d_pt_to_onnx.py

# YOLO (.pt -> ONNX)
uv run helpers/yolo_pt_to_onnx.py
```

Путь назначения по умолчанию - `model_repository/<model_name>/1/model.onnx`.

Подробности по структуре Triton-репозитория: `model_repository/README.md`.

</details>

## Документация

- ML system design: `docs/ML_System_Design_Doc.md`
- Docker compose files: `docker/docker-compose.yml`, `docker/docker-compose.infra.yml`
- TODO's: `docs/todos.md`

## License

Проект распространяется под лицензией, указанной в `LICENSE`.
