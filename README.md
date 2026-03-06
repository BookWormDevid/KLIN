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
  -> Worker читает очередь и запускает инференс (x3d -> VideoMAE -> YOLO)
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
- `make`

<details>
<summary>## Как запустить проект</summary>

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
make infra-up
```

### 4) Применить миграции (один раз на БД)

```bash
make migration
```

### 5) Поднять API и worker (docker или локально)

#### docker

```bash
make app-up
```

#### local (В основном для дебага)

```bash
make uv-dev
```

В отдельных терминалах

```bash
make start-api-local
```

```bash
make start-queue-local
```

### 6) Проверка

- Swagger: `http://localhost/api/docs`
- Live health: `http:localhost/api/v1/klin/health/live`
- Web UI: `http://localhost/frontend`

- Swagger: `http://localhost:8008/api/docs`                    (if local)
- Live health: `http://localhost:8008/api/v1/Klin/health/live` (if local)
- Web UI: `http://localhost:8008/frontend`                     (if local)

</details>

## API

Базовый префикс: `/api/v1/Klin`

| Метод  | Путь            |  Назначение                       |
|--------|-----------------|-----------------------------------|
| `POST` | `/upload`       | Загрузить видео на обработку      |
| `GET`  | `/{klin_id}`    | Получить статус и результат по ID |
| `GET`  | `/`             | Получить последние записи         |
| `GET`  | `/health/live`  | Live-check API                    |
| `GET`  | `/health/ready` | Readiness-check сервиса           |

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
  --checkpoint models/pre_trained_x3d_model.pt \
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
make docker-stop
make infra-down
make app-down
```

## Документация

- ML system design: `docs/ML_System_Design_Doc.md`

## License

Проект распространяется под лицензией, указанной в `LICENSE`.
