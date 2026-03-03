# KLIN Klin Logical Inference Negation

**Система на ИИ моделях, которыя может находить, предсказывать и выделять агрессию на видеопотоке.**

## Инструкция для запуска и дебага

Не забыть что все консольные команды и запуски source надо делать в папке проекта!

### Виртуальное окружение

```bash
uv venv
uv sync
uv sync --dev # Если надо менять код
pre-commit install # Если надо менять код
```

### Окружениe

! ВАЖНО ! Сделать `.env` файл по `example.env` и заполнить все значения без дефолтных паролей.

```bash
docker compose -f docker-compose.infra.yml up --build -d
docker compose -f docker-compose.yml up --build -d
```

### Общие комманды из makefile

```bash
make # Линтер Ruff + Mypy + PyLint
```

### EDA и обучение VideoMAE (все Action Recognition splits)

```bash
# EDA по всем fold train_*.txt/test_*.txt (по умолчанию 13 классов без Normal)
uv run python sandbox/src/eda_action_splits.py

# Если нужно EDA с Normal классом (14 классов)
uv run python sandbox/src/eda_action_splits.py --include-normal

# Обучение VideoMAE по всем fold (k-fold цикл) + логирование в MLflow
uv run python sandbox/src/train.py
```

MLflow tracking URI берется из переменной окружения `MLFLOW_TRACKING_URI`.
Если переменная не задана, используется локальная SQLite база `mlflow/mlflow.db`.
Для обучения с `Normal` классом установите `INCLUDE_NORMAL_CLASS=1` перед запуском train.

### API

**<http://0.0.0.0:8008/api/docs>**

### Web Site

**<http://0.0.0.0:8080>**

### Alertmanager

**<http://alertmanager.localhost>**

### Что делать дальше?

Тест процессора. Изменить формат чтения видео. Оценивать каждый чанк и выводить результат по нему. Вывод всех классов что нашёл со всех чанков. Всё вместе положить в json.
get в каком формате отправить данные - json(time: tuple(start: float, end: float), answer: str, confident: float), objects, bounding box, timesteps всё с yolo.
get_filltered фильтровать по запросу answer в бд.

1. Bounding box от yolo хранить и выводить xyxyn

#### ML System Design Doc +-

#### Оформленный репозиторий сервиса

- Github/Gitlab репозиторий с историей коммитов +-
- Код в Clean Architecture +
- Покрытие юнит-тестами > 70% -
- Есть понятный Readme -

#### Оформленный сервис

- Код имеет pin зависимостей через uv +
- Есть dockerfile +-
- Настроен CI/CD деплоя на удаленный сервер с push моделью передачи -
- Образы залиты в хранилище образов -
- При сборке в пайплайне успешно проходят линтеры: flake8, isort, pylint, mypy, black. Отключение линтеров в коде не допускается +
- Секреты прокинуты через .env и хранилища секретов +

#### Батчевый сервис

- Есть рабочий Airflow DAG, основанный на коде проекта
- (Нужен отдельный docker образ для чего то вроде triton)
- (DAG по дате исопользовать)
- Использован DockerOperator -
- Обеспечена идемпотентность -
- Продемонстрирован бэкфилл -

#### REST-сервис?

- Имеется не менее трех эндпоинтов +
- Есть healthcheck +
- Есть OpenAPI документация в Swagger +

#### Мониторинг ?

- Логи приложения пишутся в Prometheus +
- Логи отрисовываются в Grafana +
- Настроен алертинг на почту/в телеграм -   (grafana | github actions)
- Логируются ML-метрики качества, а также PSI и CSI, где возможно +-

#### Архитектура решений ?

- Есть ADR на батчевый и онлайн-сервисы -
- Имеется схема в нотации C4 -

#### Общие требования

- Использование отчужденного postgres как базы +
- Использование S3 для хранения артефактов, ничего локального -
- Можно использовать внешние модели, но требования к мониторингу их качества обязательны (техническое качество, ML-качество) +-
- Минус балл за любые креды (пароли/ключи) в гите, дефолтные креды на сервисах +
