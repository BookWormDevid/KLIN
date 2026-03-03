"""
Тесты бизнес-логики
"""
# pylint: disable="import-error"
# pylint: disable="redefined-outer-name"
# pylint: disable="too-many-positional-arguments"
# pylint: disable="unused-argument"
# pylint: disable="too-many-arguments"

import os
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.dto import KlinProcessDto, KlinReadDto, KlinUploadDto
from app.application.services.klin import KlinService
from app.models import KlinModel, ProcessingState


# Фикстуры — общие заготовки для всех тестов
@pytest.fixture
def mock_repository():
    """Мок репозитория — имитирует работу с базой данных."""
    return AsyncMock()


@pytest.fixture
def mock_inference_service():
    """Мок сервиса инференса — имитирует запуск YOLO + MAE."""
    return AsyncMock()


@pytest.fixture
def mock_process_producer():
    """Мок продюсера сообщений в очередь (например, RabbitMQ / Kafka)."""
    return AsyncMock()


@pytest.fixture
def mock_callback_sender():
    """Мок отправителя колбэков на внешний URL."""
    return AsyncMock()


@pytest.fixture
def klin_service(
    mock_repository, mock_inference_service, mock_process_producer, mock_callback_sender
):
    """
    Фикстура основного сервиса KlinService.
    Все зависимости заменены моками → изолированные тесты.
    """
    return KlinService(
        _klin_repository=mock_repository,
        _klin_inference_service=mock_inference_service,
        _klin_process_producer=mock_process_producer,
        _klin_callback_sender=mock_callback_sender,
    )


@pytest.fixture
def sample_klin_model():
    """
    Пример объекта KlinModel для повторного использования в тестах.
    Имитирует запись в БД в состоянии PENDING.
    """
    return KlinModel(
        id=uuid.uuid4(),
        response_url="https://example.com/callback",
        video_path="/tmp/test_video.mp4",
        state=ProcessingState.PENDING,
        created_at=datetime.now(),
        mae=None,
        yolo=None,
        objects=[],
        all_classes=[],
    )


# Тесты метода klin_image (создание записи + отправка в очередь)
@pytest.mark.asyncio
class TestKlinImage:
    """Группа тестов для метода klin_image
    загрузка видео и постановка задачи в очередь.
    """

    async def test_klin_image_success(
        self, klin_service, mock_repository, mock_process_producer
    ):
        """
        Проверяет счастливый сценарий:
        - создаётся запись в БД
        - отправляется задача в очередь
        - возвращается созданный объект
        """
        # Arrange — подготовка входных данных и моков
        upload_dto = KlinUploadDto(
            response_url="https://example.com/callback",
            video_path="/tmp/test_video.mp4",
        )

        expected_klin = KlinModel(
            id=uuid.uuid4(),
            response_url=upload_dto.response_url,
            video_path=upload_dto.video_path,
            state=ProcessingState.PENDING,
        )
        mock_repository.create.return_value = expected_klin

        # Act — вызов метода
        result = await klin_service.klin_image(upload_dto)

        # Assert — проверки
        assert result == expected_klin
        assert result.state == ProcessingState.PENDING
        mock_repository.create.assert_called_once()
        mock_process_producer.send.assert_called_once_with(
            KlinProcessDto(klin_id=expected_klin.id)
        )

    async def test_klin_image_repository_error(
        self, klin_service, mock_repository, mock_process_producer
    ):
        """
        Проверяет обработку ошибки при создании записи в БД:
        - исключение пробрасывается наверх
        - сообщение в очередь НЕ отправляется
        """
        upload_dto = KlinUploadDto(
            response_url="https://example.com/callback",
            video_path="/tmp/test_video.mp4",
        )
        mock_repository.create.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await klin_service.klin_image(upload_dto)

        mock_process_producer.send.assert_not_called()


# Тесты метода perform_klin — основной этап обработки видео
@pytest.mark.asyncio
class TestPerformKlin:
    """
    Группа тестов для метода perform_klin — запуск инференса и обновление статуса.
    """

    async def test_perform_klin_success(
        self,
        klin_service,
        mock_repository,
        mock_inference_service,
        mock_callback_sender,
        sample_klin_model,
    ):
        """
        Успешная обработка:
        - модель найдена
        - инференс прошёл
        - поля обновлены
        - статус → FINISHED
        - отправлен колбэк
        - обновление сохранено в БД
        """
        mock_repository.get_by_id.return_value = sample_klin_model

        process_result = AsyncMock()
        process_result.mae = {"predictions": [{"time": 0, "value": 0.5}]}
        process_result.yolo = [{"bbox": [1, 2, 3, 4], "class": "person"}]
        process_result.objects = ["person", "car"]
        process_result.all_classes = ["person", "car", "dog"]

        mock_inference_service.analyze.return_value = process_result

        await klin_service.perform_klin(sample_klin_model.id)

        assert sample_klin_model.mae == process_result.mae
        assert sample_klin_model.yolo == process_result.yolo
        assert sample_klin_model.objects == process_result.objects
        assert sample_klin_model.all_classes == process_result.all_classes
        assert sample_klin_model.state == ProcessingState.FINISHED

        mock_repository.update.assert_called_once_with(sample_klin_model)
        mock_callback_sender.post_consumer.assert_called_once_with(sample_klin_model)

    async def test_perform_klin_with_none_objects(
        self,
        klin_service,
        mock_repository,
        mock_inference_service,
        mock_callback_sender,
        sample_klin_model,
    ):
        """
        Проверяет, что при None в objects/all_classes
        сервис преобразует их в пустые списки.
        Это важно для консистентности данных в БД.
        """
        mock_repository.get_by_id.return_value = sample_klin_model

        process_result = AsyncMock()
        process_result.mae = {"predictions": [{"time": 0, "value": 0.5}]}
        process_result.yolo = [{"bbox": [1, 2, 3, 4], "class": "person"}]
        process_result.objects = None
        process_result.all_classes = None

        mock_inference_service.analyze.return_value = process_result

        await klin_service.perform_klin(sample_klin_model.id)

        assert sample_klin_model.objects == []
        assert sample_klin_model.all_classes == []
        assert sample_klin_model.state == ProcessingState.FINISHED

    async def test_perform_klin_error(
        self,
        klin_service,
        mock_repository,
        mock_inference_service,
        mock_callback_sender,
        sample_klin_model,
    ):
        """
        Проверяет обработку ошибки инференса:
        - статус → ERROR
        - сообщение об ошибке сохраняется в mae
        - колбэк всё равно отправляется
        - обновление БД вызывается
        """
        mock_repository.get_by_id.return_value = sample_klin_model
        mock_inference_service.analyze.side_effect = ValueError("Inference failed")

        await klin_service.perform_klin(sample_klin_model.id)

        assert sample_klin_model.mae == "Inference failed"
        assert sample_klin_model.state == ProcessingState.ERROR
        mock_callback_sender.post_consumer.assert_called_once_with(sample_klin_model)
        mock_repository.update.assert_called_once_with(sample_klin_model)

    async def test_perform_klin_file_deletion_success(
        self,
        klin_service,
        mock_repository,
        mock_inference_service,
        mock_callback_sender,
        sample_klin_model,
        monkeypatch,
    ):
        """
        Проверяет удаление временного видео-файла после успешной обработки.
        """
        mock_repository.get_by_id.return_value = sample_klin_model
        mock_inference_service.analyze.return_value = AsyncMock()

        mock_exists = MagicMock(return_value=True)
        mock_remove = MagicMock()

        monkeypatch.setattr(os.path, "exists", mock_exists)
        monkeypatch.setattr(os, "remove", mock_remove)

        await klin_service.perform_klin(sample_klin_model.id)

        mock_exists.assert_called_once_with(sample_klin_model.video_path)
        mock_remove.assert_called_once_with(sample_klin_model.video_path)

    async def test_perform_klin_file_deletion_error(
        self,
        klin_service,
        mock_repository,
        mock_inference_service,
        mock_callback_sender,
        sample_klin_model,
        monkeypatch,
        caplog,
    ):
        """
        Проверяет, что при ошибке удаления файла:
        - обработка не падает
        - в лог пишется предупреждение
        """
        mock_repository.get_by_id.return_value = sample_klin_model
        mock_inference_service.analyze.return_value = AsyncMock()

        mock_exists = MagicMock(return_value=True)
        mock_remove = MagicMock(side_effect=OSError("Permission denied"))

        monkeypatch.setattr(os.path, "exists", mock_exists)
        monkeypatch.setattr(os, "remove", mock_remove)

        await klin_service.perform_klin(sample_klin_model.id)

        assert "Failed to delete temp file" in caplog.text

    async def test_perform_klin_update_error(
        self,
        klin_service,
        mock_repository,
        mock_inference_service,
        mock_callback_sender,
        sample_klin_model,
        caplog,
    ):
        """
        Проверяет устойчивость к ошибке обновления БД:
        - колбэк всё равно отправляется
        - ошибка логируется
        """
        mock_repository.get_by_id.return_value = sample_klin_model
        mock_inference_service.analyze.return_value = AsyncMock()
        mock_repository.update.side_effect = Exception("Update failed")

        await klin_service.perform_klin(sample_klin_model.id)

        assert "Failed to persist klin state" in caplog.text


# Тесты метода get_inference_status
@pytest.mark.asyncio
class TestGetInferenceStatus:
    """Тесты получения статуса обработки по ID."""

    async def test_get_inference_status_success(
        self, klin_service, mock_repository, sample_klin_model
    ):
        """
        Успешное получение статуса готовой задачи.
        Проверяем, что все поля DTO заполнены корректно.
        """
        sample_klin_model.state = ProcessingState.FINISHED
        sample_klin_model.mae = {"predictions": [{"time": 0, "value": 0.5}]}
        sample_klin_model.yolo = [{"bbox": [1, 2, 3, 4], "class": "person"}]
        sample_klin_model.objects = ["person"]
        sample_klin_model.all_classes = ["person", "car"]

        mock_repository.get_by_id.return_value = sample_klin_model

        result = await klin_service.get_inference_status(sample_klin_model.id)

        assert isinstance(result, KlinReadDto)
        assert result.id == sample_klin_model.id
        assert result.state == ProcessingState.FINISHED
        assert result.mae == sample_klin_model.mae
        assert result.yolo == sample_klin_model.yolo
        assert result.objects == sample_klin_model.objects
        assert result.all_classes == sample_klin_model.all_classes

    async def test_get_inference_status_not_found(self, klin_service, mock_repository):
        """
        Проверяет, что при отсутствии записи выбрасывается ValueError.
        """
        klin_id = uuid.uuid4()
        mock_repository.get_by_id.return_value = None

        with pytest.raises(ValueError, match=f"MAE {klin_id} not found"):
            await klin_service.get_inference_status(klin_id)


# Тесты метода get_n_inferences
@pytest.mark.asyncio
class TestGetNInferences:
    """Тесты получения списка последних N задач."""

    async def test_get_n_inferences_success(self, klin_service, mock_repository):
        """
        Успешное получение списка из N записей.
        """
        count = 3
        expected_inferences = [
            KlinModel(id=uuid.uuid4(), video_path=f"/tmp/test_{i}.mp4")
            for i in range(count)
        ]
        mock_repository.get_first_n.return_value = expected_inferences

        result = await klin_service.get_n_imferences(count)

        assert result == expected_inferences
        assert len(result) == count
        mock_repository.get_first_n.assert_called_once_with(count)

    async def test_get_n_inferences_empty(self, klin_service, mock_repository):
        """
        Проверяет возврат пустого списка, если записей нет.
        """
        count = 5
        mock_repository.get_first_n.return_value = []

        result = await klin_service.get_n_imferences(count)

        assert result == []
        assert len(result) == 0

    async def test_get_n_inferences_zero(self, klin_service, mock_repository):
        """
        Пограничный случай: запрашиваем 0 записей → пустой список.
        """
        count = 0
        mock_repository.get_first_n.return_value = []

        result = await klin_service.get_n_imferences(count)

        assert result == []
        mock_repository.get_first_n.assert_called_once_with(0)

    async def test_get_n_inferences_repository_error(
        self, klin_service, mock_repository
    ):
        """
        Проверяет проброс ошибки из репозитория.
        """
        count = 10
        mock_repository.get_first_n.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            await klin_service.get_n_imferences(count)
