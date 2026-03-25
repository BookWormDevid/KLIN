"""
Тесты бизнес-логики
"""

import os
import tempfile
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.dto import KlinProcessDto, KlinReadDto, KlinUploadDto
from app.application.exceptions import KlinEnqueueError, KlinNotFoundError
from app.application.services.klin import KlinService
from app.models import KlinModel, ProcessingState


PerformKlinContext = tuple[
    KlinService,
    AsyncMock,
    AsyncMock,
    AsyncMock,
    AsyncMock,
    KlinModel,
]


# Фикстуры — общие заготовки для всех тестов
@pytest.fixture(name="mock_repository")
def fixture_mock_repository() -> AsyncMock:
    """Мок репозитория — имитирует работу с базой данных."""
    return AsyncMock()


@pytest.fixture(name="mock_inference_service")
def fixture_mock_inference_service() -> AsyncMock:
    """Мок сервиса инференса — имитирует запуск YOLO + MAE."""
    return AsyncMock()


@pytest.fixture(name="mock_process_producer")
def fixture_mock_process_producer() -> AsyncMock:
    """Мок продюсера сообщений в очередь (например, RabbitMQ / Kafka)."""
    return AsyncMock()


@pytest.fixture(name="mock_callback_sender")
def fixture_mock_callback_sender() -> AsyncMock:
    """Мок отправителя колбэков на внешний URL."""
    return AsyncMock()


@pytest.fixture(name="mock_video_storage")
def fixture_mock_video_storage() -> AsyncMock:
    """РњРѕРє S3-compatible storage for uploaded videos."""
    return AsyncMock()


@pytest.fixture(name="klin_service")
def fixture_klin_service(
    mock_repository: AsyncMock,
    mock_inference_service: AsyncMock,
    mock_process_producer: AsyncMock,
    mock_callback_sender: AsyncMock,
    mock_video_storage: AsyncMock,
) -> KlinService:
    """
    Фикстура основного сервиса KlinService.
    Все зависимости заменены моками → изолированные тесты.
    """
    return KlinService(
        _klin_repository=mock_repository,
        _klin_inference_service=mock_inference_service,
        _klin_process_producer=mock_process_producer,
        _klin_callback_sender=mock_callback_sender,
        _klin_video_storage=mock_video_storage,
    )


@pytest.fixture(name="sample_klin_model")
def fixture_sample_klin_model() -> KlinModel:
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


@pytest.fixture(name="perform_klin_context")
def fixture_perform_klin_context(
    klin_service: KlinService,
    mock_repository: AsyncMock,
    mock_inference_service: AsyncMock,
    mock_callback_sender: AsyncMock,
    mock_video_storage: AsyncMock,
    sample_klin_model: KlinModel,
) -> PerformKlinContext:
    """Общий набор зависимостей для тестов perform_klin."""
    sample_klin_model.state = ProcessingState.PROCESSING
    mock_repository.claim_for_processing.return_value = sample_klin_model
    return (
        klin_service,
        mock_repository,
        mock_inference_service,
        mock_callback_sender,
        mock_video_storage,
        sample_klin_model,
    )


# Тесты метода klin_image (создание записи + отправка в очередь)
@pytest.mark.anyio
class TestKlinImage:
    """Группа тестов для метода klin_image
    загрузка видео и постановка задачи в очередь.
    """

    async def test_klin_image_success(
        self,
        klin_service: KlinService,
        mock_repository: AsyncMock,
        mock_process_producer: AsyncMock,
    ) -> None:
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
        self,
        klin_service: KlinService,
        mock_repository: AsyncMock,
        mock_process_producer: AsyncMock,
    ) -> None:
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

    async def test_klin_image_queue_publish_error_marks_task_error(
        self,
        klin_service: KlinService,
        mock_repository: AsyncMock,
        mock_process_producer: AsyncMock,
        mock_callback_sender: AsyncMock,
    ):
        """
        Если публикация в очередь не удалась, задача не должна зависнуть в PENDING.
        """
        upload_dto = KlinUploadDto(video_path="/tmp/test_video.mp4")
        created_klin = KlinModel(
            id=uuid.uuid4(),
            response_url=None,
            video_path=upload_dto.video_path,
            state=ProcessingState.PENDING,
        )
        mock_repository.create.return_value = created_klin
        mock_process_producer.send.side_effect = RuntimeError("broker down")

        with pytest.raises(KlinEnqueueError):
            await klin_service.klin_image(upload_dto)

        assert created_klin.state == ProcessingState.ERROR
        assert "Failed to enqueue klin_id" in (created_klin.mae or "")
        mock_repository.update.assert_called_once_with(created_klin)
        mock_callback_sender.post_consumer.assert_called_once_with(created_klin)

    async def test_klin_image_queue_publish_error_persists_before_callback(
        self,
        klin_service: KlinService,
        mock_repository: AsyncMock,
        mock_process_producer: AsyncMock,
        mock_callback_sender: AsyncMock,
    ) -> None:
        """
        callback.
        """
        upload_dto = KlinUploadDto(video_path="/tmp/test_video.mp4")
        created_klin = KlinModel(
            id=uuid.uuid4(),
            response_url="https://example.com/callback",
            video_path=upload_dto.video_path,
            state=ProcessingState.PENDING,
        )
        mock_repository.create.return_value = created_klin
        mock_process_producer.send.side_effect = RuntimeError("broker down")

        call_order: list[str] = []

        async def update_side_effect(model: KlinModel) -> None:
            assert model is created_klin
            call_order.append("update")

        async def callback_side_effect(model: KlinModel) -> None:
            assert model is created_klin
            call_order.append("callback")

        mock_repository.update.side_effect = update_side_effect
        mock_callback_sender.post_consumer.side_effect = callback_side_effect

        with pytest.raises(KlinEnqueueError):
            await klin_service.klin_image(upload_dto)

        assert call_order == ["update", "callback"]


@pytest.mark.anyio
class TestS3Cleanup:
    async def test_prepare_processing_video_removes_temp_file_when_download_fails(
        self,
        klin_service: KlinService,
        mock_video_storage: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source_uri = "s3://klin-videos/klin/uploads/broken-video.mp4"
        expected_local_path = "/temp/broken-video.mp4"
        mock_video_storage.download_to_path.side_effect = RuntimeError(
            "download failed"
        )

        monkeypatch.setattr(
            tempfile, "mkstemp", lambda suffix: (123, expected_local_path)
        )
        monkeypatch.setattr(os, "close", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda path: path == expected_local_path)
        mock_remove = MagicMock()
        monkeypatch.setattr(os, "remove", mock_remove)

        with pytest.raises(RuntimeError, match="download failed"):
            await klin_service._prepare_processing_video(source_uri)

        mock_video_storage.download_to_path.assert_awaited_once_with(
            source_uri=source_uri,
            destination_path=expected_local_path,
        )
        mock_remove.assert_called_once_with(expected_local_path)

    async def test_cleanup_video_artifacts_keeps_local_cleanup_when_s3_delete_fails(
        self,
        klin_service: KlinService,
        mock_video_storage: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        source_uri = "s3://klin-videos/klin/uploads/test-video.mp4"
        local_video_path = "temp/downloaded-video.mp4"
        mock_video_storage.delete.side_effect = RuntimeError("s3 delete failed")

        monkeypatch.setattr(os.path, "exists", lambda path: path == local_video_path)
        mock_remove = MagicMock()
        monkeypatch.setattr(os, "remove", mock_remove)

        await klin_service._cleanup_video_artifacts(
            source_video_path=source_uri,
            local_video_path=local_video_path,
        )

        mock_video_storage.delete.assert_awaited_once_with(source_uri)
        mock_remove.assert_called_once_with(local_video_path)
        assert "Failed to delete S3 object" in caplog.text


# Тесты метода perform_klin — основной этап обработки видео
@pytest.mark.anyio
class TestPerformKlin:
    """
    Группа тестов для метода perform_klin — запуск инференса и обновление статуса.
    """

    async def test_perform_klin_skips_when_claim_not_acquired(
        self,
        perform_klin_context: PerformKlinContext,
    ) -> None:
        """
        Если задача не захвачена (уже обрабатывается/финальная), обработка пропускается.
        """
        (
            klin_service,
            mock_repository,
            mock_inference_service,
            mock_callback_sender,
            _,
            sample_klin_model,
        ) = perform_klin_context
        mock_repository.claim_for_processing.return_value = None

        await klin_service.perform_klin(sample_klin_model.id)

        mock_inference_service.analyze.assert_not_called()
        mock_repository.update.assert_not_called()
        mock_callback_sender.post_consumer.assert_not_called()

    async def test_perform_klin_success_after_claim(
        self,
        perform_klin_context: PerformKlinContext,
    ) -> None:
        """
        Успешная обработка:
        - модель найдена
        - инференс прошёл
        - поля обновлены
        - статус → FINISHED
        - отправлен колбэк
        - обновление сохранено в БД
        """
        (
            klin_service,
            mock_repository,
            mock_inference_service,
            mock_callback_sender,
            _,
            sample_klin_model,
        ) = perform_klin_context

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

    async def test_perform_klin_persists_before_callback(
        self,
        perform_klin_context: PerformKlinContext,
    ) -> None:
        """
        callback.
        """
        (
            klin_service,
            mock_repository,
            mock_inference_service,
            mock_callback_sender,
            _,
            sample_klin_model,
        ) = perform_klin_context

        process_result = AsyncMock()
        process_result.x3d = "x3d"
        process_result.mae = "mae"
        process_result.yolo = "yolo"
        process_result.objects = []
        process_result.all_classes = []
        mock_inference_service.analyze.return_value = process_result

        call_order: list[str] = []

        async def update_side_effect(model: KlinModel) -> None:
            assert model is sample_klin_model
            call_order.append("update")

        async def callback_side_effect(model: KlinModel) -> None:
            assert model is sample_klin_model
            call_order.append("callback")

        mock_repository.update.side_effect = update_side_effect
        mock_callback_sender.post_consumer.side_effect = callback_side_effect

        await klin_service.perform_klin(sample_klin_model.id)

        assert call_order == ["update", "callback"]

    async def test_perform_klin_with_none_objects_after_claim(
        self,
        perform_klin_context: PerformKlinContext,
    ) -> None:
        """
        Проверяет, что при None в objects/all_classes
        сервис преобразует их в пустые списки.
        Это важно для консистентности данных в БД.
        """
        (
            klin_service,
            _,
            mock_inference_service,
            _,
            _,
            sample_klin_model,
        ) = perform_klin_context

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

    async def test_perform_klin_error_after_claim(
        self,
        perform_klin_context: PerformKlinContext,
    ) -> None:
        """
        Проверяет обработку ошибки инференса:
        - статус → ERROR
        - сообщение об ошибке сохраняется в mae
        - колбэк всё равно отправляется
        - обновление БД вызывается
        """
        (
            klin_service,
            mock_repository,
            mock_inference_service,
            mock_callback_sender,
            _,
            sample_klin_model,
        ) = perform_klin_context
        mock_inference_service.analyze.side_effect = ValueError("Inference failed")

        await klin_service.perform_klin(sample_klin_model.id)

        assert sample_klin_model.mae == "Inference failed"
        assert sample_klin_model.state == ProcessingState.ERROR
        mock_callback_sender.post_consumer.assert_called_once_with(sample_klin_model)
        mock_repository.update.assert_called_once_with(sample_klin_model)

    async def test_perform_klin_error_persists_before_callback(
        self,
        perform_klin_context: PerformKlinContext,
    ) -> None:
        (
            klin_service,
            mock_repository,
            mock_inference_service,
            mock_callback_sender,
            _,
            sample_klin_model,
        ) = perform_klin_context
        mock_inference_service.analyze.side_effect = ValueError("Inference failed")

        call_order: list[str] = []

        async def update_side_effect(model: KlinModel) -> None:
            assert model is sample_klin_model
            call_order.append("update")

        async def callback_side_effect(model: KlinModel) -> None:
            assert model is sample_klin_model
            call_order.append("callback")

        mock_repository.update.side_effect = update_side_effect
        mock_callback_sender.post_consumer.side_effect = callback_side_effect

        await klin_service.perform_klin(sample_klin_model.id)

        assert sample_klin_model.state == ProcessingState.ERROR
        assert call_order == ["update", "callback"]

    async def test_perform_klin_file_deletion_success(
        self,
        perform_klin_context: PerformKlinContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Проверяет удаление временного видео-файла после успешной обработки.
        """
        (
            klin_service,
            _,
            mock_inference_service,
            _,
            _,
            sample_klin_model,
        ) = perform_klin_context
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
        perform_klin_context: PerformKlinContext,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Проверяет, что при ошибке удаления файла:
        - обработка не падает
        - в лог пишется предупреждение
        """
        (
            klin_service,
            _,
            mock_inference_service,
            _,
            _,
            sample_klin_model,
        ) = perform_klin_context
        mock_inference_service.analyze.return_value = AsyncMock()

        mock_exists = MagicMock(return_value=True)
        mock_remove = MagicMock(side_effect=OSError("Permission denied"))

        monkeypatch.setattr(os.path, "exists", mock_exists)
        monkeypatch.setattr(os, "remove", mock_remove)

        await klin_service.perform_klin(sample_klin_model.id)

        assert "Failed to delete temp file" in caplog.text

    async def test_perform_klin_downloads_and_deletes_s3_artifact(
        self,
        perform_klin_context: PerformKlinContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        РџСЂРѕРІРµСЂСЏРµС‚ S3 workflow: download before inference and cleanup after.
        """

        (
            klin_service,
            _,
            mock_inference_service,
            _,
            mock_video_storage,
            sample_klin_model,
        ) = perform_klin_context
        sample_klin_model.video_path = "s3://klin-videos/klin/uploads/test-video.mp4"

        expected_local_path = "C:\\temp\\downloaded-video.mp4"
        seen_video_path: dict[str, str] = {}

        async def analyze_side_effect(model: KlinModel) -> AsyncMock:
            seen_video_path["value"] = model.video_path
            result = AsyncMock()
            result.x3d = "x3d"
            result.mae = "mae"
            result.yolo = "yolo"
            result.objects = []
            result.all_classes = []
            return result

        mock_inference_service.analyze.side_effect = analyze_side_effect

        monkeypatch.setattr(
            tempfile, "mkstemp", lambda suffix: (123, expected_local_path)
        )
        monkeypatch.setattr(os, "close", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda _path: True)
        mock_remove = MagicMock()
        monkeypatch.setattr(os, "remove", mock_remove)

        await klin_service.perform_klin(sample_klin_model.id)

        mock_video_storage.download_to_path.assert_awaited_once_with(
            source_uri=sample_klin_model.video_path,
            destination_path=expected_local_path,
        )
        mock_video_storage.delete.assert_awaited_once_with(sample_klin_model.video_path)
        assert seen_video_path["value"] == expected_local_path
        mock_remove.assert_called_once_with(expected_local_path)

    async def test_perform_klin_cleans_s3_artifacts_after_inference_error(
        self,
        perform_klin_context: PerformKlinContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        finally.
        """

        (
            klin_service,
            _,
            mock_inference_service,
            _,
            mock_video_storage,
            sample_klin_model,
        ) = perform_klin_context
        sample_klin_model.video_path = "s3://klin-videos/klin/uploads/test-video.mp4"
        mock_inference_service.analyze.side_effect = RuntimeError("Inference failed")

        expected_local_path = "temp/failed-video.mp4"
        monkeypatch.setattr(
            tempfile, "mkstemp", lambda suffix: (123, expected_local_path)
        )
        monkeypatch.setattr(os, "close", MagicMock())
        monkeypatch.setattr(os.path, "exists", lambda _path: True)
        mock_remove = MagicMock()
        monkeypatch.setattr(os, "remove", mock_remove)

        await klin_service.perform_klin(sample_klin_model.id)

        assert sample_klin_model.state == ProcessingState.ERROR
        mock_video_storage.download_to_path.assert_awaited_once_with(
            source_uri=sample_klin_model.video_path,
            destination_path=expected_local_path,
        )
        mock_video_storage.delete.assert_awaited_once_with(sample_klin_model.video_path)
        mock_remove.assert_called_once_with(expected_local_path)

    async def test_perform_klin_update_error(
        self,
        perform_klin_context: PerformKlinContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Проверяет устойчивость к ошибке обновления БД:
        - колбэк всё равно отправляется
        - ошибка логируется
        """
        (
            klin_service,
            mock_repository,
            mock_inference_service,
            _,
            _,
            sample_klin_model,
        ) = perform_klin_context
        mock_inference_service.analyze.return_value = AsyncMock()
        mock_repository.update.side_effect = Exception("Update failed")

        await klin_service.perform_klin(sample_klin_model.id)

        assert "Failed to persist klin state" in caplog.text

    async def test_perform_klin_attempts_callback_after_update_failure(
        self,
        perform_klin_context: PerformKlinContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (
            klin_service,
            mock_repository,
            mock_inference_service,
            mock_callback_sender,
            _,
            sample_klin_model,
        ) = perform_klin_context

        process_result = AsyncMock()
        process_result.x3d = "x3d"
        process_result.mae = "mae"
        process_result.yolo = "yolo"
        process_result.objects = []
        process_result.all_classes = []
        mock_inference_service.analyze.return_value = process_result

        call_order: list[str] = []

        async def update_side_effect(model: KlinModel) -> None:
            assert model is sample_klin_model
            call_order.append("update")
            raise RuntimeError("Update failed")

        async def callback_side_effect(model: KlinModel) -> None:
            assert model is sample_klin_model
            call_order.append("callback")

        mock_repository.update.side_effect = update_side_effect
        mock_callback_sender.post_consumer.side_effect = callback_side_effect

        await klin_service.perform_klin(sample_klin_model.id)

        assert call_order == ["update", "callback"]
        assert "Failed to persist klin state" in caplog.text


# Тесты метода get_inference_status
@pytest.mark.anyio
class TestGetInferenceStatus:
    """Тесты получения статуса обработки по ID."""

    async def test_get_inference_status_success(
        self, klin_service: KlinService, mock_repository, sample_klin_model
    ) -> None:
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

    async def test_get_inference_status_not_found(
        self, klin_service: KlinService, mock_repository: AsyncMock
    ) -> None:
        """
        Проверяет, что при отсутствии записи выбрасывается ValueError.
        """
        klin_id = uuid.uuid4()
        mock_repository.get_by_id.side_effect = KlinNotFoundError(klin_id)

        with pytest.raises(KlinNotFoundError, match=f"Klin {klin_id} not found"):
            await klin_service.get_inference_status(klin_id)


# Тесты метода get_n_inferences
@pytest.mark.anyio
class TestGetNInferences:
    """Тесты получения списка последних N задач."""

    async def test_get_n_inferences_success(
        self, klin_service: KlinService, mock_repository: AsyncMock
    ) -> None:
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

    async def test_get_n_inferences_empty(
        self, klin_service: KlinService, mock_repository: AsyncMock
    ) -> None:
        """
        Проверяет возврат пустого списка, если записей нет.
        """
        count = 5
        mock_repository.get_first_n.return_value = []

        result = await klin_service.get_n_imferences(count)

        assert result == []
        assert len(result) == 0

    async def test_get_n_inferences_zero(
        self, klin_service: KlinService, mock_repository: AsyncMock
    ) -> None:
        """
        Пограничный случай: запрашиваем 0 записей → пустой список.
        """
        count = 0
        mock_repository.get_first_n.return_value = []

        result = await klin_service.get_n_imferences(count)

        assert result == []
        mock_repository.get_first_n.assert_called_once_with(0)

    async def test_get_n_inferences_repository_error(
        self, klin_service: KlinService, mock_repository: AsyncMock
    ) -> None:
        """
        Проверяет проброс ошибки из репозитория.
        """
        count = 10
        mock_repository.get_first_n.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            await klin_service.get_n_imferences(count)
