import tempfile
import unittest
import uuid
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

from app.application.dto import KlinProcessDto, KlinResultDto, KlinUploadDto
from app.application.interfaces import (
    IKlinCallbackSender,
    IKlinInference,
    IKlinProcessProducer,
    IKlinRepository,
)
from app.application.services import KlinService
from app.models import KlinModel, ProcessingState


def make_model(video_path: str) -> KlinModel:
    model = KlinModel(
        response_url="http://localhost/callback",
        video_path=video_path,
        state=ProcessingState.PENDING,
    )
    model.id = uuid.uuid4()
    return model


class RepoFake(IKlinRepository):
    def __init__(self) -> None:
        self.get_by_id_m = AsyncMock()
        self.create_m = AsyncMock()
        self.update_m = AsyncMock()
        self.get_first_n_m = AsyncMock()

    async def get_by_id(self, klin_id: uuid.UUID) -> KlinModel:
        return cast(KlinModel, await self.get_by_id_m(klin_id))

    async def create(self, model: KlinModel) -> KlinModel:
        return cast(KlinModel, await self.create_m(model))

    async def update(self, model: KlinModel) -> None:
        await self.update_m(model)

    async def get_first_n(self, count: int) -> list[KlinModel]:
        return cast(list[KlinModel], await self.get_first_n_m(count))


class InferenceFake(IKlinInference):
    def __init__(self) -> None:
        self.analyze_m = AsyncMock()

    async def analyze(self, model: KlinModel) -> KlinResultDto:
        return cast(KlinResultDto, await self.analyze_m(model))


class ProducerFake(IKlinProcessProducer):
    def __init__(self) -> None:
        self.send_m = AsyncMock()

    async def send(self, data: KlinProcessDto) -> None:
        await self.send_m(data)


class CallbackFake(IKlinCallbackSender):
    def __init__(self) -> None:
        self.post_m = AsyncMock()

    async def post_consumer(self, model: KlinModel) -> None:
        await self.post_m(model)


class TestKlinSvc(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.repo = RepoFake()
        self.inf = InferenceFake()
        self.prod = ProducerFake()
        self.cb = CallbackFake()
        self.service = KlinService(
            _klin_repository=self.repo,
            _klin_inference_service=self.inf,
            _klin_process_producer=self.prod,
            _klin_callback_sender=self.cb,
        )

    @staticmethod
    def _tmp_model() -> tuple[KlinModel, str]:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        return make_model(path), path

    async def test_queue_on_create(self) -> None:
        created = make_model("tmp/video.mp4")
        self.repo.create_m.return_value = created
        payload = KlinUploadDto(
            response_url="http://localhost/callback",
            video_path="tmp/video.mp4",
        )

        result = await self.service.klin_image(payload)

        self.assertIs(result, created)
        self.repo.create_m.assert_awaited_once()
        self.prod.send_m.assert_awaited_once()
        await_args = self.prod.send_m.await_args
        if await_args is None:  # pragma: no cover
            self.fail("await_args should not be None after assert_awaited_once")
        arg = await_args.args[0]
        self.assertIsInstance(arg, KlinProcessDto)
        self.assertEqual(arg.klin_id, created.id)

    async def test_run_success(self) -> None:
        model, temp_path = self._tmp_model()
        self.repo.get_by_id_m.return_value = model
        self.inf.analyze_m.return_value = KlinResultDto(
            mae="violence",
            yolo="person",
            objects=["person"],
            all_classes=["Fighting", "Abuse"],
        )

        await self.service.perform_klin(model.id)

        self.assertEqual(model.state, ProcessingState.FINISHED)
        self.assertEqual(model.mae, "violence")
        self.assertEqual(model.yolo, "person")
        self.assertEqual(model.objects, ["person"])
        self.assertEqual(model.all_classes, ["Fighting", "Abuse"])
        self.cb.post_m.assert_awaited_once_with(model)
        self.repo.update_m.assert_awaited_once_with(model)
        self.assertFalse(Path(temp_path).exists())

    async def test_run_error(self) -> None:
        model, temp_path = self._tmp_model()
        self.repo.get_by_id_m.return_value = model
        self.inf.analyze_m.side_effect = RuntimeError("inference failed")

        await self.service.perform_klin(model.id)

        self.assertEqual(model.state, ProcessingState.ERROR)
        self.assertIn("inference failed", model.mae or "")
        self.cb.post_m.assert_awaited_once_with(model)
        self.repo.update_m.assert_awaited_once_with(model)
        self.assertFalse(Path(temp_path).exists())

    async def test_run_sets_empty_lists(self) -> None:
        model, _ = self._tmp_model()
        self.repo.get_by_id_m.return_value = model
        self.inf.analyze_m.return_value = KlinResultDto(
            mae="ok",
            yolo="ok",
            objects=None,
            all_classes=None,
        )

        await self.service.perform_klin(model.id)

        self.assertEqual(model.objects, [])
        self.assertEqual(model.all_classes, [])
        self.assertEqual(model.state, ProcessingState.FINISHED)

    async def test_run_swallow_update_error(self) -> None:
        model, _ = self._tmp_model()
        self.repo.get_by_id_m.return_value = model
        self.inf.analyze_m.return_value = KlinResultDto(
            mae="ok",
            yolo="ok",
            objects=["o"],
            all_classes=["c"],
        )
        self.repo.update_m.side_effect = RuntimeError("db write failed")

        await self.service.perform_klin(model.id)

        self.cb.post_m.assert_awaited_once_with(model)
        self.repo.update_m.assert_awaited_once_with(model)

    async def test_status_ok(self) -> None:
        model = make_model("tmp/video.mp4")
        model.mae = "done"
        model.yolo = "done"
        model.objects = ["person"]
        model.all_classes = ["Abuse"]
        model.state = ProcessingState.FINISHED
        self.repo.get_by_id_m.return_value = model

        dto = await self.service.get_inference_status(model.id)

        self.assertEqual(dto.id, model.id)
        self.assertEqual(dto.mae, "done")
        self.assertEqual(dto.state, ProcessingState.FINISHED)

    async def test_status_missing(self) -> None:
        self.repo.get_by_id_m.return_value = None

        with self.assertRaises(ValueError):
            await self.service.get_inference_status(uuid.uuid4())

    async def test_last_n(self) -> None:
        model = make_model("tmp/video.mp4")
        self.repo.get_first_n_m.return_value = [model]

        result = await self.service.get_n_imferences(5)

        self.repo.get_first_n_m.assert_awaited_once_with(5)
        self.assertEqual(result, [model])
