import importlib
import importlib.metadata as importlib_metadata
import sys
import uuid
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from litestar import Response, Router
from litestar.datastructures import UploadFile
from litestar.exceptions import HTTPException

from app.application.exceptions import KlinEnqueueError, KlinNotFoundError
from app.application.interfaces import IKlinVideoStorage
from app.application.mappers import to_klin_read_dto
from app.application.services import KlinService
from app.models import KlinModel, ProcessingState


HandlerFn = Callable[..., Awaitable[Any]]
LITESTAR_MODULES = (
    "app.presentation.litestar.auth",
    "app.presentation.litestar.controllers.v1.auth",
    "app.presentation.litestar.controllers.v1.klin",
    "app.presentation.litestar.controllers.v1",
    "app.presentation.litestar.controllers",
    "app.presentation.litestar.app",
    "app.presentation.litestar.run",
)


def patch_safe_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    real_distribution = importlib_metadata.distribution

    def safe_distribution(name: str) -> Any:
        if name == "attrs":
            return SimpleNamespace(version="21.3.0")
        distribution = real_distribution(name)
        version = getattr(distribution, "version", None)
        if not isinstance(version, str) or not version:
            return SimpleNamespace(version="0")
        return distribution

    monkeypatch.setattr(importlib_metadata, "distribution", safe_distribution)


def clear_litestar_modules() -> None:
    for module_name in LITESTAR_MODULES:
        sys.modules.pop(module_name, None)


def load_litestar_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any]:
    patch_safe_distribution(monkeypatch)
    clear_litestar_modules()
    controller_module = importlib.import_module(
        "app.presentation.litestar.controllers.v1.klin"
    )
    app_module = importlib.import_module("app.presentation.litestar.app")
    return controller_module, app_module


def get_handler_fn(handler: Any) -> HandlerFn:
    return cast(HandlerFn, handler.fn)


def make_controller(controller_cls: type[Any]) -> Any:
    return controller_cls(owner=Router(path="/", route_handlers=[]))


def clear_settings_cache(module: Any) -> None:
    module.app_settings.env_properties.clear()


def make_request(dependencies: dict[type[Any], object]) -> Any:
    container = AsyncMock()

    async def get(dependency_type: type[Any], *, component: str = "") -> object:
        assert component == ""
        return dependencies[dependency_type]

    container.get.side_effect = get
    return SimpleNamespace(state=SimpleNamespace(dishka_container=container))


def make_klin_model() -> KlinModel:
    return KlinModel(
        id=uuid.UUID(int=1),
        response_url="https://callback.example/result",
        video_path="s3://klin/uploads/clip.mp4",
        state=ProcessingState.FINISHED,
        x3d="violence",
        mae="fight",
        yolo="person",
        objects=["person"],
        all_classes=["person", "fight"],
    )


def test_build_object_key_uses_prefix_and_default_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller_module, _ = load_litestar_modules(monkeypatch)
    controller_cls = controller_module.KlinController
    fixed_uuid = uuid.UUID(int=7)

    monkeypatch.setattr(controller_module.uuid, "uuid4", lambda: fixed_uuid)
    monkeypatch.setenv("S3_KEY_PREFIX", "klin/uploads")
    clear_settings_cache(controller_module)

    assert (
        controller_cls._build_object_key("clip.MOV")
        == "klin/uploads/00000000-0000-0000-0000-000000000007.mov"
    )

    monkeypatch.setenv("S3_KEY_PREFIX", "")
    clear_settings_cache(controller_module)

    assert (
        controller_cls._build_object_key(None)
        == "00000000-0000-0000-0000-000000000007.mp4"
    )


@pytest.mark.anyio
async def test_file_upload_returns_dto_and_closes_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller_module, _ = load_litestar_modules(monkeypatch)
    controller_cls = controller_module.KlinController
    controller = make_controller(controller_cls)
    handler = get_handler_fn(controller_cls.file_upload)
    klin_service = AsyncMock()
    klin_video_storage = AsyncMock()
    klin_model = make_klin_model()
    upload = UploadFile(
        content_type="video/mp4",
        filename="clip.mov",
        file_data=b"video-bytes",
    )

    monkeypatch.setenv("S3_KEY_PREFIX", "klin/uploads")
    clear_settings_cache(controller_module)
    monkeypatch.setattr(
        controller_module.uuid,
        "uuid4",
        lambda: uuid.UUID("00000000-0000-0000-0000-000000000123"),
    )
    request = make_request(
        {
            KlinService: klin_service,
            IKlinVideoStorage: klin_video_storage,
        }
    )
    klin_video_storage.upload_fileobj.return_value = klin_model.video_path
    klin_service.klin_image.return_value = klin_model

    result = await handler(
        controller,
        data=upload,
        response_url="https://callback.example/result",
        request=request,
    )

    assert result == to_klin_read_dto(klin_model)
    klin_video_storage.upload_fileobj.assert_awaited_once()
    assert klin_video_storage.upload_fileobj.await_args.kwargs["object_key"].endswith(
        ".mov"
    )
    klin_service.klin_image.assert_awaited_once()
    upload_dto = klin_service.klin_image.await_args.args[0]
    assert upload_dto.video_path == klin_model.video_path
    assert upload_dto.response_url == "https://callback.example/result"
    assert upload.file.closed is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error_message", "status_code"),
    [
        ("File too large", 413),
        ("Unsupported content type", 400),
    ],
)
async def test_file_upload_translates_storage_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
    error_message: str,
    status_code: int,
) -> None:
    controller_module, _ = load_litestar_modules(monkeypatch)
    controller_cls = controller_module.KlinController
    controller = make_controller(controller_cls)
    handler = get_handler_fn(controller_cls.file_upload)
    klin_service = AsyncMock()
    klin_video_storage = AsyncMock()
    upload = UploadFile(
        content_type="video/mp4",
        filename="clip.mp4",
        file_data=b"video-bytes",
    )
    request = make_request(
        {
            KlinService: klin_service,
            IKlinVideoStorage: klin_video_storage,
        }
    )

    klin_video_storage.upload_fileobj.side_effect = ValueError(error_message)

    with pytest.raises(HTTPException) as exc_info:
        await handler(
            controller,
            data=upload,
            request=request,
        )

    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail == error_message
    klin_service.klin_image.assert_not_awaited()
    assert upload.file.closed is True


@pytest.mark.anyio
async def test_file_upload_cleans_up_object_on_enqueue_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller_module, _ = load_litestar_modules(monkeypatch)
    controller_cls = controller_module.KlinController
    controller = make_controller(controller_cls)
    handler = get_handler_fn(controller_cls.file_upload)
    klin_service = AsyncMock()
    klin_video_storage = AsyncMock()
    upload = UploadFile(
        content_type="video/mp4",
        filename="clip.mp4",
        file_data=b"video-bytes",
    )
    object_uri = "s3://klin/uploads/clip.mp4"
    request = make_request(
        {
            KlinService: klin_service,
            IKlinVideoStorage: klin_video_storage,
        }
    )

    klin_video_storage.upload_fileobj.return_value = object_uri
    klin_video_storage.delete.side_effect = RuntimeError("cleanup failed")
    klin_service.klin_image.side_effect = KlinEnqueueError("queue unavailable")

    with pytest.raises(HTTPException) as exc_info:
        await handler(
            controller,
            data=upload,
            request=request,
        )

    assert exc_info.value.status_code == 503
    assert "queue unavailable" in str(exc_info.value.detail)
    klin_video_storage.delete.assert_awaited_once_with(object_uri)


@pytest.mark.anyio
async def test_status_and_listing_handlers_return_expected_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller_module, _ = load_litestar_modules(monkeypatch)
    controller_cls = controller_module.KlinController
    controller = make_controller(controller_cls)
    klin_service = AsyncMock()
    klin_model = make_klin_model()
    request = make_request({KlinService: klin_service})

    get_status = get_handler_fn(controller_cls.get_inference_status)
    get_all = get_handler_fn(controller_cls.get_all)

    klin_service.get_inference_status.return_value = to_klin_read_dto(klin_model)
    klin_service.get_n_imferences.return_value = [klin_model]

    status_response = await get_status(
        controller,
        klin_id=klin_model.id,
        request=request,
    )
    list_response = await get_all(controller, request=request)

    assert isinstance(status_response, Response)
    assert status_response.content.id == klin_model.id
    assert isinstance(list_response, Response)
    assert [item.id for item in list_response.content] == [klin_model.id]


@pytest.mark.anyio
async def test_get_inference_status_maps_not_found_to_http_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller_module, _ = load_litestar_modules(monkeypatch)
    controller_cls = controller_module.KlinController
    controller = make_controller(controller_cls)
    klin_service = AsyncMock()
    get_status = get_handler_fn(controller_cls.get_inference_status)
    klin_id = uuid.uuid4()
    request = make_request({KlinService: klin_service})

    klin_service.get_inference_status.side_effect = KlinNotFoundError(klin_id)

    with pytest.raises(HTTPException) as exc_info:
        await get_status(controller, klin_id=klin_id, request=request)

    assert exc_info.value.status_code == 404
    assert str(klin_id) in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_health_and_readiness_handlers_cover_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller_module, _ = load_litestar_modules(monkeypatch)
    controller_cls = controller_module.KlinController
    controller = make_controller(controller_cls)
    klin_service = AsyncMock()
    request = make_request({KlinService: klin_service})

    health_check = get_handler_fn(controller_cls.health_check)
    readiness_check = get_handler_fn(controller_cls.readiness_check)

    assert await health_check(controller) == "healthy"

    ready_response = await readiness_check(controller, request=request)
    assert ready_response.content == {"status": "ready"}

    klin_service.get_n_imferences.side_effect = RuntimeError("database unavailable")

    with pytest.raises(HTTPException) as exc_info:
        await readiness_check(controller, request=request)

    assert exc_info.value.status_code == 503
    assert "database unavailable" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_issue_token_returns_signed_jwt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_safe_distribution(monkeypatch)
    clear_litestar_modules()
    auth_controller_module = importlib.import_module(
        "app.presentation.litestar.controllers.v1.auth"
    )
    auth_module = importlib.import_module("app.presentation.litestar.auth")
    controller_cls = auth_controller_module.AuthController
    controller = make_controller(controller_cls)
    handler = get_handler_fn(controller_cls.issue_token)

    monkeypatch.setenv("KLIN_SECRET", "bootstrap-secret-long-enough-for-tests")
    monkeypatch.setenv("JWT_SECRET", "jwt-signing-secret-long-enough-for-tests")
    monkeypatch.setenv("JWT_TOKEN_TTL_MINUTES", "15")
    clear_settings_cache(auth_controller_module)
    clear_settings_cache(auth_module)

    from app.application.dto import JWTLoginDto

    result = await handler(
        controller,
        data=JWTLoginDto(secret="bootstrap-secret-long-enough-for-tests"),
    )
    jwt_auth = auth_module.build_jwt_auth()
    token = jwt_auth.token_cls.decode(
        encoded_token=result.access_token,
        secret=jwt_auth.token_secret,
        algorithm=jwt_auth.algorithm,
    )

    assert result.token_type == "bearer"
    assert result.expires_in == 900
    assert token.sub == "klin-api"


@pytest.mark.anyio
async def test_issue_token_rejects_invalid_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_safe_distribution(monkeypatch)
    clear_litestar_modules()
    auth_controller_module = importlib.import_module(
        "app.presentation.litestar.controllers.v1.auth"
    )
    controller_cls = auth_controller_module.AuthController
    controller = make_controller(controller_cls)
    handler = get_handler_fn(controller_cls.issue_token)

    monkeypatch.setenv("KLIN_SECRET", "bootstrap-secret-long-enough-for-tests")
    clear_settings_cache(auth_controller_module)

    from app.application.dto import JWTLoginDto

    with pytest.raises(HTTPException) as exc_info:
        await handler(controller, data=JWTLoginDto(secret="wrong-secret"))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid credentials"


@pytest.mark.anyio
async def test_lifespan_connects_broker_and_closes_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, app_module = load_litestar_modules(monkeypatch)
    rabbit_broker = AsyncMock()
    container = AsyncMock()
    container.get.return_value = rabbit_broker
    app = SimpleNamespace(state=SimpleNamespace(dishka_container=container))

    async with app_module.lifespan(app):
        pass

    container.get.assert_awaited_once()
    rabbit_broker.connect.assert_awaited_once()
    container.close.assert_awaited_once()


def test_create_litestar_app_builds_configured_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, app_module = load_litestar_modules(monkeypatch)
    container = MagicMock()
    setup_dishka = MagicMock()

    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost")
    monkeypatch.setenv("KLIN_SECRET", "bootstrap-secret-long-enough-for-tests")
    clear_settings_cache(app_module)
    monkeypatch.setattr(app_module, "setup_litestar_dishka", setup_dishka)

    app = app_module.create_litestar_app(container, group_path=True)
    routes = {route.path for route in app.routes}
    plugin_names = {plugin.__class__.__name__ for plugin in app.plugins}

    assert app.request_max_body_size == 200 * 1024 * 1024
    assert app.cors_config.allow_origins == ["http://localhost"]
    assert app.openapi_config.path == "/api/docs"
    assert app.openapi_config.security == [{"BearerAuth": []}]
    assert "/api/v1/auth/token" in routes
    assert "/api/v1/Klin/upload" in routes
    assert "/metrics" in routes
    assert "/frontend" in routes
    assert "StructlogPlugin" in plugin_names
    setup_dishka.assert_called_once_with(container, app)


def test_run_module_creates_app_from_container(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_safe_distribution(monkeypatch)
    clear_litestar_modules()

    app_module = importlib.import_module("app.presentation.litestar.app")
    dishka_module = importlib.import_module("dishka")
    ioc_module = importlib.import_module("app.ioc")
    container = MagicMock()
    app = MagicMock()

    monkeypatch.setattr(ioc_module, "get_api_providers", lambda: ("provider",))
    monkeypatch.setattr(dishka_module, "make_async_container", lambda *args: container)
    monkeypatch.setattr(app_module, "create_litestar_app", lambda arg: app)

    run_module = importlib.import_module("app.presentation.litestar.run")

    assert run_module.container is container
    assert run_module.app is app
