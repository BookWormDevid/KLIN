import asyncio
from uuid import uuid4


async def repository_roundtrip_smoke() -> None:
    from dishka import make_container

    from app.application.interfaces import IKlinRepository
    from app.ioc import get_worker_providers
    from app.models import KlinStreamState, ProcessingState

    container = make_container(*get_worker_providers())
    repo = container.get(IKlinRepository)
    camera_id = f"repo_test_camera_{uuid4().hex}"

    stream = KlinStreamState(
        camera_id=camera_id,
        camera_url=None,
        state=ProcessingState.PENDING,
    )

    created = await repo.create(stream)
    loaded = await repo.get_by_id_stream(created.id)

    assert created.id is not None
    assert loaded.id == created.id
    assert loaded.camera_id == camera_id


if __name__ == "__main__":
    asyncio.run(repository_roundtrip_smoke())
