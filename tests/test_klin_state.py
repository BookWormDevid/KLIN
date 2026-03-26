import asyncio

from dishka import make_container

from app.application.interfaces import IKlinRepository
from app.ioc import get_worker_providers
from app.models import KlinStreamState, ProcessingState


async def test_repository():
    container = make_container(*get_worker_providers())

    repo = container.get(IKlinRepository)

    stream = KlinStreamState(
        camera_id="repo_test_camera",
        camera_url=None,
        state=ProcessingState.PENDING,
    )

    created = await repo.create(stream)

    print("CREATED:", created.id)


if __name__ == "__main__":
    asyncio.run(test_repository())
