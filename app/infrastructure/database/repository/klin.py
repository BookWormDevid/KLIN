from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.interfaces import IKlinRepository
from app.models.klin import KlinModel


@dataclass
class KlinRepository(IKlinRepository):
    session: async_sessionmaker[AsyncSession]

    async def get_by_id(self, klin_id: UUID) -> KlinModel:
        async with self.session() as session:
            stmt = select(KlinModel).where(KlinModel.id == klin_id)
            result = await session.execute(stmt)

            klin = result.scalar_one_or_none()
            if klin is None:
                raise Exception(f"MAE with id {klin_id} does not exist")

            return klin

    async def create(self, model: KlinModel) -> KlinModel:
        async with self.session() as session:
            async with session.begin():
                session.add(model)
            await session.refresh(model)
            return model

    async def update(self, model: KlinModel) -> None:
        async with self.session() as session:
            async with session.begin():
                await session.merge(model)
            await session.commit()

    async def get_first_n(self, count: int) -> list[KlinModel]:
        async with self.session() as session:
            mass_query = (
                select(KlinModel).order_by(KlinModel.created_at.desc()).limit(count)
            )
            imfers = await session.execute(mass_query)
            imfer_list: list[KlinModel] = list(imfers.scalars().all())
            if not imfer_list:
                raise ValueError
            return imfer_list
