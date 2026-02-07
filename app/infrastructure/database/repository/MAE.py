from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.interfaces import IMAERepository
from app.models.MAE import MAEModel


@dataclass
class MAERepository(IMAERepository):
    session: async_sessionmaker[AsyncSession]

    async def get_by_id(self, MAE_id: UUID) -> MAEModel:
        async with self.session() as session:
            stmt = select(MAEModel).where(MAEModel.id == MAE_id)
            result = await session.execute(stmt)

            MAE = result.scalar_one_or_none()
            if MAE is None:
                raise Exception(f"MAE with id {MAE_id} does not exist")

            return MAE

    async def create(self, model: MAEModel) -> MAEModel:
        async with self.session() as session:
            async with session.begin():
                session.add(model)
            await session.refresh(model)
            return model

    async def update(self, model: MAEModel) -> None:
        async with self.session() as session:
            async with session.begin():
                await session.merge(model)
            await session.commit()
