"""
Содержит методы для взаимодействия с базой данных
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.exceptions import KlinNotFoundError
from app.application.interfaces import IKlinRepository
from app.models.klin import KlinModel


@dataclass
class KlinRepository(IKlinRepository):
    """
    Класс для взаимодействия с базой данных
    """

    session: async_sessionmaker[AsyncSession]

    async def get_by_id(self, klin_id: UUID) -> KlinModel:
        """
        Получение всех столбцов по конкретному id
        """
        async with self.session() as session:
            query = select(KlinModel).where(KlinModel.id == klin_id).limit(1)
            klin = await session.scalar(query)
            if not klin:
                raise KlinNotFoundError(klin_id)
            return klin

    async def get_first_n(self, count: int) -> list[KlinModel]:
        """
        Получить n количество последних строк в бд
        """
        async with self.session() as session:
            mass_query = (
                select(KlinModel).order_by(KlinModel.created_at.desc()).limit(count)
            )
            imfers = await session.execute(mass_query)
            imfer_list: list[KlinModel] = list(imfers.scalars().all())
            return imfer_list

    async def create(self, model: KlinModel) -> KlinModel:
        """
        Создать транзакцию к бд
        """
        async with self.session() as session:
            async with session.begin():
                session.add(model)
            await session.refresh(model)
            return model

    async def update(self, model: KlinModel) -> None:
        """
        Обновить поля в бд
        """
        async with self.session() as session:
            async with session.begin():
                await session.merge(model)
            await session.commit()
