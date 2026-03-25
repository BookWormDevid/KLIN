"""
Содержит методы для взаимодействия с базой данных
"""

import uuid
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import msgspec
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.exceptions import KlinNotFoundError
from app.application.interfaces import IKlinRepository
from app.models.klin import (
    KlinMaeResult,
    KlinModel,
    KlinStreamState,
    KlinX3DResult,
    KlinYoloResult,
    ProcessingState,
)


@dataclass
class KlinRepository(IKlinRepository):
    """
    Класс для взаимодействия с базой данных
    """

    session: async_sessionmaker[AsyncSession]

    @staticmethod
    def _encode_payload(payload: dict) -> str:
        return msgspec.json.encode(payload).decode("utf-8")

    @staticmethod
    def _merge_unique(
        existing: list[str] | None, additions: list[str] | tuple[str, ...]
    ) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()

        for value in [*(existing or []), *additions]:
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)

        return merged

    async def save_yolo(self, event: KlinYoloResult) -> None:
        async with self.session() as session:
            async with session.begin():
                exists = await session.scalar(
                    select(KlinStreamState.id)
                    .where(KlinStreamState.id == event.stream_id)
                    .limit(1)
                )
                if exists is None:
                    raise ValueError(f"Stream not found: {event.stream_id}")

                stmt = (
                    insert(KlinYoloResult)
                    .values(
                        stream_id=event.stream_id,
                        camera_id=event.camera_id,
                        event_id=event.event_id,
                        frame_idx=event.frame_idx,
                        ts=event.ts,
                        detections=event.detections,
                    )
                    .on_conflict_do_nothing(index_elements=["event_id"])
                )

                await session.execute(stmt)

                objects = [d["label"] for d in event.detections]

                await session.execute(
                    update(KlinStreamState)
                    .where(KlinStreamState.id == event.stream_id)
                    .values(
                        objects=objects,
                        all_classes=list(set(objects)),
                    )
                )

    async def save_mae(self, event: KlinMaeResult) -> None:
        async with self.session() as session:
            async with session.begin():
                exists = await session.scalar(
                    select(KlinStreamState.id)
                    .where(KlinStreamState.id == event.stream_id)
                    .limit(1)
                )
                if exists is None:
                    raise ValueError(f"Stream not found: {event.stream_id}")

                stmt = (
                    insert(KlinMaeResult)
                    .values(
                        stream_id=event.stream_id,
                        camera_id=event.camera_id,
                        event_id=event.event_id,
                        label=event.label,
                        confidence=event.confidence,
                        start_ts=event.start_ts,
                        end_ts=event.end_ts,
                        probs=event.probs,
                    )
                    .on_conflict_do_nothing(index_elements=["event_id"])
                )

                await session.execute(stmt)

                # snapshot
                await session.execute(
                    update(KlinStreamState)
                    .where(KlinStreamState.id == event.stream_id)
                    .values(
                        last_mae_label=event.label,
                        last_mae_confidence=event.confidence,
                    )
                )

    async def save_x3d(self, event: KlinX3DResult) -> None:
        async with self.session() as session:
            async with session.begin():
                exists = await session.scalar(
                    select(KlinStreamState.id)
                    .where(KlinStreamState.id == event.stream_id)
                    .limit(1)
                )
                if exists is None:
                    raise ValueError(f"Stream not found: {event.stream_id}")

                stmt = (
                    insert(KlinX3DResult)
                    .values(
                        stream_id=event.stream_id,
                        camera_id=event.camera_id,
                        event_id=event.event_id,
                        label=event.label,
                        confidence=event.confidence,
                        ts=event.ts,
                    )
                    .on_conflict_do_nothing(index_elements=["event_id"])
                )

                await session.execute(stmt)

                await session.execute(
                    update(KlinStreamState)
                    .where(KlinStreamState.id == event.stream_id)
                    .values(
                        last_x3d_label=event.label,
                        last_x3d_confidence=event.confidence,
                    )
                )

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

    async def get_by_id_stream(self, stream_id: uuid.UUID) -> KlinStreamState:
        """
        Получение всех столбцов по конкретному id
        """
        async with self.session() as session:
            query = (
                select(KlinStreamState).where(KlinStreamState.id == stream_id).limit(1)
            )
            klin = await session.scalar(query)
            if not klin:
                raise KlinNotFoundError(stream_id)
            return klin

    async def claim_for_processing(self, klin_id: UUID) -> KlinModel | None:
        """
        Атомарно захватывает задачу для обработки.
        Переводит состояние из PENDING в PROCESSING.
        """
        async with self.session() as session:
            async with session.begin():
                claim_stmt = (
                    update(KlinModel)
                    .where(
                        KlinModel.id == klin_id,
                        KlinModel.state == ProcessingState.PENDING,
                    )
                    .values(state=ProcessingState.PROCESSING)
                    .returning(KlinModel.id)
                )
                claimed_id = await session.scalar(claim_stmt)

            if claimed_id is None:
                return None

            query = select(KlinModel).where(KlinModel.id == claimed_id).limit(1)
            klin = await session.scalar(query)
            return cast(KlinModel | None, klin)

    async def claim_for_processing_stream(
        self, klin_id: UUID
    ) -> KlinStreamState | None:
        """
        Атомарно захватывает задачу для обработки.
        Переводит состояние из PENDING в PROCESSING.
        """
        async with self.session() as session:
            async with session.begin():
                claim_stmt = (
                    update(KlinStreamState)
                    .where(
                        KlinStreamState.id == klin_id,
                        KlinStreamState.state == ProcessingState.PENDING,
                    )
                    .values(state=ProcessingState.PROCESSING)
                    .returning(KlinStreamState.id)
                )
                claimed_id = await session.scalar(claim_stmt)

            if claimed_id is None:
                return None

            query = (
                select(KlinStreamState).where(KlinStreamState.id == claimed_id).limit(1)
            )
            klin = await session.scalar(query)
            return cast(KlinStreamState | None, klin)

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

    async def create(
        self, model: KlinModel | KlinStreamState
    ) -> KlinModel | KlinStreamState:
        """
        Создать транзакцию к бд
        """
        async with self.session() as session:
            async with session.begin():
                session.add(model)
            await session.refresh(model)
            return model

    async def update(self, model: KlinModel | KlinStreamState) -> None:
        """
        Обновить поля в бд
        """
        async with self.session() as session:
            async with session.begin():
                await session.merge(model)
            await session.commit()

    async def mark_stopped(self, stream_id: uuid.UUID) -> None:
        async with self.session() as session:
            async with session.begin():
                await session.execute(
                    update(KlinStreamState)
                    .where(KlinStreamState.id == stream_id)
                    .values(state=ProcessingState.STOPPED)
                )
