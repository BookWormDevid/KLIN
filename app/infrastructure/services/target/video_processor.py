"""
Процессор для взаимодействия с приложением litestar
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from typing import Any, cast

import aiohttp
import async_timeout
import cv2
import msgspec
import numpy as np
from numpy.typing import NDArray

from app.application.dto import KlinResultDto
from app.application.interfaces import IKlinCallbackSender, IKlinInference
from app.infrastructure.helpers import (
    LoggingHelper,
    PipelineQueues,
    StreamProcessingContext,
    VideoProcessingStats,
    VideoStreamState,
)
from app.models.klin import KlinModel

from .runtime_processor import ProcessorConfig, build_processor_runtime


logger = logging.getLogger(__name__)


class InferenceProcessor(IKlinInference):
    """
    Процессор. Содержит методы подключения моделей,
    обработки видео с помощью videomae, yolo, x3d.
    """

    def __init__(self) -> None:
        self.processing = ProcessorConfig()
        self.logging = LoggingHelper()
        self.runtime = build_processor_runtime(self.processing)

    def _quick_x3d_check(self, video_path: str) -> dict[str, float]:
        cap = cv2.VideoCapture(video_path)
        frames_list: list[np.ndarray] = []

        while len(frames_list) < 16:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame)

        cap.release()

        if len(frames_list) == 0:
            raise ValueError(f"Видео не содержит кадров: {video_path}")
        if len(frames_list) < 16:
            raise ValueError(
                f"Недостаточно кадров для X3D. "
                f"Получено {len(frames_list)}, требуется 16"
            )

        logits = self.runtime.x3d_processor.infer_clip(frames_list)
        return self.runtime.business_processor.classify_x3d_logits(logits)

    async def _infer_yolo_batch_async(
        self, batch_imgs: np.ndarray
    ) -> list[NDArray[np.float32]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._infer_yolo_batch, batch_imgs)

    async def _infer_mae_probs_async(
        self, chunk_frames: list[np.ndarray]
    ) -> NDArray[np.float32]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._infer_mae_probs, chunk_frames)

    def _infer_yolo_batch(self, batch_imgs: np.ndarray) -> list[NDArray[np.float32]]:
        return self.runtime.yolo_processor.infer_batch(batch_imgs)

    async def _process_yolo_batch(self, state: VideoStreamState, fps: float) -> None:
        if not state.yolo_buffer:
            return

        batch_array, frame_indices = self._build_yolo_batch(state.yolo_buffer)
        batch_preds_list = await self._infer_yolo_batch_async(batch_array)

        for frame_idx, preds in zip(frame_indices, batch_preds_list, strict=False):
            timesteps = frame_idx / fps if fps > 0 else 0.0
            self._store_yolo_detections(state, preds, timesteps)

    def _infer_mae_probs(self, chunk_frames: list[np.ndarray]) -> NDArray[np.float32]:
        return self.runtime.mae_processor.infer_probs(chunk_frames)

    async def _predict_mae_chunk(
        self,
        *,
        chunk_frames: list[np.ndarray],
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> dict[str, Any]:
        probs = await self._infer_mae_probs_async(chunk_frames)
        return self.runtime.business_processor.build_mae_result(
            probs,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
        )

    async def _run_predict_mae_chunk(
        self,
        *,
        chunk_frames: list[np.ndarray],
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> dict[str, Any]:
        predict_mae_chunk = cast(Any, self._predict_mae_chunk)
        result: Any = predict_mae_chunk(
            chunk_frames=chunk_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
        )
        if inspect.isawaitable(result):
            return cast(dict[str, Any], await result)
        return cast(dict[str, Any], result)

    async def _yolo_pipeline(
        self, frame_queue: asyncio.Queue, state: VideoStreamState, fps: float
    ) -> None:
        while True:
            item = await frame_queue.get()
            if item is None:
                frame_queue.task_done()
                break

            frame_resized, frame_idx = item

            if frame_idx % self.processing.yolo_stride != 0:
                frame_queue.task_done()
                continue

            state.yolo_buffer.append((frame_idx, frame_resized))
            if len(state.yolo_buffer) >= self.processing.yolo_batch_size:
                await self._process_yolo_batch(state, fps)
                state.yolo_buffer.clear()

            frame_queue.task_done()

    async def _mae_pipeline(
        self, frame_queue: asyncio.Queue, state: VideoStreamState, fps: float
    ) -> None:
        while True:
            item = await frame_queue.get()
            if item is None:
                frame_queue.task_done()
                break

            frame_resized, frame_idx = item

            state.chunk_buffer.append(frame_resized)
            if len(state.chunk_buffer) == self.processing.chunk_size:
                state.mae_results.append(
                    await self._run_predict_mae_chunk(
                        chunk_frames=state.chunk_buffer,
                        start_frame=state.chunk_start_frame,
                        end_frame=frame_idx,
                        fps=fps,
                    )
                )
                state.chunk_buffer = []
                state.chunk_start_frame = frame_idx + 1

            frame_queue.task_done()

    async def _flush_partial_chunk(self, state: VideoStreamState, fps: float) -> None:
        if not state.chunk_buffer:
            return

        pad_count = self.processing.chunk_size - len(state.chunk_buffer)
        if pad_count > 0:
            state.chunk_buffer.extend([state.chunk_buffer[-1]] * pad_count)

        state.mae_results.append(
            await self._run_predict_mae_chunk(
                chunk_frames=state.chunk_buffer,
                start_frame=state.chunk_start_frame,
                end_frame=state.frame_idx - 1,
                fps=fps,
            )
        )
        state.chunk_buffer = []

    async def _flush_yolo_batch(self, state: VideoStreamState, fps: float) -> None:
        if state.yolo_buffer:
            await self._process_yolo_batch(state, fps)
            state.yolo_buffer.clear()

    def _build_yolo_batch(
        self, yolo_buffer: list[tuple[int, np.ndarray]]
    ) -> tuple[np.ndarray, list[int]]:
        frame_indices = [frame_idx for frame_idx, _ in yolo_buffer]
        batch_imgs = [
            self.runtime.prepare.prepare_yolo_frame_for_triton(frame)[0]
            for _, frame in yolo_buffer
        ]
        return np.stack(batch_imgs, axis=0), frame_indices

    def _parse_yolo_detection(self, pred: np.ndarray) -> tuple[int, list[float]] | None:
        return self.runtime.business_processor.parse_yolo_detection(pred)

    def _store_yolo_detections(
        self, state: VideoStreamState, preds: np.ndarray, timesteps: float
    ) -> None:
        for pred in preds:
            detection = self._parse_yolo_detection(pred)
            if detection is None:
                continue
            class_id, bbox = detection
            state.bbox_by_time[timesteps].append(bbox)
            state.detected_class_ids.add(class_id)

    def _resolve_detected_objects(self, detected_class_ids: set[int]) -> list[str]:
        return self.runtime.business_processor.resolve_detected_objects(
            detected_class_ids
        )

    def _create_stream_context(
        self,
        *,
        total_frames: int,
        fps: float,
    ) -> StreamProcessingContext:
        state = VideoStreamState()
        yolo_queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        mae_queue: asyncio.Queue = asyncio.Queue(maxsize=128)
        pipeline = PipelineQueues(
            yolo_queue=yolo_queue,
            mae_queue=mae_queue,
            yolo_task=asyncio.create_task(self._yolo_pipeline(yolo_queue, state, fps)),
            mae_task=asyncio.create_task(self._mae_pipeline(mae_queue, state, fps)),
        )
        stats = VideoProcessingStats(
            total_frames=total_frames,
            fps=fps,
            duration=total_frames / fps if fps > 0 else 0.0,
        )
        return StreamProcessingContext(
            pipeline=pipeline,
            state=state,
            stats=stats,
        )

    async def _queue_video_frame(
        self,
        context: StreamProcessingContext,
        frame: np.ndarray,
    ) -> None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.processing.frame_size)
        await context.pipeline.yolo_queue.put((frame_resized, context.stats.frame_idx))
        await context.pipeline.mae_queue.put((frame_resized, context.stats.frame_idx))
        context.stats.frame_idx += 1
        context.state.frame_idx = context.stats.frame_idx

    async def _close_stream_context(self, context: StreamProcessingContext) -> None:
        await context.pipeline.yolo_queue.put(None)
        await context.pipeline.mae_queue.put(None)

    async def _finalize_stream_context(
        self, context: StreamProcessingContext
    ) -> tuple[
        list[dict[str, Any]],
        dict[float, list[list[float]]],
        list[str],
        dict[str, Any],
    ]:
        pipeline_results = await asyncio.gather(
            context.pipeline.yolo_task,
            context.pipeline.mae_task,
            return_exceptions=True,
        )
        for result in pipeline_results:
            if isinstance(result, BaseException):
                raise result

        await self._flush_partial_chunk(context.state, context.stats.fps)
        await self._flush_yolo_batch(context.state, context.stats.fps)

        objects = self._resolve_detected_objects(context.state.detected_class_ids)
        video_info = self.logging.build_video_info(
            total_frames=context.stats.total_frames,
            fps=context.stats.fps,
            duration=context.stats.duration,
            frames_read=context.stats.frame_idx,
        )
        return (
            context.state.mae_results,
            dict(context.state.bbox_by_time),
            objects,
            video_info,
        )

    async def _process_video_stream(
        self, video_path: str
    ) -> tuple[
        list[dict[str, Any]],
        dict[float, list[list[float]]],
        list[str],
        dict[str, Any],
    ]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Невозможно открыть видео: {video_path}")

        ctx = self._create_stream_context(
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                await self._queue_video_frame(ctx, frame)
        finally:
            cap.release()
            await self._close_stream_context(ctx)

        return await self._finalize_stream_context(ctx)

    async def process_video_stream(
        self, video_path: str
    ) -> tuple[
        list[dict[str, Any]],
        dict[float, list[list[float]]],
        list[str],
        dict[str, Any],
    ]:
        """
        Public wrapper around the internal video stream pipeline.
        """

        return await self._process_video_stream(video_path)

    async def analyze(self, model: KlinModel) -> KlinResultDto:
        start_ts = time.time()
        video_name = os.path.basename(model.video_path)

        result_dict = self._quick_x3d_check(model.video_path)
        is_fight = list(result_dict.keys())[0]

        if is_fight == "False":
            return KlinResultDto(
                x3d=msgspec.json.encode(result_dict).decode("utf-8"),
                mae="[]",
                yolo="{}",
                all_classes=[],
                objects=[],
            )

        try:
            (
                mae_results,
                yolo_bbox,
                detected_objects,
                video_info,
            ) = await self._process_video_stream(model.video_path)

            all_classes = list({result["answer"] for result in mae_results})

            processing_time = time.time() - start_ts
            self.logging.log_processing(video_name, video_info, processing_time)

            return KlinResultDto(
                x3d=msgspec.json.encode(result_dict).decode("utf-8"),
                mae=msgspec.json.encode(mae_results).decode("utf-8"),
                yolo=msgspec.json.encode(yolo_bbox).decode("utf-8"),
                all_classes=all_classes,
                objects=detected_objects,
            )
        except Exception as exc:
            logger.exception("Ошибка обработки видео %s: %s", video_name, exc)
            raise


class KlinCallbackSender(IKlinCallbackSender):
    """
    Класс для отправки вывода результата
    """

    def build_payload(self, model: KlinModel) -> dict[str, Any]:
        """
        Builds the callback payload for the processed model.
        """

        return {
            "klin_id": str(model.id),
            "x3d": model.x3d,
            "mae": model.mae,
            "yolo": model.yolo,
            "objects": model.objects,
            "all_classes": model.all_classes,
            "state": model.state,
        }

    async def post_consumer(self, model: KlinModel) -> None:
        """
        Отправляет вывод результата в виде json.
        Если попыток больше чем 3 выдаёт ошибку и выдаёт ошибку.
        """

        if not model.response_url:
            return

        data = msgspec.json.encode(self.build_payload(model))

        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                async with async_timeout.timeout(30):  # 30 секунд таймаут
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            model.response_url,
                            data=data,
                            headers={"Content-Type": "application/json"},
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as resp:
                            if 200 <= resp.status < 300:
                                return

                            body = await resp.text()
                            raise RuntimeError(f"HTTP {resp.status}, body={body}")

            except Exception as exc:
                if attempt == max_attempts:
                    logger.error(
                        "Callback failed after %d attempts. "
                        "model.id=%s response_url=%s error=%s",
                        max_attempts,
                        model.id,
                        model.response_url,
                        exc,
                    )
                    return
