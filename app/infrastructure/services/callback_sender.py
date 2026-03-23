"""
HTTP callback sender for processed KLIN tasks.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import async_timeout
import msgspec

from app.application.interfaces import IKlinCallbackSender
from app.models.klin import KlinModel


logger = logging.getLogger(__name__)


class KlinCallbackSender(IKlinCallbackSender):
    """Sends processed task results to an external callback endpoint."""

    def build_payload(self, model: KlinModel) -> dict[str, Any]:
        """Builds the callback payload for the processed model."""

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
        """Sends the callback if the processed model has a response URL."""

        if not model.response_url:
            return

        data = msgspec.json.encode(self.build_payload(model))
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                async with async_timeout.timeout(30):
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            model.response_url,
                            data=data,
                            headers={"Content-Type": "application/json"},
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as response:
                            if 200 <= response.status < 300:
                                return

                            body = await response.text()
                            raise RuntimeError(f"HTTP {response.status}, body={body}")
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
