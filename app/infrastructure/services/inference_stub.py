"""
Lightweight API-side inference stub.
"""

from app.application.dto import KlinResultDto
from app.application.interfaces import IKlinInference
from app.models.klin import KlinModel


class ApiInferenceStub(IKlinInference):
    """
    Protects the API process from running local inference.
    """

    def unavailable_reason(self, model: KlinModel | None = None) -> str:
        """
        Returns a clear message about where inference should run.
        """

        klin_suffix = "" if model is None else f" for klin_id={model.id}"
        return (
            "Inference processor is not available in the API container. "
            f"Use the queue worker to process tasks{klin_suffix}."
        )

    async def analyze(self, model: KlinModel) -> KlinResultDto:
        """Fails fast if the API process is asked to run inference."""

        raise RuntimeError(self.unavailable_reason(model))
