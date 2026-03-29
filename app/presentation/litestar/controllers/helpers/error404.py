"""
Хелперы для litestar
"""

from typing import NoReturn

from litestar.exceptions import HTTPException
from litestar.status_codes import HTTP_404_NOT_FOUND


class LitestarErrors:  # pylint: disable=( (too-few-public-methods))
    """
    Класс для хелперов
    """

    def raise_404(self, exc: Exception) -> NoReturn:
        """
        Если чего-то нет выдать ошибку 404
        """
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
