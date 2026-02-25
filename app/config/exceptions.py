"""
Исключение если ключа не существует
"""


class EnvVariableNotExistError(Exception):
    """
    Класс исключения
    """

    def __init__(self, key: str) -> None:
        super().__init__(f"{key} does not exist in environment")
