"""Helpers for lazy package exports."""

from importlib import import_module
from typing import Any


def load_lazy_export(
    *,
    package_name: str,
    export_name: str,
    exports: dict[str, str],
) -> Any:
    """Import and return one lazily exported symbol."""

    module_name = exports.get(export_name)
    if module_name is None:
        raise AttributeError(
            f"module {package_name!r} has no attribute {export_name!r}"
        )

    module = import_module(module_name)
    return getattr(module, export_name)
