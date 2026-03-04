"""
Smoke-тесты CLI для helpers/model_exporters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from helpers import model_exporters


def test_validate_args_requires_existing_videomae_dir(tmp_path: Path) -> None:
    """
    Валидация должна отклонять несуществующий каталог модели.
    """
    missing_dir = tmp_path / "missing-model-dir"
    parser = model_exporters.build_parser()
    args = parser.parse_args(["videomae", "--model-dir", str(missing_dir)])

    with pytest.raises(ValueError, match="Model directory not found"):
        model_exporters.validate_args(args)


def test_main_dispatches_videomae_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    main() должен маршрутизировать команду videomae в нужный exporter.
    """
    model_dir = tmp_path / "videomae"
    model_dir.mkdir()
    output = tmp_path / "model.onnx"
    called: dict[str, object] = {}

    def fake_export(**kwargs: object) -> None:
        called.update(kwargs)

    monkeypatch.setattr(model_exporters, "export_videomae_to_onnx", fake_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_exporters.py",
            "videomae",
            "--model-dir",
            str(model_dir),
            "--output",
            str(output),
            "--target-ir-version",
            "9",
        ],
    )

    model_exporters.main()

    assert called["model_dir"] == model_dir
    assert called["output_path"] == output
    assert called["target_ir_version"] == 9


def test_validate_args_rejects_non_positive_target_ir(tmp_path: Path) -> None:
    """
    Валидация должна отклонять невалидную IR-версию ONNX.
    """
    model_dir = tmp_path / "videomae"
    model_dir.mkdir()

    parser = model_exporters.build_parser()
    args = parser.parse_args(
        [
            "videomae",
            "--model-dir",
            str(model_dir),
            "--target-ir-version",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="--target-ir-version must be > 0"):
        model_exporters.validate_args(args)
