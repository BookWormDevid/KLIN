import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_ioc_import_does_not_require_worker_only_modules() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import importlib

        real_import = builtins.__import__

        def guarded_import(
            name,
            globals_dict=None,
            locals_dict=None,
            fromlist=(),
            level=0,
        ):
            if name == "cv2" or name.startswith("tritonclient"):
                raise ImportError(f"blocked import: {name}")
            return real_import(name, globals_dict, locals_dict, fromlist, level)

        builtins.__import__ = guarded_import

        module = importlib.import_module("app.ioc")
        providers = module.get_api_providers()
        assert providers
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
