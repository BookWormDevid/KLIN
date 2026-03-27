import importlib.util
import re
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "helpers" / "sync_env.py"
MODULE_SPEC = importlib.util.spec_from_file_location("sync_env_module", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
SYNC_ENV_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = SYNC_ENV_MODULE
MODULE_SPEC.loader.exec_module(SYNC_ENV_MODULE)
sync_env_file = SYNC_ENV_MODULE.sync_env_file


def extract_secret(env_text: str) -> str:
    match = re.search(r"^AIRFLOW_SECRET_KEY=(.+)$", env_text, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_sync_env_file_creates_env_and_generates_airflow_secret(tmp_path) -> None:
    template_path = tmp_path / "example.env"
    env_path = tmp_path / ".env"

    template_path.write_text(
        "DEBUG=1\n"
        "AIRFLOW_SECRET_KEY=change_me_long_random_string\n"
        "POSTGRES_USER=klin_user_change_me\n",
        encoding="utf-8",
    )

    result = sync_env_file(template_path, env_path)
    env_text = env_path.read_text(encoding="utf-8")

    assert result.created is True
    assert result.added_keys == []
    assert result.generated_keys == ["AIRFLOW_SECRET_KEY"]
    assert "DEBUG=1" in env_text
    assert "POSTGRES_USER=klin_user_change_me" in env_text
    assert extract_secret(env_text) != "change_me_long_random_string"


def test_sync_env_file_backfills_missing_keys_without_overwriting_values(
    tmp_path,
) -> None:
    template_path = tmp_path / "example.env"
    env_path = tmp_path / ".env"

    template_path.write_text(
        "DEBUG=1\n"
        "AIRFLOW_SECRET_KEY=change_me_long_random_string\n"
        "POSTGRES_USER=from_template\n",
        encoding="utf-8",
    )
    env_path.write_text("DEBUG=0\n", encoding="utf-8")

    result = sync_env_file(template_path, env_path)
    env_text = env_path.read_text(encoding="utf-8")

    assert result.created is False
    assert result.added_keys == ["AIRFLOW_SECRET_KEY", "POSTGRES_USER"]
    assert result.generated_keys == ["AIRFLOW_SECRET_KEY"]
    assert "DEBUG=0" in env_text
    assert "POSTGRES_USER=from_template" in env_text
    assert extract_secret(env_text) != "change_me_long_random_string"
