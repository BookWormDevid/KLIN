"""
Sync a target .env with keys from a template file without overwriting existing values.
"""

from __future__ import annotations

import argparse
import re
import secrets
from dataclasses import dataclass
from pathlib import Path


ENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
GENERATED_VALUE_FACTORIES = {
    "AIRFLOW_SECRET_KEY": lambda: secrets.token_urlsafe(48),
}


@dataclass(frozen=True)
class EnvEntry:
    """Represents a single KEY=value assignment from an env file."""

    key: str
    raw_line: str


@dataclass(frozen=True)
class SyncResult:
    """Describes what changed during sync."""

    created: bool
    added_keys: list[str]
    generated_keys: list[str]


def parse_env_entries(text: str) -> list[EnvEntry]:
    """Extract ordered env assignments while ignoring comments and blank lines."""

    entries: list[EnvEntry] = []
    for line in text.splitlines():
        match = ENV_LINE_RE.match(line)
        if match:
            entries.append(EnvEntry(key=match.group(1), raw_line=line))
    return entries


def ensure_trailing_newline(text: str) -> str:
    """Normalize text so appended lines land on their own line."""

    if text.endswith("\n"):
        return text
    return f"{text}\n"


def render_entry(entry: EnvEntry) -> tuple[str, bool]:
    """Render an env line, generating secure values for selected keys."""

    factory = GENERATED_VALUE_FACTORIES.get(entry.key)
    if factory is None:
        return entry.raw_line, False
    return f"{entry.key}={factory()}", True


def replace_generated_values(text: str) -> tuple[str, list[str]]:
    """Replace generated keys inside freshly created env files."""

    generated_keys: list[str] = []
    updated_text = text
    for key, factory in GENERATED_VALUE_FACTORIES.items():
        pattern = re.compile(rf"(?m)^{re.escape(key)}=.*$")
        if pattern.search(updated_text):
            updated_text = pattern.sub(f"{key}={factory()}", updated_text, count=1)
            generated_keys.append(key)
    return ensure_trailing_newline(updated_text), generated_keys


def sync_env_file(template_path: Path, env_path: Path) -> SyncResult:
    """Create or backfill the target env file from the template."""

    template_text = template_path.read_text(encoding="utf-8")
    template_entries = parse_env_entries(template_text)

    if not env_path.exists():
        created_text, created_generated_keys = replace_generated_values(template_text)
        env_path.write_text(created_text, encoding="utf-8")
        return SyncResult(
            created=True,
            added_keys=[],
            generated_keys=created_generated_keys,
        )

    env_text = env_path.read_text(encoding="utf-8")
    current_keys = {entry.key for entry in parse_env_entries(env_text)}
    missing_entries = [
        entry for entry in template_entries if entry.key not in current_keys
    ]

    if not missing_entries:
        return SyncResult(created=False, added_keys=[], generated_keys=[])

    lines_to_add: list[str] = []
    appended_generated_keys: list[str] = []
    for entry in missing_entries:
        rendered_line, generated = render_entry(entry)
        lines_to_add.append(rendered_line)
        if generated:
            appended_generated_keys.append(entry.key)

    updated_env = ensure_trailing_newline(env_text)
    updated_env += "\n# Added by make init-env from example.env\n"
    updated_env += "\n".join(lines_to_add)
    updated_env += "\n"
    env_path.write_text(updated_env, encoding="utf-8")

    return SyncResult(
        created=False,
        added_keys=[entry.key for entry in missing_entries],
        generated_keys=appended_generated_keys,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Create or backfill a .env file from an env template.",
    )
    parser.add_argument(
        "--template", type=Path, required=True, help="Path to the template"
    )
    parser.add_argument(
        "--env", type=Path, required=True, help="Path to the target env file"
    )
    return parser


def main() -> int:
    """CLI entry point."""

    args = build_parser().parse_args()
    result = sync_env_file(args.template, args.env)

    if result.created:
        print(f"created {args.env} from {args.template}")
    elif result.added_keys:
        print(f"updated {args.env}: added {', '.join(result.added_keys)}")
    else:
        print(f"{args.env} already up to date")

    if result.generated_keys:
        print(f"generated secure value for {', '.join(result.generated_keys)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
