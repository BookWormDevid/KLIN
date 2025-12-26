# Default target
default: lint

lint:
	uv run ruff check --fix
	uv run ruff format
	uv run -m mypy .

