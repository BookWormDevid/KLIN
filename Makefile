# Default target
default: lint

lint:
	uv run ruff check --fix
	uv run ruff format
	uv run blake .
    uv run isort .
    uv run flake8 .
    uv run pylint .
	uv run -m mypy .
