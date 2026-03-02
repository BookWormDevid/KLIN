# Default target
default: lint

lint:
	uv run ruff check --fix
	uv run ruff format
	uv run -m mypy .
	-uv run pylint app

start-api:
	uv run -m uvicorn --host 0.0.0.0 --port 8000 app.presentation.litestar.run:app

start-queue:
	uv run -m faststream run app.presentation.faststream.app:app
