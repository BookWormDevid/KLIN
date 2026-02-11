@echo off
REM
start "Uvicorn" cmd /k "uv run -m uvicorn --host 0.0.0.0 --port 8000 app.presentation.litestar.run:app"
start "Faststream" cmd /k "uv run -m faststream run app.presentation.faststream.app:app"

REM
timeout /t 5 /nobreak >nul

REM
start "" "http://localhost:8000/api/docs"
