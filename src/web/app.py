from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
import os
from pathlib import Path

app = FastAPI(
    title="КЛИН - Анализ видео",
    version="1.0"
)

# Статические файлы
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze/file")
async def analyze_video_file(file: UploadFile = File(...)):
    """Обработка видео файла"""
    try:
        # Проверка формата
        allowed = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed:
            return JSONResponse(
                content={"success": False, "error": f"Неподдерживаемый формат. Используйте: {', '.join(allowed)}"},
                status_code=400
            )
        
        # Отправка в API
        async with httpx.AsyncClient(timeout=120) as client:
            files = {'file': (file.filename, await file.read(), file.content_type)}
            response = await client.post("http://localhost:8000/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            return JSONResponse(content=result)
        else:
            error = response.json().get('detail', 'Ошибка сервера')
            return JSONResponse(
                content={"success": False, "error": error},
                status_code=response.status_code
            )
            
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": f"Ошибка: {str(e)}"},
            status_code=500
        )

@app.post("/api/analyze/url")
async def analyze_video_url(request: Request):
    try:
        data = await request.json()
        url = data.get("url", "").strip()
        
        if not url:
            return JSONResponse(
                content={"success": False, "error": "URL не может быть пустым"},
                status_code=400
            )
        
        if not url.startswith(('http://', 'https://')):
            return JSONResponse(
                content={"success": False, "error": "Некорректный URL. Должен начинаться с http:// или https://"},
                status_code=400
            )
        
        # Отправка в API с большим таймаутом
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                "http://localhost:8000/predict_from_url",
                json={"url": url},
                timeout=300
            )
        
        if response.status_code == 200:
            result = response.json()
            return JSONResponse(content=result)
        else:
            error = response.json().get('detail', 'Ошибка сервера')
            return JSONResponse(
                content={"success": False, "error": error},
                status_code=response.status_code
            )
            
    except httpx.TimeoutException:
        return JSONResponse(
            content={"success": False, "error": "Таймаут при загрузке видео. Попробуйте другое видео или загрузите файл."},
            status_code=408
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": f"Ошибка: {str(e)}"},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Проверка здоровья фронтенда"""
    return {
        "status": "healthy",
        "service": "КЛИН веб-интерфейс",
        "version": "1.0"
    }

@app.get("/debug/files")
async def debug_files():
    """Отладка файловой структуры"""
    static_dir = BASE_DIR / "static"
    templates_dir = BASE_DIR / "templates"
    
    return {
        "static_files": {
            "css": (static_dir / "css" / "style.css").exists(),
            "js": (static_dir / "js" / "script.js").exists()
        },
        "templates": {
            "index.html": (templates_dir / "index.html").exists()
        }
    }

@app.get("/test")
async def test_page(request: Request):
    """Тестовая страница"""
    return templates.TemplateResponse("test.html", {"request": request}) if (BASE_DIR / "templates" / "test.html").exists() else {
        "message": "Тестовая страница",
        "status": "ok"
    }

@app.get("/api/health")
async def api_health_check():
    """Проверка здоровья API"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/health")
            return response.json()
    except:
        return {"status": "unavailable", "error": "API не отвечает"}

if __name__ == "__main__":
    import uvicorn
    print("=" * 40)
    print("Веб-интерфейс КЛИН")
    print("Порт: 8080")
    print("Проверка здоровья: http://localhost:8080/health")
    print("Отладка: http://localhost:8080/debug/files")
    print("=" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")