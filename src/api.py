from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import tempfile
import uuid
from typing import List
import pathlib

# Импортируем ваш существующий VideoClassifier
from predict import VideoClassifier

BASE_DIR = pathlib.Path(__file__).parent.parent

app = FastAPI(
    title="Video Classification API",
    description="API для классификации видеофайлов с использованием VideoMAE",
    version="1.0.0"
)

# Инициализация классификатора при запуске
MODEL_PATH = os.path.join(BASE_DIR, "videomae_results", "checkpoint-14172")
print(f"[API] Загрузка модели из: {MODEL_PATH}")

# Проверяем существование модели
if not os.path.exists(MODEL_PATH):
    print(f"[API] Ошибка: Модель не найдена по пути: {MODEL_PATH}")
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

try:
    # Используем ваш существующий VideoClassifier
    classifier = VideoClassifier(MODEL_PATH)
    print(f"[API] Модель загружена! Доступные классы: {classifier.model.config.id2label}")
except Exception as e:
    print(f"[API] Ошибка инициализации классификатора: {e}")
    classifier = None


class URLRequest(BaseModel):
    url: str


@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "Video Classification API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": classifier is not None,
        "available_classes": classifier.model.config.id2label if classifier else None
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "device": str(classifier.device) if classifier else None
    }


@app.get("/classes")
async def get_classes():
    """Получение списка всех классов модели"""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    return {
        "success": True,
        "classes": classifier.model.config.id2label,
        "num_classes": len(classifier.model.config.id2label)
    }


@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    """
    Классификация загруженного видео файла

    Поддерживаемые форматы: mp4, avi, mov, mkv, wmv, flv, webm
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    # Проверка расширения файла
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Разрешенные форматы: {', '.join(allowed_extensions)}"
        )

    # Создаем временный файл для загруженного видео
    temp_filepath = None
    try:
        # Создаем уникальное имя для временного файла
        temp_dir = tempfile.gettempdir()
        temp_filename = f"video_{uuid.uuid4().hex}{file_ext}"
        temp_filepath = os.path.join(temp_dir, temp_filename)

        # Сохраняем загруженный файл
        content = await file.read()
        with open(temp_filepath, "wb") as buffer:
            buffer.write(content)

        print(f"[API] Обработка видео: {file.filename}")

        # Используем ваш существующий метод predict_video
        result = classifier.predict_video(temp_filepath)

        # Удаляем временный файл
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)

        # Проверяем на ошибки
        if result.get('predicted_class') == 'ERROR':
            error_msg = result.get('error', 'Неизвестная ошибка')
            print(f"[API] Ошибка обработки: {error_msg}")
            
            return JSONResponse(
                content={
                    "success": False,
                    "error": error_msg,
                    "predicted_class": "ERROR",
                    "confidence": 0.0
                },
                status_code=500
            )

        # Форматируем ответ с нужной информацией
        confidence = result.get('confidence', 0.0)
        
        response = {
            "success": True,
            "filename": file.filename,
            "predicted_class": result['predicted_class'],
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "processing_info": {
                "processing_time_seconds": round(result.get('processing_time', 0), 2),
                "video_duration": round(result.get('video_duration', 0), 2),
                "total_frames": result.get('total_frames', 0),
                "video_fps": result.get('video_fps', 0),
                "device": str(classifier.device)
            }
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        # Удаляем временный файл в случае ошибки
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except:
                pass

        print(f"[API] Ошибка: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки видео: {str(e)}"
        )


@app.post("/predict_from_url")
async def predict_from_url(url_request: URLRequest):
    """
    Классификация видео по URL
    Поддерживает: YouTube, Vimeo, прямые ссылки на видеофайлы
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    url = url_request.url.strip()
    
    if not url:
        raise HTTPException(status_code=400, detail="URL не может быть пустым")
    
    if not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Некорректный URL. Должен начинаться с http:// или https://")
    
    try:
        print(f"[API] Обработка видео по URL: {url}")
        
        # Используем метод predict_video_from_url
        result = classifier.predict_video_from_url(url)

        # Проверяем на ошибки
        if result.get('predicted_class') == 'ERROR':
            error_msg = result.get('error', 'Неизвестная ошибка')
            print(f"[API] Ошибка обработки URL: {error_msg}")
            
            return JSONResponse(
                content={
                    "success": False,
                    "error": error_msg,
                    "predicted_class": "ERROR",
                    "confidence": 0.0
                },
                status_code=500
            )

        # Форматируем ответ
        confidence = result.get('confidence', 0.0)
        
        response = {
            "success": True,
            "url": url,
            "filename": result.get('video_name', 'video_from_url'),
            "predicted_class": result['predicted_class'],
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "processing_info": {
                "processing_time_seconds": round(result.get('processing_time', 0), 2),
                "video_duration": round(result.get('video_duration', 0), 2),
                "total_frames": result.get('total_frames', 0),
                "video_fps": result.get('video_fps', 0),
                "device": str(classifier.device),
                "source": "url"
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        print(f"[API] Ошибка обработки URL: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки видео по URL: {str(e)}"
        )


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Пакетная классификация нескольких видео файлов
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Не загружено ни одного файла")

    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    results = []

    for file in files:
        try:
            # Проверка расширения
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in allowed_extensions:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"Неподдерживаемый формат: {file_ext}"
                })
                continue

            # Создаем временный файл
            temp_dir = tempfile.gettempdir()
            temp_filename = f"video_{uuid.uuid4().hex}{file_ext}"
            temp_filepath = os.path.join(temp_dir, temp_filename)

            content = await file.read()
            with open(temp_filepath, "wb") as buffer:
                buffer.write(content)

            # Классификация
            result = classifier.predict_video(temp_filepath)

            # Удаляем временный файл
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)

            if result.get('predicted_class') == 'ERROR':
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": result.get('error', 'Неизвестная ошибка')
                })
            else:
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "predicted_class": result['predicted_class'],
                    "confidence": result['confidence'],
                    "confidence_percent": round(result['confidence'] * 100, 2)
                })

        except Exception as e:
            if 'temp_filepath' in locals() and temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass

            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    # Статистика
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    response = {
        "total_files": len(files),
        "successful": len(successful),
        "failed": len(failed),
        "results": results
    }

    return JSONResponse(content=response)


@app.get("/test")
async def test_endpoint():
    """Тестовый эндпоинт"""
    return {
        "status": "ok",
        "message": "API работает",
        "model_loaded": classifier is not None
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("Запуск FastAPI сервера...")
    print("Документация API: http://localhost:8000/docs")
    print("Redoc: http://localhost:8000/redoc")
    print("=" * 50)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )