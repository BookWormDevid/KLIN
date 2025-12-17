from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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
MODEL_PATH = os.path.join(BASE_DIR, "models", "KLIN-model")
print(MODEL_PATH)
# Проверяем существование модели
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Модель не найдена по пути: {MODEL_PATH}")

try:
    # Используем ваш существующий VideoFolderClassifier, но переименуем для API
    classifier = VideoClassifier(MODEL_PATH)  # Используйте VideoFolderClassifier если он называется так
    print("✅ Классификатор инициализирован")
except Exception as e:
    print(f"❌ Ошибка инициализации классификатора: {e}")
    classifier = None


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
    try:
        # Создаем уникальное имя для временного файла
        temp_dir = tempfile.gettempdir()
        temp_filename = f"video_{uuid.uuid4().hex}{file_ext}"
        temp_filepath = os.path.join(temp_dir, temp_filename)

        # Сохраняем загруженный файл
        with open(temp_filepath, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        print(f"📥 Видео загружено: {file.filename} -> {temp_filepath}")

        # Используем ваш существующий метод predict_video
        result = classifier.predict_video(temp_filepath)

        # Удаляем временный файл
        os.remove(temp_filepath)

        # Проверяем на ошибки
        if result.get('predicted_class') == 'ERROR':
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка обработки видео: {result.get('error', 'Неизвестная ошибка')}"
            )

        # Форматируем ответ
        response = {
            "success": True,
            "filename": file.filename,
            "predicted_class": result['predicted_class'],
            "confidence": result['confidence'],
            "confidence_percent": round(result['confidence'] * 100, 2),
            "processing_info": {
                "frames_processed": result.get('num_frames', 0),
                "chunks_created": result.get('num_chunks', 0),
                "device": str(classifier.device)
            }
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        # Удаляем временный файл в случае ошибки
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.remove(temp_filepath)

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки видео: {str(e)}"
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

            with open(temp_filepath, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Классификация
            result = classifier.predict_video(temp_filepath)

            # Удаляем временный файл
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
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                os.remove(temp_filepath)

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


@app.post("/predict_from_url")
async def predict_from_url(url: str):
    """
    Классификация видео по URL
    """
    # В будущем можно добавить загрузку видео по URL
    return {
        "message": "Функционал в разработке",
        "url": url
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Запуск FastAPI сервера...")
    print(f"📖 Документация API: http://localhost:8000/docs")
    print(f"📖 Redoc: http://localhost:8000/redoc")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )