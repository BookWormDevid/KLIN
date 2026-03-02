from pathlib import Path

from ultralytics import YOLO


# Путь к папке проекта
parent = Path(__file__).resolve().parent.parent

print(parent)
# Проверяем, существует ли видео
video_path = parent / "videos/fi004.mp4"
if not video_path.exists():
    raise FileNotFoundError(f"Видео не найдено: {video_path}")

# Загружаем модель (автоматически скачает, если нет)
model = YOLO("../models/yolov8x.pt")  # маленькая универсальная модель YOLO

# Делаем предсказание на видео
results = model.predict(
    source=video_path,
    show=True,  # показывать видео с предсказаниями
    conf=0.6,  # порог уверенности
    save=False,  # не сохранять результат, можно поставить True
)

print("Предсказание завершено!")
