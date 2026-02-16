import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def run_process(name, cmd, cwd=None):
    """Запуск процесса с минимальным выводом"""
    print(f"Запуск {name}...")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        cwd=cwd,
    )

    def log_reader():
        if process.stdout is None:
            return
        for line in process.stdout:
            line = line.strip()
            if line and any(
                keyword in line
                for keyword in [
                    "Загрузка модели",
                    "Модель загружена",
                    "Доступные классы",
                    "Uvicorn running",
                    "Документация API",
                    "Redoc",
                    "Ошибка",
                    "Error",
                    "Exception",
                    "Анализ видео",
                    "Длительность",
                    "Кадры",
                    "FPS",
                    "Результат",
                    "Время обработки",
                    "Видео:",
                    "Результат:",
                    "Обработка видео:",
                ]
            ):
                print(f"[{name}] {line}")

    threading.Thread(target=log_reader, daemon=True).start()
    return process


def main():
    src_dir = Path(__file__).parent
    web_dir = src_dir / "web"

    print("=" * 50)
    print("СИСТЕМА АНАЛИЗА ВИДЕО КЛИН")
    print("=" * 50)

    processes = []

    try:
        # Запуск API
        print("\n1. Запуск API сервера...")
        api_process = run_process("API", [sys.executable, "api.py"], cwd=str(src_dir))
        processes.append(api_process)

        # Ждем загрузки модели
        print("Ожидание загрузки модели...")
        time.sleep(10)

        # Запуск веб-интерфейса
        print("\n2. Запуск веб-интерфейса...")
        web_process = run_process("WEB", [sys.executable, "appp.py"], cwd=str(web_dir))
        processes.append(web_process)

        time.sleep(3)

        print("\n" + "=" * 50)
        print("СИСТЕМА ГОТОВА К РАБОТЕ")
        print("=" * 50)
        print("Веб-интерфейс: http://localhost:8082")
        print("API сервер:    http://localhost:8008")
        print("Документация:  http://localhost:8008/docs")
        print("\nИнструкция:")
        print("1. Откройте http://localhost:8082")
        print("2. Загрузите видео файл или введите URL")
        print("3. Нажмите 'Начать анализ'")
        print("\nДля остановки нажмите Ctrl+C")
        print("=" * 50)

        # Открываем браузер
        try:
            webbrowser.open("http://192.168.210.85:8082")
        except RuntimeWarning:
            print("Не удалось открыть браузер автоматически")

        # Главный цикл
        while True:
            time.sleep(1)
            for proc in processes:
                if proc.poll() is not None:
                    return

    except KeyboardInterrupt:
        print("\nОстановка системы...")
    finally:
        print("\nЗавершение работы...")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1)
        print("Система остановлена")


if __name__ == "__main__":
    main()
