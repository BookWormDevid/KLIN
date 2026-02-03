import os
import safetensors

from transformers import VideoMAEForVideoClassification


class SafetensorsFileActions:
    @staticmethod
    def find_safetensors_models(base_dir):
        """Найти все модели с .safetensors файлами"""
        model_paths = {}

        for root, _dirs, files in os.walk(base_dir):
            if "model.safetensors" in files:
                checkpoint_name = os.path.basename(root)
                model_paths[checkpoint_name] = root

            # Также ищем в поддиректориях
            for file in files:
                if file.endswith(".safetensors") and file != "model.safetensors":
                    checkpoint_name = file.replace(".safetensors", "")
                    model_paths[checkpoint_name] = os.path.join(root, file)

        print(f"Найдено моделей: {len(model_paths)}")
        for name, path in model_paths.items():
            print(f"  - {name}: {path}")

        return model_paths

    @staticmethod
    def analyze_safetensors_file_corrected(model_path):
        """Исправленный анализ файла .safetensors"""
        print(f"\n🔍 Анализ: {model_path}")

        try:
            if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
                # Это отдельный файл
                with safetensors.safe_open(model_path, framework="pt") as f:
                    metadata = f.metadata()
                    tensors = list(f.keys())
                    print(f"  Количество тензоров: {len(tensors)}")
                    print(f"  Метаданные: {metadata}")

                    # Анализ размеров тензоров - ИСПРАВЛЕННАЯ ЧАСТЬ
                    total_params = 0
                    print("\n  Основные тензоры:")

                    for key in tensors[:15]:  # Показать первые 15
                        try:
                            # ПОПРАВКА: используем get_tensor вместо get_shape
                            tensor = f.get_tensor(key)
                            tensor_shape = tensor.shape
                            params = tensor.numel()
                            total_params += params
                            print(f"    {key}: {tensor_shape} ({params:,} параметров)")
                        except Exception as e:
                            print(f"    {key}: ошибка чтения - {e}")

                    print(f"  Всего параметров в показанных тензорах: {total_params:,}")

                    # Анализ самых больших тензоров
                    print("\n  🏆 Самые большие тензоры:")
                    tensor_sizes = []
                    for key in tensors:
                        try:
                            tensor = f.get_tensor(key)
                            tensor_sizes.append((key, tensor.shape, tensor.numel()))
                        except RuntimeWarning:
                            print("No keys... in tensors...")
                            continue

                    # Сортируем по размеру
                    tensor_sizes.sort(key=lambda x: x[2], reverse=True)
                    for key, shape, params in tensor_sizes[:10]:
                        print(f"    {key}: {shape} ({params:,} параметров)")

            elif os.path.isdir(model_path):
                # Это директория с model.safetensors
                safetensors_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    return safetensors_path
                else:
                    print("  ❌ model.safetensors не найден в директории")

        except Exception as e:
            print(f"  ❌ Ошибка анализа: {e}")

    @staticmethod
    def load_model_from_safetensors(model_path):
        """Загрузка полной модели с конфигом"""
        try:
            if os.path.isdir(model_path):
                # Загрузка из директории с конфигом
                model = VideoMAEForVideoClassification.from_pretrained(model_path)
                print("  ✅ Модель загружена из директории")
                return model
            else:
                # Для отдельных .safetensors файлов нужна дополнительная обработка
                print("  ⚠️  Отдельный .safetensors файл - нужен config")
                return None

        except Exception as e:
            print(f"  ❌ Ошибка загрузки: {e}")
            return None