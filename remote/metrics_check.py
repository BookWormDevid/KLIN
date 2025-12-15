import os
import json
import torch
import safetensors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

new_model_name = "videomae-base-finetuned-kinetics-repurposed-klin/checkpoint-200"

def find_safetensors_models(base_dir):
    """Найти все модели с .safetensors файлами"""
    model_paths = {}

    for root, dirs, files in os.walk(base_dir):
        if 'model.safetensors' in files:
            checkpoint_name = os.path.basename(root)
            model_paths[checkpoint_name] = root

        # Также ищем в поддиректориях
        for file in files:
            if file.endswith('.safetensors') and file != 'model.safetensors':
                checkpoint_name = file.replace('.safetensors', '')
                model_paths[checkpoint_name] = os.path.join(root, file)

    print(f"Найдено моделей: {len(model_paths)}")
    for name, path in model_paths.items():
        print(f"  - {name}: {path}")

    return model_paths

model_paths = find_safetensors_models(new_model_name)


def analyze_safetensors_file_corrected(model_path):
    """Исправленный анализ файла .safetensors"""
    print(f"\n🔍 Анализ: {model_path}")

    try:
        if os.path.isfile(model_path) and model_path.endswith('.safetensors'):
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
                print(f"\n  🏆 Самые большие тензоры:")
                tensor_sizes = []
                for key in tensors:
                    try:
                        tensor = f.get_tensor(key)
                        tensor_sizes.append((key, tensor.shape, tensor.numel()))
                    except:
                        continue

                # Сортируем по размеру
                tensor_sizes.sort(key=lambda x: x[2], reverse=True)
                for key, shape, params in tensor_sizes[:10]:
                    print(f"    {key}: {shape} ({params:,} параметров)")

        elif os.path.isdir(model_path):
            # Это директория с model.safetensors
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                return analyze_safetensors_file_corrected(safetensors_path)
            else:
                print(f"  ❌ model.safetensors не найден в директории")

    except Exception as e:
        print(f"  ❌ Ошибка анализа: {e}")


# Анализ всех найденных моделей с исправленной функцией
print("🔄 ПЕРЕЗАПУСК АНАЛИЗА С ИСПРАВЛЕНИЕМ...")
for model_name, model_path in model_paths.items():
    analyze_safetensors_file_corrected(model_path)

def load_model_from_safetensors(model_path):
    """Загрузка полной модели с конфигом"""
    try:
        if os.path.isdir(model_path):
            # Загрузка из директории с конфигом
            model = VideoMAEForVideoClassification.from_pretrained(model_path)
            print(f"  ✅ Модель загружена из директории")
            return model
        else:
            # Для отдельных .safetensors файлов нужна дополнительная обработка
            print(f"  ⚠️  Отдельный .safetensors файл - нужен config")
            return None

    except Exception as e:
        print(f"  ❌ Ошибка загрузки: {e}")
        return None


# Загрузка и анализ моделей
models = {}
for model_name, model_path in model_paths.items():
    print(f"\n📥 Загрузка модели: {model_name}")
    model = load_model_from_safetensors(model_path)
    if model is not None:
        models[model_name] = model


def detailed_model_analysis(model, model_name):
    """Детальный анализ архитектуры модели"""
    print(f"\n🏗️ Архитектурный анализ: {model_name}")
    print("=" * 50)

    # Основная информация
    print(f"Модель: {model.__class__.__name__}")
    print(f"Количество классов: {model.config.num_labels}")

    # Анализ параметров
    total_params = 0
    trainable_params = 0
    layer_stats = []

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

        # Собираем статистику по слоям
        layer_name = name.split('.')[0] if '.' in name else name
        layer_stats.append({
            'layer': layer_name,
            'name': name,
            'shape': tuple(param.shape),
            'parameters': param_count,
            'trainable': param.requires_grad,
            'mean': param.data.mean().item(),
            'std': param.data.std().item()
        })

    print(f"📊 Общее количество параметров: {total_params:,}")
    print(f"🎯 Обучаемых параметров: {trainable_params:,}")
    print(f"📈 Процент обучаемых: {(trainable_params / total_params) * 100:.2f}%")

    # Анализ по типам слоев
    layer_df = pd.DataFrame(layer_stats)
    layer_summary = layer_df.groupby('layer').agg({
        'parameters': 'sum',
        'trainable': 'mean'
    }).sort_values('parameters', ascending=False)

    print("\n📋 Распределение по слоям:")
    for layer, row in layer_summary.head(10).iterrows():
        trainable_pct = row['trainable'] * 100
        print(f"  {layer:20} {row['parameters']:>12,} params ({trainable_pct:.1f}% trainable)")

    return layer_df


# Анализ всех загруженных моделей
model_stats = {}
for model_name, model in models.items():
    stats_df = detailed_model_analysis(model, model_name)
    model_stats[model_name] = stats_df

import matplotlib.pyplot as plt
import numpy as np


def create_basic_plots(model, model_name):
    """Создание простых графиков для анализа модели"""
    print(f"\n📊 СОЗДАНИЕ ГРАФИКОВ ДЛЯ: {model_name}")

    try:
        # График 1: Распределение весов классификатора
        classifier_weights = model.classifier.weight.data.cpu().flatten().numpy()
        classifier_bias = model.classifier.bias.data.cpu().numpy()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Гистограмма весов
        ax1.hist(classifier_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title(f'Распределение весов классификатора\n{model_name}')
        ax1.set_xlabel('Значение веса')
        ax1.set_ylabel('Количество')
        ax1.grid(True, alpha=0.3)

        # 2. Значения смещений
        classes = range(len(classifier_bias))
        bars = ax2.bar(classes, classifier_bias, color=['lightcoral', 'lightgreen'])
        ax2.set_title('Смещения по классам')
        ax2.set_xlabel('Класс')
        ax2.set_ylabel('Значение смещения')
        ax2.set_xticks(classes)
        # Добавляем значения на столбцы
        for bar, value in zip(bars, classifier_bias):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{value:.4f}', ha='center', va='bottom')
        ax2.grid(True, alpha=0.3)

        # 3. Heatmap весов (упрощенный)
        weights_2d = model.classifier.weight.data.cpu().numpy()
        im = ax3.imshow(weights_2d, aspect='auto', cmap='coolwarm')
        ax3.set_title('Матрица весов классификатора')
        ax3.set_xlabel('Признаки (упрощенно)')
        ax3.set_ylabel('Классы')
        ax3.set_xticks([])  # Убираем подписи для упрощения
        plt.colorbar(im, ax=ax3)

        # 4. Сравнение статистик
        stats_data = {
            'Среднее': np.mean(classifier_weights),
            'Стд. откл.': np.std(classifier_weights),
            'Мин.': np.min(classifier_weights),
            'Макс.': np.max(classifier_weights)
        }

        ax4.bar(stats_data.keys(), stats_data.values(), color='lightsteelblue')
        ax4.set_title('Статистика весов')
        ax4.set_ylabel('Значение')
        # Добавляем значения на столбцы
        for i, (key, value) in enumerate(stats_data.items()):
            ax4.text(i, value, f'{value:.4f}', ha='center', va='bottom')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("✅ Графики успешно созданы!")

        # Вывод числовой статистики
        print(f"\n📈 СТАТИСТИКА МОДЕЛИ {model_name}:")
        print(f"   Веса классификатора:")
        print(f"     - Среднее: {stats_data['Среднее']:.6f}")
        print(f"     - Стандартное отклонение: {stats_data['Стд. откл.']:.6f}")
        print(f"     - Диапазон: [{stats_data['Мин.']:.6f}, {stats_data['Макс.']:.6f}]")
        print(f"   Смещения по классам: {classifier_bias}")

    except Exception as e:
        print(f"❌ Ошибка при создании графиков: {e}")
        print("⚠️  Пробуем упрощенный вариант...")
        create_simple_plot(model, model_name)


def create_simple_plot(model, model_name):
    """Упрощенный график если основной не работает"""
    try:
        weights = model.classifier.weight.data.cpu().flatten().numpy()
        bias = model.classifier.bias.data.cpu().numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(weights, bins=30, alpha=0.7, color='blue')
        plt.title(f'Веса классификатора - {model_name}')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.bar(['Класс 0', 'Класс 1'], bias, color=['red', 'green'])
        plt.title('Смещения классов')
        plt.ylabel('Значение')
        plt.grid(True, alpha=0.3)

        # Добавляем значения на столбцы
        for i, value in enumerate(bias):
            plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        print("✅ Упрощенные графики созданы!")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")


# Запускаем создание графиков
for model_name, model in models.items():
    create_basic_plots(model, model_name)