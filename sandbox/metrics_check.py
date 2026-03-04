from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.analyze_safetensors_file import SafetensorsFileActions


class MetricCheck:
    def __init__(self):
        self.STFA = SafetensorsFileActions()

    def detailed_model_analysis(self, model, model_name):
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
            layer_name = name.split(".")[0] if "." in name else name
            layer_stats.append(
                {
                    "layer": layer_name,
                    "name": name,
                    "shape": tuple(param.shape),
                    "parameters": param_count,
                    "trainable": param.requires_grad,
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                }
            )

        print(f"📊 Общее количество параметров: {total_params:,}")
        print(f"🎯 Обучаемых параметров: {trainable_params:,}")
        print(f"📈 Процент обучаемых: {(trainable_params / total_params) * 100:.2f}%")

        # Анализ по типам слоев
        layer_df = pd.DataFrame(layer_stats)
        layer_summary = (
            layer_df.groupby("layer")
            .agg({"parameters": "sum", "trainable": "mean"})
            .sort_values("parameters", ascending=False)
        )

        print("\n📋 Распределение по слоям:")
        for layer, row in layer_summary.head(10).iterrows():
            trainable_pct = row["trainable"] * 100
            print(
                f"  {layer:20} {row['parameters']:>12,}params"
                f" ({trainable_pct:.1f}% trainable)"
            )

        return layer_df

    def create_basic_plots(self, model, model_name):
        """Создание простых графиков для анализа модели"""
        print(f"\n📊 СОЗДАНИЕ ГРАФИКОВ ДЛЯ: {model_name}")

        try:
            # График 1: Распределение весов классификатора
            classifier_weights = model.classifier.weight.data.cpu().flatten().numpy()
            classifier_bias = model.classifier.bias.data.cpu().numpy()

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # 1. Гистограмма весов
            ax1.hist(
                classifier_weights,
                bins=50,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            ax1.set_title(f"Распределение весов классификатора\n{model_name}")
            ax1.set_xlabel("Значение веса")
            ax1.set_ylabel("Количество")
            ax1.grid(True, alpha=0.3)

            # 2. Значения смещений
            classes = range(len(classifier_bias))
            bars = ax2.bar(classes, classifier_bias, color=["lightcoral", "lightgreen"])
            ax2.set_title("Смещения по классам")
            ax2.set_xlabel("Класс")
            ax2.set_ylabel("Значение смещения")
            ax2.set_xticks(classes)
            # Добавляем значения на столбцы
            for bar, value in zip(bars, classifier_bias, strict=False):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                )
            ax2.grid(True, alpha=0.3)

            # 3. Heatmap весов (упрощенный)
            weights_2d = model.classifier.weight.data.cpu().numpy()
            im = ax3.imshow(weights_2d, aspect="auto", cmap="coolwarm")
            ax3.set_title("Матрица весов классификатора")
            ax3.set_xlabel("Признаки (упрощенно)")
            ax3.set_ylabel("Классы")
            ax3.set_xticks([])  # Убираем подписи для упрощения
            plt.colorbar(im, ax=ax3)

            # 4. Сравнение статистик
            stats_data = {
                "Среднее": np.mean(classifier_weights),
                "Стд. откл.": np.std(classifier_weights),
                "Мин.": np.min(classifier_weights),
                "Макс.": np.max(classifier_weights),
            }

            ax4.bar(stats_data.keys(), stats_data.values(), color="lightsteelblue")
            ax4.set_title("Статистика весов")
            ax4.set_ylabel("Значение")
            # Добавляем значения на столбцы
            for i, (_key, value) in enumerate(stats_data.items()):
                ax4.text(i, value, f"{value:.4f}", ha="center", va="bottom")
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            print("✅ Графики успешно созданы!")

            # Вывод числовой статистики
            print(f"\n📈 СТАТИСТИКА МОДЕЛИ {model_name}:")
            print("   Веса классификатора:")
            print(f"- Среднее: {stats_data['Среднее']:.6f}")
            print(f"- Стандартное отклонение: {stats_data['Стд. откл.']:.6f}")
            print(f"- Диапазон: [{stats_data['Мин.']:.6f}, {stats_data['Макс.']:.6f}]")
            print(f"   Смещения по классам: {classifier_bias}")

        except Exception as e:
            print(f"❌ Ошибка при создании графиков: {e}")

    def run(self, path: Path):
        model_paths = self.STFA.find_safetensors_models(path)

        # Анализ всех найденных моделей с исправленной функцией
        print("🔄 ПЕРЕЗАПУСК АНАЛИЗА С ИСПРАВЛЕНИЕМ...")
        for _model_name, model_path in model_paths.items():
            self.STFA.analyze_safetensors_file_corrected(model_path)

        # Загрузка и анализ моделей
        models = {}
        for model_name, model_path in model_paths.items():
            print(f"\n📥 Загрузка модели: {model_name}")
            model = self.STFA.load_model_from_safetensors(model_path)
            if model is not None:
                models[model_name] = model

        # Анализ всех загруженных моделей
        model_stats = {}
        for model_name, model in models.items():
            stats_df = self.detailed_model_analysis(model, model_name)
            model_stats[model_name] = stats_df
