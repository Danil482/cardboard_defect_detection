import pandas as pd
from scipy import stats
import itertools

# Список файлов с результатами (предполагаем, что они в текущей директории)
files = {
    'yolov9t': 'bootstrap_results_yolov9t.csv',
    'yolo11n': 'bootstrap_results_yolo11n.csv',
    'yolov10n': 'bootstrap_results_yolov10n.csv',
    'yolov8n': 'bootstrap_results_yolov8n.csv',
    'yolov8n_tuned': 'bootstrap_results_yolov8n_tuned.csv'
}

# Загружаем данные и извлекаем mAP50 для каждой модели
mAP50_groups = {}
for model, file in files.items():
    df = pd.read_csv(file)
    mAP50_groups[model] = df['mAP50'].values.tolist()  # Список mAP50 для модели

# Вычисление средних и стандартных отклонений
means = {model: pd.Series(data).mean() for model, data in mAP50_groups.items()}
stds = {model: pd.Series(data).std() for model, data in mAP50_groups.items()}

# Вывод таблицы средних и стандартных отклонений
print("Описательные статистики mAP50 для каждой модели:")
print("Модель\tСреднее\tСтандартное отклонение")
for model in means:
    print(f"{model}\t{means[model]:.4f}\t{stds[model]:.4f}")

# Тест Kruskal-Wallis
kruskal_stat, kruskal_p = stats.kruskal(*mAP50_groups.values())
print("\nРезультат теста Kruskal-Wallis:")
print(f"Статистика: {kruskal_stat:.4f}")
print(f"p-значение: {kruskal_p:.4e}")
if kruskal_p < 0.05:
    print("Есть статистически значимые различия между моделями (p < 0.05).")
else:
    print("Нет статистически значимых различий между моделями (p >= 0.05).")

# Пост-хок анализ (Mann-Whitney U-тест для пар с поправкой Bonferroni)
if kruskal_p < 0.05:
    models = list(mAP50_groups.keys())
    num_pairs = len(models) * (len(models) - 1) // 2
    alpha_corrected = 0.05 / num_pairs
    print(f"\nПост-хок анализ (Mann-Whitney U-тест, скорректированный alpha = {alpha_corrected:.4f}):")
    print("Пара моделей\tp-значение\tЗначимо?")
    for model1, model2 in itertools.combinations(models, 2):
        u_stat, p = stats.mannwhitneyu(mAP50_groups[model1], mAP50_groups[model2])
        significant = p < alpha_corrected
        print(f"{model1} vs {model2}\t{p:.4e}\t{'Да' if significant else 'Нет'}")